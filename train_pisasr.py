import os
import gc
import re
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from pisasr import CSDLoss, PiSASR
from src.my_utils.training_utils import parse_args  
from src.datasets.dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random
import logging

# Ortho-PiSA imports
from src.ortho_utils import (
    extract_lora_subspace,
    register_ortho_hooks,
    save_subspaces,
    load_subspaces,
)


def setup_logger(log_dir, filename="training_log.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # console handler (optional)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_training_state(path, optimizer, lr_scheduler, global_step, epoch,
                        training_phase, lambda_l2, lambda_lpips, lambda_csd):
    """Save full training state for resume."""
    state = {
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'training_phase': training_phase,  # 'pixel' or 'semantic'
        'lambda_l2': lambda_l2,
        'lambda_lpips': lambda_lpips,
        'lambda_csd': lambda_csd,
    }
    torch.save(state, path)


def load_training_state(path, optimizer, lr_scheduler, device='cuda'):
    """Load training state from checkpoint."""
    state = torch.load(path, map_location=device)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])
    return (
        state['global_step'],
        state['epoch'],
        state['training_phase'],
        state['lambda_l2'],
        state['lambda_lpips'],
        state['lambda_csd'],
    )


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # ✅ Setup the logger (only once on main process)
    if accelerator.is_main_process:
        logger = setup_logger(args.output_dir)
        logger.info("===== Training Started =====")
    else:
        logger = logging.getLogger("train_logger")
        logger.addHandler(logging.NullHandler())

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Seed set to: {args.seed}")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        logger.info("Output directories created.")

    net_pisasr = PiSASR(args)
    logger.info("Model initialized: PiSASR")

    if args.gradient_checkpointing:
        net_pisasr.unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    logger.info("Loss networks initialized (CSD, LPIPS).")

    # ============================================
    # Determine initial training phase
    # ============================================
    # Default: start with pixel phase
    initial_phase = 'pixel'
    global_step = 0
    start_epoch = 0
    lambda_l2 = args.lambda_l2
    lambda_lpips = 0
    lambda_csd = 0
    ortho_projector = None
    subspaces = None

    # Check if resuming from training state
    if args.resume_training_state and os.path.exists(args.resume_training_state):
        logger.info(f"Found training state file: {args.resume_training_state}")
        # We'll load optimizer/scheduler state after accelerator.prepare()
        state = torch.load(args.resume_training_state, map_location='cpu')
        global_step = state['global_step']
        start_epoch = state['epoch']
        initial_phase = state['training_phase']
        lambda_l2 = state['lambda_l2']
        lambda_lpips = state['lambda_lpips']
        lambda_csd = state['lambda_csd']
        logger.info(f"Will resume from step {global_step}, epoch {start_epoch}, phase: {initial_phase}")
    elif args.resume_ckpt:
        # Resume from model checkpoint without training state
        # Try to infer global_step from checkpoint filename
        match = re.search(r'model_(\d+)\.pkl', args.resume_ckpt)
        if match:
            global_step = int(match.group(1))
            logger.info(f"Inferred global_step={global_step} from checkpoint filename")

        # Determine phase based on global_step vs pix_steps
        if global_step >= args.pix_steps:
            initial_phase = 'semantic'
            lambda_l2 = args.lambda_l2_sem if args.lambda_l2_sem is not None else args.lambda_l2
            lambda_lpips = args.lambda_lpips
            lambda_csd = args.lambda_csd
            logger.info(f"Resuming in semantic phase (step {global_step} >= pix_steps {args.pix_steps})")
        else:
            initial_phase = 'pixel'
            logger.info(f"Resuming in pixel phase (step {global_step} < pix_steps {args.pix_steps})")

        logger.warning("No training_state file provided. Optimizer/scheduler will be reinitialized.")

    # Set up adapters based on phase
    if initial_phase == 'pixel' and global_step < args.pix_steps:
        net_pisasr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
        net_pisasr.set_train_pix()
        logger.info("Adapter set for pixel training phase.")
    else:
        # Semantic phase or resuming after pix_steps
        net_pisasr.unet.set_adapter([
            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
        net_pisasr.set_train_sem()
        initial_phase = 'semantic'
        logger.info("Adapter set for semantic training phase.")

    layers_to_opt = [p for n, p in net_pisasr.unet.named_parameters() if "lora" in n]
    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    logger.info(f"Optimizer & LR scheduler initialized with LR={args.learning_rate}")

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    logger.info(f"Datasets loaded: {len(dataset_train)} train, {len(dataset_val)} val")

    # RAM model
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
              pretrained_condition=None, image_size=384, vit='swin_l')
    RAM.eval().to("cuda", dtype=torch.float16)
    logger.info("RAM model loaded for captioning.")

    net_pisasr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pisasr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)

    # ============================================
    # Load optimizer/scheduler state AFTER accelerator.prepare()
    # ============================================
    if args.resume_training_state and os.path.exists(args.resume_training_state):
        state = torch.load(args.resume_training_state, map_location=accelerator.device)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])
        logger.info(f"Loaded optimizer and scheduler state from {args.resume_training_state}")

    # ============================================
    # Load SVD subspaces and register hooks if resuming in semantic phase
    # ============================================
    if initial_phase == 'semantic' and args.ortho_enabled:
        svd_path = os.path.join(args.output_dir, "checkpoints", "pixel_lora_subspaces.pt")
        if os.path.exists(svd_path):
            logger.info(f"Loading SVD subspaces from {svd_path}")
            subspaces = load_subspaces(svd_path)

            # Register gradient projection hooks
            logger.info("Registering gradient projection hooks for resumed semantic phase")
            unet_model = net_pisasr.module.unet if args.is_module else net_pisasr.unet
            ortho_projector = register_ortho_hooks(
                unet_model,
                subspaces,
                semantic_adapter_names=['default_encoder_sem', 'default_decoder_sem', 'default_others_sem'],
            )
            logger.info("Ortho hooks registered successfully")
        else:
            logger.warning(f"SVD subspaces file not found at {svd_path}. Ortho projection will NOT be applied!")

    # ============================================
    # Initialize trackers
    # ============================================
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # ============================================
    # Training Loop
    # ============================================
    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)
    progress_bar.update(global_step)  # Update progress bar to current step if resuming

    current_phase = initial_phase

    logger.info(f"Begin training from step {global_step}, phase: {current_phase}")

    for epoch in range(start_epoch, args.num_training_epochs):
        # Accumulator for expert utilization metrics across gradient accumulation steps
        # Reset each epoch to avoid data leaking between epochs
        expert_util_accumulated = []
        for step, batch in enumerate(dl_train):
            # Check if training should end
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(net_pisasr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]

                # ============================================
                # Phase Transition: Pixel -> Semantic
                # ============================================
                if global_step == args.pix_steps and current_phase == 'pixel':
                    # ORTHO-PISA: SVD Extraction (BEFORE phase transition)
                    if args.ortho_enabled:
                        logger.info("=" * 50)
                        logger.info("Ortho-PiSA: Extracting Pixel LoRA subspace via SVD")

                        # Get the actual unet (handle DDP wrapper)
                        unet_model = net_pisasr.module.unet if args.is_module else net_pisasr.unet

                        # Extract subspace from trained Pixel LoRA
                        subspaces = extract_lora_subspace(
                            unet_model,
                            adapter_names=['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'],
                            energy_threshold=args.svd_energy_threshold,
                        )

                        # Always save subspaces for resume capability
                        svd_path = os.path.join(args.output_dir, "checkpoints", "pixel_lora_subspaces.pt")
                        save_subspaces(subspaces, svd_path)

                    # Standard Phase Transition
                    if args.is_module:
                        net_pisasr.module.unet.set_adapter([
                            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                        net_pisasr.module.set_train_sem()
                    else:
                        net_pisasr.unet.set_adapter([
                            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                        net_pisasr.set_train_sem()

                    # ORTHO-PISA: Register hooks AFTER enabling Semantic LoRA
                    if args.ortho_enabled and subspaces is not None:
                        logger.info("Ortho-PiSA: Registering gradient projection hooks")
                        unet_model = net_pisasr.module.unet if args.is_module else net_pisasr.unet
                        ortho_projector = register_ortho_hooks(
                            unet_model,
                            subspaces,
                            semantic_adapter_names=['default_encoder_sem', 'default_decoder_sem', 'default_others_sem'],
                        )
                        logger.info("=" * 50)

                    # Set loss weights for semantic phase
                    lambda_l2 = args.lambda_l2_sem if args.lambda_l2_sem is not None else args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd = args.lambda_csd
                    current_phase = 'semantic'
                    logger.info(f"Switched to semantic phase: L2={lambda_l2}, LPIPS={lambda_lpips}, CSD={lambda_csd}")

                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_pisasr(x_src, x_tgt, batch=batch, args=args)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float()) * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips
                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args) * lambda_csd

                # Load balancing loss for MoLE
                loss_lb = torch.tensor(0.0, device=x_src.device)
                if args.use_load_balance_loss and args.num_experts_pix > 1:
                    if args.is_module:
                        loss_lb = net_pisasr.module.compute_load_balance_loss() * args.lambda_load_balance
                        expert_util = net_pisasr.module.compute_expert_utilization()
                    else:
                        loss_lb = net_pisasr.compute_load_balance_loss() * args.lambda_load_balance
                        expert_util = net_pisasr.compute_expert_utilization()

                    # Accumulate metrics across gradient accumulation steps
                    if expert_util:
                        expert_util_accumulated.append(expert_util)

                loss = loss_l2 + loss_lpips + loss_csd + loss_lb

                accelerator.backward(loss)

                # Reset gate stats IMMEDIATELY after backward to free memory
                if args.use_load_balance_loss and args.num_experts_pix > 1:
                    if args.is_module:
                        net_pisasr.module.reset_mole_gate_stats()
                    else:
                        net_pisasr.reset_mole_gate_stats()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Compute averaged expert utilization before resetting (for all processes)
                avg_expert_util = {}
                if expert_util_accumulated and args.use_load_balance_loss and args.num_experts_pix > 1:
                    keys = expert_util_accumulated[0].keys()
                    for key in keys:
                        values = [eu[key] for eu in expert_util_accumulated if key in eu]
                        if values:
                            avg_expert_util[key] = [
                                sum(v[i] for v in values) / len(values)
                                for i in range(len(values[0]))
                            ]
                    # Reset accumulator for next gradient accumulation window
                    expert_util_accumulated = []

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2": loss_l2.item(),
                        "loss_lpips": loss_lpips.item(),
                        "loss_csd": loss_csd.item(),
                        "loss_lb": loss_lb.item() if isinstance(loss_lb, torch.Tensor) else loss_lb,
                    }

                    # Format expert utilization for logging
                    util_str = ""
                    if avg_expert_util:
                        # Add to TensorBoard logs
                        for key, values in avg_expert_util.items():
                            for i, v in enumerate(values):
                                logs[f"expert_util/{key}/expert_{i}"] = v

                        # Format for console logging
                        def fmt_util(arr):
                            s = "[" + ", ".join(f"{v:.2f}" for v in arr) + "]"
                            if max(arr) > 0.5:  # Collapse warning threshold
                                s += " ⚠️"
                            return s

                        if 'global' in avg_expert_util:
                            util_str = f"\n  Expert Util Global: {fmt_util(avg_expert_util['global'])}"
                        for key in ['encoder_pix', 'decoder_pix', 'others_pix']:
                            if key in avg_expert_util:
                                util_str += f"\n  Expert Util {key}: {fmt_util(avg_expert_util[key])}"

                    logger.info(f"Step {global_step}: L2={logs['loss_l2']:.6f}, LPIPS={logs['loss_lpips']:.6f}, CSD={logs['loss_csd']:.6f}, LB={logs['loss_lb']:.6f}{util_str}")

                    # ============================================
                    # Save checkpoints (model + training state)
                    # ============================================
                    if global_step % args.checkpointing_steps == 1:
                        # Save model weights
                        model_path = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pisasr).save_model(model_path)

                        # Save training state (optimizer, scheduler, step, phase)
                        state_path = os.path.join(args.output_dir, "checkpoints", f"training_state_{global_step}.pt")
                        save_training_state(
                            state_path,
                            optimizer,
                            lr_scheduler,
                            global_step,
                            epoch,
                            current_phase,
                            lambda_l2,
                            lambda_lpips,
                            lambda_csd
                        )
                        logger.info(f"Checkpoint saved: {model_path}, {state_path}")

                    # evaluation
                    if global_step % args.eval_freq == 1:
                        logger.info(f"Running evaluation at step {global_step}")
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # get text prompts from LR
                                x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                batch_val["prompt"] = caption
                                # forward pass
                                x_tgt_pred, latents_pred, _, _ = accelerator.unwrap_model(net_pisasr)(x_src, x_tgt,
                                                                                                      batch=batch_val,
                                                                                                      args=args)
                                # save the output
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                else:
                                    pass
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

                    accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

    logger.info("===== Training Completed =====")

if __name__ == "__main__":
    args = parse_args()
    main(args)

