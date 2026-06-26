import os
import sys
import time
import random
import copy
from types import SimpleNamespace

# Use vendor/peft instead of pip-installed peft
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook
from src.models.degradation_features import DegradationFeatureExtractor


import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def initialize_unet(rank_pix, rank_sem, return_lora_module_names=False, pretrained_model_path=None, num_experts_pix=None, top_k_pix=None, num_shared_experts_pix=None, use_freq_gate=False, freq_dim=7):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_pix.append(n.replace(".weight",""))
                l_target_modules_decoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_pix.append(n.replace(".weight",""))
                l_modules_others_sem.append(n.replace(".weight",""))
                break


    # Pixel mixture of lora experts initialization (Approach 2: num_experts inside adapter)
    # num_experts_pix controls number of routed experts INSIDE each adapter
    # num_shared_experts_pix controls number of shared experts (always activated)
    # use_freq_gate enables frequency-based routing for better degradation handling
    num_experts = num_experts_pix if num_experts_pix is not None else 1
    top_k = top_k_pix if top_k_pix is not None else num_experts
    num_shared = num_shared_experts_pix if num_shared_experts_pix is not None else 0
    lora_conf_encoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian", target_modules=l_target_modules_encoder_pix, number_of_lora=num_experts, top_k_lora=top_k, num_shared_experts=num_shared, use_freq_gate=use_freq_gate, freq_dim=freq_dim)
    lora_conf_decoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian", target_modules=l_target_modules_decoder_pix, number_of_lora=num_experts, top_k_lora=top_k, num_shared_experts=num_shared, use_freq_gate=use_freq_gate, freq_dim=freq_dim)
    lora_conf_others_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian", target_modules=l_modules_others_pix, number_of_lora=num_experts, top_k_lora=top_k, num_shared_experts=num_shared, use_freq_gate=use_freq_gate, freq_dim=freq_dim)

    unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
    unet.add_adapter(lora_conf_others_pix, adapter_name="default_others_pix")
    
    
    # Semantic lora initialization
    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_modules_others_sem)
    unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    else:
        return unet


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        # Move to device and dtype BEFORE enabling xformers (xformers requires fp16/bf16 on newer GPUs)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                # Skip diffusers' fp32 test by setting processor directly
                from diffusers.models.attention_processor import XFormersAttnProcessor
                self.unet_fix.set_attn_processor(XFormersAttnProcessor())
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()


class PiSASR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        # Frequency-based routing settings
        self.use_freq_gate = getattr(args, 'use_freq_gate', False)
        self.freq_dim = getattr(args, 'freq_dim', 7)
        self.num_freq_bands = getattr(args, 'num_freq_bands', 4)

        # Initialize degradation feature extractor if using frequency-based routing
        if self.use_freq_gate:
            self.deg_feature_extractor = DegradationFeatureExtractor(num_bands=self.num_freq_bands).cuda()
        else:
            self.deg_feature_extractor = None

        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_pix, lora_unet_modules_decoder_pix, lora_unet_others_pix, \
                lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem, =\
                    initialize_unet(
                        rank_pix=args.lora_rank_unet_pix,
                        rank_sem=args.lora_rank_unet_sem,
                        pretrained_model_path=args.pretrained_model_path,
                        return_lora_module_names=True,
                        num_experts_pix=args.num_experts_pix,
                        top_k_pix=args.top_k_pix,
                        num_shared_experts_pix=getattr(args, 'num_shared_experts_pix', 0),
                        use_freq_gate=self.use_freq_gate,
                        freq_dim=self.freq_dim,
                    )

            self.lora_rank_unet_pix = args.lora_rank_unet_pix
            self.lora_rank_unet_sem = args.lora_rank_unet_sem
            self.num_experts_pix = args.num_experts_pix
            self.top_k_pix = args.top_k_pix
            self.num_shared_experts_pix = getattr(args, 'num_shared_experts_pix', 0)
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
                self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
                lora_unet_modules_encoder_pix, lora_unet_modules_decoder_pix, lora_unet_others_pix, \
                    lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_pix = stage1_args.lora_rank_unet_pix
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_sem
            pisasr = torch.load(args.resume_ckpt)
            self.load_ckpt_from_state_dict(pisasr)
        # unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

    def set_train_pix(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n:
                _p.requires_grad = True
            if "sem" in n:
                _p.requires_grad = False
    
    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n:
                _p.requires_grad = True
            if "pix" in n:
                _p.requires_grad = False

    def get_pix_adapter_names(self):
        """Return list of all pixel adapter names.
        With Approach 2 (MoLE), num_experts is INSIDE each adapter, so we always have 3 adapters.
        """
        return ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix']

    def get_sem_adapter_names(self):
        """Return list of all semantic adapter names"""
        return ['default_encoder_sem', 'default_decoder_sem', 'default_others_sem']

    def get_all_adapter_names(self):
        """Return list of all adapter names (pix + sem)"""
        return self.get_pix_adapter_names() + self.get_sem_adapter_names()

    def compute_load_balance_loss(self) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss for MoLE.

        This loss encourages balanced routing across experts by penalizing
        concentrated routing probabilities.

        Formula: L_lb = N * Σ(P_i²)
        - P_i: average routing probability to expert i across all tokens/layers
        - N: number of experts

        Loss range:
        - Minimum = 1.0 when perfectly balanced (P_i = 1/N for all i)
        - Maximum = N when all tokens route to single expert

        Returns:
            torch.Tensor: Scalar loss value with gradient
        """
        all_P = []
        num_experts = None

        for module in self.unet.modules():
            if hasattr(module, 'gate_weights_for_lb') and module.gate_weights_for_lb:
                for adapter_name, gate_list in module.gate_weights_for_lb.items():
                    for gw in gate_list:
                        if gw.numel() == 0:  # Skip empty tensors
                            continue
                        # gw shape: [batch, num_experts] or [batch, seq, num_experts]
                        # Flatten to [N, num_experts] then mean over N
                        P = gw.reshape(-1, gw.shape[-1]).mean(dim=0)  # [num_experts]
                        all_P.append(P)
                        num_experts = gw.shape[-1]

        if not all_P or num_experts is None or num_experts <= 1:
            device = next(self.unet.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Average P across all layers and all forward calls
        P_avg = torch.stack(all_P).mean(dim=0)  # [num_experts]

        # Load balancing loss: N * sum(P_i^2)
        lb_loss = num_experts * (P_avg ** 2).sum()

        return lb_loss

    def reset_mole_gate_stats(self):
        """Reset gate statistics after computing loss to free memory."""
        for module in self.unet.modules():
            if hasattr(module, 'gate_weights_for_lb'):
                module.gate_weights_for_lb = {}

    def compute_expert_utilization(self) -> dict:
        """
        Compute expert utilization metrics.

        Returns:
            dict: {
                'global': [P_0, P_1, ..., P_{N-1}],  # average prob per expert
                'encoder_pix': [P_0, P_1, ..., P_{N-1}],
                'decoder_pix': [P_0, P_1, ..., P_{N-1}],
                'others_pix': [P_0, P_1, ..., P_{N-1}],
            }
        """
        adapter_gates = {
            'default_encoder_pix': [],
            'default_decoder_pix': [],
            'default_others_pix': [],
        }

        num_experts = None

        for module in self.unet.modules():
            if hasattr(module, 'gate_weights_for_lb') and module.gate_weights_for_lb:
                for adapter_name, gate_list in module.gate_weights_for_lb.items():
                    if adapter_name not in adapter_gates:
                        continue
                    for gw in gate_list:
                        if gw.numel() == 0:
                            continue
                        # gw shape: [..., num_experts] -> flatten -> mean
                        P = gw.detach().reshape(-1, gw.shape[-1]).mean(dim=0)  # [num_experts]
                        adapter_gates[adapter_name].append(P)
                        num_experts = gw.shape[-1]

        if num_experts is None:
            return {}

        result = {}
        all_P = []

        for adapter_name, P_list in adapter_gates.items():
            if P_list:
                P_avg = torch.stack(P_list).mean(dim=0)  # [num_experts]
                short_name = adapter_name.replace('default_', '')  # 'encoder_pix', etc.
                result[short_name] = P_avg.cpu().tolist()
                all_P.extend(P_list)

        if all_P:
            result['global'] = torch.stack(all_P).mean(dim=0).cpu().tolist()

        return result

    def load_ckpt_from_state_dict(self, sd):
        # Load MoLE params (with fallback for old checkpoints without MoLE)
        num_experts_pix = sd.get("num_experts_pix", 1)
        top_k_pix = sd.get("top_k_pix", num_experts_pix)
        num_shared_experts_pix = sd.get("num_shared_experts_pix", 0)
        self.num_experts_pix = num_experts_pix
        self.top_k_pix = top_k_pix
        self.num_shared_experts_pix = num_shared_experts_pix

        # Load frequency-based routing params (with fallback for old checkpoints)
        use_freq_gate = sd.get("use_freq_gate", False)
        freq_dim = sd.get("freq_dim", 7)
        num_freq_bands = sd.get("num_freq_bands", 4)
        self.use_freq_gate = use_freq_gate
        self.freq_dim = freq_dim
        self.num_freq_bands = num_freq_bands

        # Initialize degradation feature extractor if using frequency-based routing
        if use_freq_gate:
            self.deg_feature_extractor = DegradationFeatureExtractor(num_bands=num_freq_bands)
        else:
            self.deg_feature_extractor = None

        # load unet lora with MoLE and frequency gate support
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"], number_of_lora=num_experts_pix, top_k_lora=top_k_pix, num_shared_experts=num_shared_experts_pix, use_freq_gate=use_freq_gate, freq_dim=freq_dim)
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"], number_of_lora=num_experts_pix, top_k_lora=top_k_pix, num_shared_experts=num_shared_experts_pix, use_freq_gate=use_freq_gate, freq_dim=freq_dim)
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"], number_of_lora=num_experts_pix, top_k_lora=top_k_pix, num_shared_experts=num_shared_experts_pix, use_freq_gate=use_freq_gate, freq_dim=freq_dim)

        # Semantic branch (no MoLE for now)
        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
        self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"], \
            sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

    # Adopted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def set_freq_features_for_lora(self, freq_features: torch.Tensor = None):
        """
        Set frequency features for all LoRA layers.

        This method sets the freq_features attribute on all LoRA modules so they
        can use it during the forward pass for frequency-based routing.

        Args:
            freq_features: [B, freq_dim] degradation features, or None to clear
        """
        for module in self.unet.modules():
            if hasattr(module, 'lora_gate') and module.lora_gate:
                module._freq_features = freq_features

    def extract_freq_features(self, img_lq: torch.Tensor) -> torch.Tensor:
        """
        Extract degradation features from low-quality input image.

        Args:
            img_lq: [B, C, H, W] low-quality input image (normalized to [-1, 1])

        Returns:
            freq_features: [B, freq_dim] degradation features, or None if not using freq gate
        """
        if self.deg_feature_extractor is None:
            return None

        with torch.no_grad():
            # Convert from [-1, 1] to [0, 1] for feature extraction
            img_normalized = (img_lq + 1) / 2
            freq_features = self.deg_feature_extractor(img_normalized)

        return freq_features

    def forward(self, c_t, c_tgt, batch=None, args=None):

        bs = c_t.shape[0]

        # Extract frequency features from LQ image for routing (if enabled)
        freq_features = self.extract_freq_features(c_t)
        self.set_freq_features_for_lora(freq_features)

        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds = self.encode_prompt(batch["null_prompt"])

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
        else:
            pos_caption_enc = prompt_embeds

        model_pred = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=pos_caption_enc.to(torch.float32),).sample
        x_denoised = encoded_control - model_pred
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        # Clear freq_features after forward to free memory
        self.set_freq_features_for_lora(None)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds


    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"] =\
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix
        sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"] =\
            self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem
        sd["lora_rank_unet_pix"] = self.lora_rank_unet_pix
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        # Save MoLE params for proper checkpoint loading
        sd["num_experts_pix"] = getattr(self, 'num_experts_pix', 1)
        sd["top_k_pix"] = getattr(self, 'top_k_pix', sd["num_experts_pix"])
        sd["num_shared_experts_pix"] = getattr(self, 'num_shared_experts_pix', 0)
        # Save frequency-based routing params
        sd["use_freq_gate"] = getattr(self, 'use_freq_gate', False)
        sd["freq_dim"] = getattr(self, 'freq_dim', 7)
        sd["num_freq_bands"] = getattr(self, 'num_freq_bands', 4)
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        torch.save(sd, outf)


class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        # Load pretrained weights (sets self.use_mole)
        self._load_pretrained_weights(args.pretrained_path)

        # Initialize VAE tiling
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        if self.use_mole:
            # MoLE: adapters already configured in _load_and_save_ckpt_from_state_dict
            # No additional merge needed
            pass
        else:
            # Original: prepare sem deltas and merge
            if not args.default:
                self._prepare_lora_deltas(["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
            set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
            self.unet.merge_and_unload()

        # Move models to device and precision
        self._move_models_to_device_and_dtype()

        # Set parameters
        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)

    def _get_dtype(self, precision):
        """Get the appropriate data type based on precision."""
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _move_models_to_device_and_dtype(self):
        """Move models to the correct device and precision."""
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and initialize LoRA adapters."""
        sd = torch.load(pretrained_path)
        self._load_and_save_ckpt_from_state_dict(sd)

    def _prepare_lora_deltas(self, adapter_names):
        """Precompute and store LoRA deltas for the given adapters."""
        self.lora_deltas_sem = {}
        key_list = [key for key, _ in self.unet.named_modules() if "lora_" not in key]

        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            base_layer = target.get_base_layer()
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight

                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):  # Conv2D 1x1
                                output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:  # Linear layer
                                output_tensor = transpose(weight_B @ weight_A, False) * target.scaling[active_adapter]
                            else:  # Conv2D 3x3
                                output_tensor = F.conv2d(
                                    weight_A.permute(1, 0, 2, 3),
                                    weight_B,
                                ).permute(1, 0, 2, 3) * target.scaling[active_adapter]

                            key = key + ".weight"
                            self.lora_deltas_sem[key] = output_tensor.data.to(dtype=self.weight_dtype, device=self.device)

    def _apply_lora_delta(self):
        """Merge LoRA deltas into UNet weights."""
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        """Restore original UNet weights."""
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _load_and_save_ckpt_from_state_dict(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        # Load MoLE params (with fallback for old checkpoints without MoLE)
        num_experts_pix = sd.get("num_experts_pix", 1)
        top_k_pix = sd.get("top_k_pix", num_experts_pix)
        num_shared_experts_pix = sd.get("num_shared_experts_pix", 0)
        self.num_experts_pix = num_experts_pix
        self.top_k_pix = top_k_pix
        self.num_shared_experts_pix = num_shared_experts_pix
        # use_mole is True only when we have multiple routed experts
        # (shared experts only exist when num_experts > 1)
        self.use_mole = num_experts_pix > 1

        # Load frequency-based routing params (with fallback for old checkpoints)
        use_freq_gate = sd.get("use_freq_gate", False)
        freq_dim = sd.get("freq_dim", 7)
        num_freq_bands = sd.get("num_freq_bands", 4)
        self.use_freq_gate = use_freq_gate
        self.freq_dim = freq_dim
        self.num_freq_bands = num_freq_bands

        # Initialize degradation feature extractor if using frequency-based routing
        # Keep float32 for FFT operations (ComplexHalf not well supported)
        if use_freq_gate:
            self.deg_feature_extractor = DegradationFeatureExtractor(num_bands=num_freq_bands).to(self.device)
        else:
            self.deg_feature_extractor = None

        # Define LoRA configurations with MoLE and frequency gate support
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"], number_of_lora=num_experts_pix, top_k_lora=top_k_pix, num_shared_experts=num_shared_experts_pix, use_freq_gate=use_freq_gate, freq_dim=freq_dim)
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"], number_of_lora=num_experts_pix, top_k_lora=top_k_pix, num_shared_experts=num_shared_experts_pix, use_freq_gate=use_freq_gate, freq_dim=freq_dim)
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"], number_of_lora=num_experts_pix, top_k_lora=top_k_pix, num_shared_experts=num_shared_experts_pix, use_freq_gate=use_freq_gate, freq_dim=freq_dim)

        # Semantic branch (no MoLE for now)
        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        if self.use_mole:
            # ============ MoLE PATH ============
            # Add ALL adapters without merging
            self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
            self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
            self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")
            self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
            self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
            self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

            # Load all weights
            for name, param in self.unet.named_parameters():
                if "lora" in name and name in sd["state_dict_unet"]:
                    param.data.copy_(sd["state_dict_unet"][name])

            # Activate all adapters
            all_adapters = [
                "default_encoder_pix", "default_decoder_pix", "default_others_pix",
                "default_encoder_sem", "default_decoder_sem", "default_others_sem"
            ]
            set_weights_and_activate_adapters(self.unet, all_adapters, [1.0] * 6)

            # NO merge_and_unload() for MoLE!
        else:
            # ============ ORIGINAL SINGLE LORA PATH ============
            # Add pix adapters
            self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
            self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
            self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

            for name, param in self.unet.named_parameters():
                if "pix" in name:
                    param.data.copy_(sd["state_dict_unet"][name])

            # Merge pix and save
            set_weights_and_activate_adapters(self.unet, ["default_encoder_pix", "default_decoder_pix", "default_others_pix"], [1.0, 1.0, 1.0])
            self.unet.merge_and_unload()

            self.ori_unet_weight = {}
            for name, param in self.unet.named_parameters():
                self.ori_unet_weight[name] = param.clone().data.to(self.weight_dtype).to("cuda")

            # Add sem adapters
            self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
            self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
            self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

            for name, param in self.unet.named_parameters():
                if "lora" in name:
                    param.data.copy_(sd["state_dict_unet"][name])


    def set_eval(self):
        """Set models to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def count_parameters(self, model):
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters()) / 1e9

    def set_freq_features_for_lora(self, freq_features: torch.Tensor = None):
        """
        Set frequency features for all LoRA layers.

        This method sets the freq_features attribute on all LoRA modules so they
        can use it during the forward pass for frequency-based routing.

        Args:
            freq_features: [B, freq_dim] degradation features, or None to clear
        """
        for module in self.unet.modules():
            if hasattr(module, 'lora_gate') and module.lora_gate:
                module._freq_features = freq_features

    def extract_freq_features(self, img_lq: torch.Tensor) -> torch.Tensor:
        """
        Extract degradation features from low-quality input image.

        Args:
            img_lq: [B, C, H, W] low-quality input image (normalized to [-1, 1])

        Returns:
            freq_features: [B, freq_dim] degradation features, or None if not using freq gate
        """
        if self.deg_feature_extractor is None:
            return None

        with torch.no_grad():
            # Convert from [-1, 1] to [0, 1] for feature extraction
            # Force float32 for FFT (ComplexHalf not supported, causes NaN)
            img_normalized = (img_lq.float() + 1) / 2
            freq_features = self.deg_feature_extractor(img_normalized)

        return freq_features

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        """Forward pass for inference."""
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)

        # Extract frequency features from LQ image for routing (if enabled)
        freq_features = self.extract_freq_features(c_t)
        self.set_freq_features_for_lora(freq_features)

        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # Tile and process latents if necessary
        model_pred = self._process_latents(encoded_control, prompt_embeds, default)

        # Decode output
        x_denoised = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        # Clear freq_features after forward to free memory
        self.set_freq_features_for_lora(None)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return total_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        """Process latents with or without tiling."""
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            print("[Tiled Latent]: Input size is small, no tiling required.")
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)

        print(f"[Tiled Latent]: Input size {h}x{w}, tiling required.")
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        """Predict on the entire latent without tiling."""
        if self.use_mole:
            pix_adapters = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
            sem_adapters = ["default_encoder_sem", "default_decoder_sem", "default_others_sem"]
            all_adapters = pix_adapters + sem_adapters

            if default:
                self.unet.set_adapter(all_adapters)
                return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
            else:
                # Adjustable mode: 2 forward passes
                self.unet.set_adapter(pix_adapters)
                model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

                self.unet.set_adapter(all_adapters)
                model_pred_full = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

                sem_only = model_pred_full - model_pred_pix
                return self.lambda_pix * model_pred_pix + self.lambda_sem * sem_only
        else:
            # Original code unchanged
            if default:
                return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

            model_pred_sem = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
            self._apply_ori_weight()
            model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
            self._apply_lora_delta()

            model_pred_sem -= model_pred_pix
            return self.lambda_pix * model_pred_pix + self.lambda_sem * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
        """Predict on the latent with tiling."""
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)

                    if self.use_mole:
                        pix_adapters = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
                        sem_adapters = ["default_encoder_sem", "default_decoder_sem", "default_others_sem"]
                        all_adapters = pix_adapters + sem_adapters

                        if default:
                            print(f"[0:Default setting (MoLE)]")
                            self.unet.set_adapter(all_adapters)
                            model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                        else:
                            print(f"[1:Adjustable setting (MoLE)]")
                            self.unet.set_adapter(pix_adapters)
                            model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

                            self.unet.set_adapter(all_adapters)
                            model_out_full = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

                            model_out_sem = model_out_full - model_out_pix
                            model_out = self.lambda_pix * model_out_pix + self.lambda_sem * model_out_sem
                    else:
                        # predict the noise residual
                        if default:
                            print(f"[0:Default setting]")
                            model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                        else:
                            print(f"[1:Adjustable setting]")
                            model_out_sem = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                            self._apply_ori_weight()
                            model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                            self._apply_lora_delta()
                            model_out_sem = model_out_sem - model_out_pix
                            model_out = self.lambda_pix * model_out_pix + self.lambda_sem * model_out_sem

                    input_list = []
                noise_preds.append(model_out)

        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        return model_pred



    

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generate a Gaussian mask for tile contributions."""
        from numpy import pi, exp, sqrt
        import numpy as np

        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        """Initialize VAE with tiled encoding/decoding."""
        encoder, decoder = self.vae.encoder, self.vae.decoder

        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward

        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)