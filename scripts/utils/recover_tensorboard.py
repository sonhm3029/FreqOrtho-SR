#!/usr/bin/env python3
"""
Recover TensorBoard logs from training_log.txt
Usage: python recover_tensorboard.py <training_log.txt> <output_dir>
"""

import re
import sys
from torch.utils.tensorboard import SummaryWriter

def parse_log_line(line):
    """Parse a log line and extract step and metrics."""
    # Pattern: [2025-11-22 17:26:53] Step 1: L2=0.048441, LPIPS=0.000000, CSD=0.000000
    pattern = r'\[.*?\] Step (\d+): L2=([\d.]+), LPIPS=([\d.]+), CSD=([\d.]+)'
    match = re.match(pattern, line)
    if match:
        step = int(match.group(1))
        l2 = float(match.group(2))
        lpips = float(match.group(3))
        csd = float(match.group(4))
        return step, {'loss_l2': l2, 'loss_lpips': lpips, 'loss_csd': csd}
    return None, None

def main():
    if len(sys.argv) != 3:
        print("Usage: python recover_tensorboard.py <training_log.txt> <output_dir>")
        sys.exit(1)

    log_file = sys.argv[1]
    output_dir = sys.argv[2]

    writer = SummaryWriter(output_dir)
    count = 0

    with open(log_file, 'r') as f:
        for line in f:
            step, metrics = parse_log_line(line.strip())
            if step is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, step)
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} steps...")

    writer.close()
    print(f"Done! Recovered {count} steps to {output_dir}")

if __name__ == "__main__":
    main()
