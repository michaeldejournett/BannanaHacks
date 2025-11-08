# Monitoring AMD GPU Usage

## Quick Methods

### Method 1: rocm-smi (AMD's Official Tool)

Install if not available:
```bash
sudo apt install rocm-smi
```

Then run:
```bash
rocm-smi
```

Or watch continuously:
```bash
watch -n 1 rocm-smi
```

### Method 2: GPU Busy Percent (Simple)

Check GPU utilization percentage:
```bash
watch -n 1 'cat /sys/class/drm/card1/device/gpu_busy_percent'
```

Or for a one-time check:
```bash
cat /sys/class/drm/card1/device/gpu_busy_percent
```

### Method 3: radeontop (Real-time Monitor)

Install:
```bash
sudo apt install radeontop
```

Run:
```bash
sudo radeontop
```

Press 'q' to quit.

### Method 4: Python Script (Custom)

Create a simple monitoring script:

```python
#!/usr/bin/env python3
import time
import os

while True:
    try:
        with open('/sys/class/drm/card1/device/gpu_busy_percent', 'r') as f:
            usage = f.read().strip()
        print(f"GPU Usage: {usage}%", end='\r')
    except:
        print("GPU usage not available")
    time.sleep(1)
```

Save as `monitor_gpu.py` and run:
```bash
python monitor_gpu.py
```

### Method 5: System Monitor (GUI)

Most Linux system monitors can show GPU usage:
- **GNOME System Monitor**: Applications → System Monitor → Resources
- **KSysGuard** (KDE)
- **htop** with GPU plugin (if available)

## Finding Your GPU Device

Your discrete GPU (RX 6650M) is likely:
- `/sys/class/drm/card1/device/gpu_busy_percent`

To find which card is which:
```bash
for card in /sys/class/drm/card*/device/uevent; do
    echo "=== $(dirname $card) ==="
    grep -E "PCI_SLOT_NAME|PCI_ID" $card
done
```

Or check GPU names:
```bash
for card in /sys/class/drm/card*/device/uevent; do
    card_num=$(basename $(dirname $(dirname $card)))
    echo "Card $card_num:"
    cat $card | grep -E "PCI_SLOT_NAME"
done
```

## Recommended: Simple Watch Command

For quick monitoring during training:
```bash
watch -n 1 'echo "GPU Usage:" && cat /sys/class/drm/card1/device/gpu_busy_percent && echo "%"'
```

Or create an alias in `~/.bashrc`:
```bash
alias gpu-usage='watch -n 1 "cat /sys/class/drm/card1/device/gpu_busy_percent"'
```

Then just run: `gpu-usage`

## During Training

Run monitoring in a separate terminal while training:

**Terminal 1 (Training):**
```bash
cd Model
python train.py ...
```

**Terminal 2 (Monitoring):**
```bash
watch -n 1 'cat /sys/class/drm/card1/device/gpu_busy_percent'
```

You should see GPU usage increase from ~12% to much higher (ideally 80-100%) during training!

