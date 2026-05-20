# Wan2.2 I2V-A14B W4A4 (Engineering MXFP4)

## Milestone 1: Replace + Forward (W4A4 modules)
```bash
python - << 'PY'
import wan
from wan.configs import WAN_CONFIGS
from quant.replace import load_quant_config, replace_wan_dit_linear_with_quant

cfg = WAN_CONFIGS['i2v-A14B']
pipe = wan.WanI2V(config=cfg, checkpoint_dir='./Wan2.2-I2V-A14B', convert_model_dtype=True)
qcfg = load_quant_config('configs/wan_i2v_w4a4_mxfp4.yaml')
replace_wan_dit_linear_with_quant(pipe.low_noise_model, qcfg, log_dir='./quant_logs', model_tag='low_noise_model')
replace_wan_dit_linear_with_quant(pipe.high_noise_model, qcfg, log_dir='./quant_logs', model_tag='high_noise_model')
print('replace done')
PY
```

## Milestone 2: Calibration with MSVD manifest
```bash
python quant/calibrate_wan_i2v.py \
  --model_path ./Wan2.2-I2V-A14B \
  --manifest /home/wjh/MSVD/msvd_wan_i2v_calib_manifest.jsonl \
  --output_dir ./quant_calib_out \
  --quant_config configs/wan_i2v_w4a4_mxfp4.yaml \
  --calib_samples 64 \
  --batch_size 1 \
  --save_quant_model
```

## Milestone 3: Error report (BF16 vs Quant)
Use `--run_eval_hooks` during calibration:
```bash
python quant/calibrate_wan_i2v.py \
  --model_path ./Wan2.2-I2V-A14B \
  --manifest /home/wjh/MSVD/msvd_wan_i2v_calib_manifest.jsonl \
  --output_dir ./quant_calib_out \
  --run_eval_hooks \
  --eval_level block
```
Outputs:
- `eval_low_noise_report.json`
- `eval_high_noise_report.json`

## Milestone 4: Export MXFP4 checkpoints
```bash
python quant/export_mxfp4.py \
  --src_ckpt_dir ./Wan2.2-I2V-A14B \
  --output_dir ./Wan2.2-I2V-A14B-W4A4-MXFP4 \
  --quant_config configs/wan_i2v_w4a4_mxfp4.yaml
```

## Load for I2V inference
```bash
python generate.py --task i2v-A14B \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --load_mxfp4 ./Wan2.2-I2V-A14B-W4A4-MXFP4 \
  --quant_config configs/wan_i2v_w4a4_mxfp4.yaml \
  --image examples/i2v_input.JPG \
  --prompt "your prompt"
```

## SVDQuant Stage-1: Wan DiT Smooth (for future low-rank + low-bit)
Smooth in this repo is designed as **stage-1** for SVDQuant:
1. calibrate per-channel smooth scale for Wan DiT target linears (`q/k/v`, `out`, `ffn up/down`);
2. shift outliers from activation side to weight side by `W'[:, i] = W[:, i] * s[i]`;
3. keep an explicit `smooth.pt` that can be reloaded later;
4. ensure stage-2 SVD uses **smoothed weights**, not original weights.

Two application modes are supported:
- `hook`: scale weight and divide input in `forward_pre_hook`, preserving functional equivalence.
- `materialize`: export a smoothed state dict for stage-2 SVD decomposition input.

Run minimal smooth pipeline:
```bash
python tools/calibrate_wan_dit_smooth.py \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --subfolder low_noise_model \
  --msvd_root /home/wjh/MSVD \
  --output_dir ./smooth_out \
  --num_samples 8 \
  --target_frames 61 --height 480 --width 832 \
  --apply_mode hook
```

Artifacts:
- `*_smooth.pt`: reusable smooth cache with metadata and per-module scales.
- `*_smoothed_weights.pt`: materialized smoothed weights for stage-2 SVD input.
