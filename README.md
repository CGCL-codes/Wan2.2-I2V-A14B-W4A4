# Wan2.2 I2V SVDQuant Inference Guide

This project is an SVDQuant / MXFP4 quantized inference implementation based on Wan2.2 I2V-A14B. The main submission includes inference code, quantization modules, configuration files, and run scripts. The archive does not include datasets, original Wan weights, PTQ quantized weights, evaluation output videos, or cache files.

## 1. Directory Overview

The following directories and files are primarily used for inference:

- `wan/`: Core Wan2.2 code, including the model, sampler, VAE, T5, I2V pipeline, and related components.
- `quant/`: Quantization implementations, including SVDQuant / MXFP4 quantized linear layers, GPTQ, activation policy, and related modules.
- `tools/infer_wan_i2v_svdquant.py`: Entry point for single-sample I2V quantized inference.
- `tools/generate_vbench_i2v_svdquant.py`: Entry point for batch generation on VBench-I2V.
- `configs/`: Quantization-related configuration files.
- `requirements.txt`, `environment.yml`, `pyproject.toml`: Environment and dependency specifications.

## 2. External File Preparation

Before running the code, prepare the following files yourself. It is recommended to place them outside the project directory or specify their paths through command-line arguments:

- The original Wan2.2 I2V-A14B bf16 checkpoint, for example `../Wan2.2-I2V-A14B-bf16`.
- The PTQ artifact exported by this algorithm, for example `outputs/gptq_ptq/ptq_stats.pt`.
- An optional activation policy, for example `outputs/act_policy.json`.
- The VBench-I2V dataset and `vbench2_i2v_full_info.json`, required only for batch evaluation generation.

The PTQ artifact supports three formats:

- A direct path to a combined file: `--ptq_dir /path/to/ptq_stats.pt`
- A path to a directory containing the combined file: `--ptq_dir /path/to/ptq_dir`
- The legacy split-directory format: `/path/to/ptq_dir/low_noise_model/ptq_state.pt` and `/path/to/ptq_dir/high_noise_model/ptq_state.pt`

## 3. Environment Setup

Python 3.10 is recommended. You can create the environment using the conda environment file:

```bash
conda env create -f environment.yml
conda activate wan
```

Alternatively, install PyTorch manually first, then install the project dependencies:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Install `torch` / `torchvision` versions that match the CUDA version on your machine. If a working Wan2.2 runtime environment is already available, you usually only need to make sure that commands are run from the root directory of this repository.

## 4. Single-Image Quantized Inference

Run the following command from the project root:

```bash
python tools/infer_wan_i2v_svdquant.py \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B-bf16 \
  --ptq_dir /path/to/ptq_stats_or_dir \
  --image examples/5.png \
  --prompt "A man gently clutching a bouquet of vibrant flowers, his eyes radiating a serene contentment as he glances at the camera." \
  --size "480*832" \
  --frame_num 61 \
  --sample_steps 40 \
  --sample_solver unipc \
  --device_id 0 \
  --offload_model True \
  --save_file outputs/demo_i2v_svdquant.mp4
```

Common arguments:

- `--ckpt_dir`: Path to the Wan2.2 I2V-A14B bf16 checkpoint.
- `--ptq_dir`: Path to the PTQ artifact file or directory.
- `--image`: Input first-frame image.
- `--prompt`: Text prompt.
- `--size`: Output size key. The default is `480*832`.
- `--frame_num`: Number of generated frames. The default is `61`.
- `--sample_steps`: Number of sampling steps. The default is `40`.
- `--device_id`: CUDA device ID.
- `--save_file`: Output video path. If omitted, an MP4 file with a timestamp will be generated in the current directory.
- `--act_policy_json`: Path to the activation policy JSON file. If it is not used, pass an empty string: `--act_policy_json ""`.
- `--act_scale_method`: Override the activation quantization scale method. Supported values are `ocp_floor` and `safe_ceil`.

By default, the experimental `--freeze_condition_latent` logic is enabled. It resets the I2V condition-frame latent after the final step to improve first-frame consistency.

## 5. VBench-I2V Batch Generation

After preparing the VBench-I2V data, batch-generate evaluation videos with the following command:

```bash
python tools/generate_vbench_i2v_svdquant.py \
  --full_info_json /path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --vbench_root /path/to/VBench/vbench2_beta_i2v \
  --ratio 16-9 \
  --output_dir outputs/vbench_quant \
  --dimensions i2v_subject i2v_background camera_motion \
  --samples_per_prompt 5 \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B-bf16 \
  --ptq_dir /path/to/ptq_stats_or_dir \
  --size "480*832" \
  --frame_num 61 \
  --sample_steps 40 \
  --device_id 0 \
  --offload_model True
```

The batch script generates MP4 files under `--output_dir` and writes `generation_manifest.jsonl`. For debugging, you can first run:

```bash
python tools/generate_vbench_i2v_svdquant.py \
  --full_info_json /path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --vbench_root /path/to/VBench/vbench2_beta_i2v \
  --dry_run
```

## 6. Optional: Keep Selected BF16 Modules

For ablation studies or to avoid quantizing specific layers, use the following arguments to keep selected modules in BF16:

```bash
python tools/infer_wan_i2v_svdquant.py \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B-bf16 \
  --ptq_dir /path/to/ptq_stats_or_dir \
  --image examples/5.png \
  --prompt "your prompt" \
  --low_keep_fp_blocks "0,1,2" \
  --high_keep_fp_modules "blocks.*.ffn.2" \
  --print_replaced_modules
```

Supported expressions:

- `--low_keep_fp_blocks "0,3-5"` / `--high_keep_fp_blocks "0,3-5"`: Keep BF16 by block index.
- `--low_keep_fp_modules` / `--high_keep_fp_modules`: Specify module names. Patterns such as `blocks.*.xxx` and `blocks.3-8.xxx` are supported.
- `--keep_fp_module_regex`: Use a regular expression to match `nn.Linear` modules in both experts.

## 7. Contents Excluded from the Submission Package

The archive intentionally excludes the following contents:

- Original Wan checkpoints, quantized PTQ artifacts, and OpenS2V/VBench weights.
- Datasets such as VBench, OpenS2V, and MSVD.
- Generated results such as `outputs/`, `opens2v_outputs/`, and `evaluation_results/`.
- Weight files such as `.pt`, `.pth`, `.safetensors`, `.ckpt`, and `.bin`.
- Generated MP4 files, caches, `__pycache__`, and Git metadata.

Therefore, after extracting the archive, you must specify external weight paths through `--ckpt_dir` and `--ptq_dir` before running quantized inference.
