# Wan2.2 I2V SVDQuant 推理说明

本项目是在 Wan2.2 I2V-A14B 基础上实现的 SVDQuant / MXFP4 量化推理工程，主要提交内容为推理代码、量化模块、配置文件和运行脚本。压缩包不包含数据集、Wan 原始权重、PTQ 量化权重、评测输出视频或缓存文件。

## 1. 目录说明

推理主要用到以下目录和文件：

- `wan/`: Wan2.2 模型、采样器、VAE、T5、I2V pipeline 等基础代码。
- `quant/`: SVDQuant / MXFP4 量化线性层、GPTQ、activation policy 等量化实现。
- `tools/infer_wan_i2v_svdquant.py`: 单条 I2V 量化推理入口。
- `tools/generate_vbench_i2v_svdquant.py`: VBench-I2V 批量生成入口。
- `configs/`: 量化相关配置。
- `requirements.txt`, `environment.yml`, `pyproject.toml`: 环境依赖说明。

## 2. 外部文件准备

运行前需要自行准备以下文件，建议放在项目目录外或按命令行参数指定路径：

- Wan2.2 I2V-A14B bf16 原始 checkpoint，例如 `../Wan2.2-I2V-A14B-bf16`。
- 本算法导出的 PTQ artifact，例如 `outputs/gptq_ptq/ptq_stats.pt`。
- 可选的 activation policy，例如 `outputs/act_policy.json`。
- VBench-I2V 数据集与 `vbench2_i2v_full_info.json`，仅在批量评测生成时需要。

PTQ artifact 支持三种形式：

- 直接传入 combined 文件：`--ptq_dir /path/to/ptq_stats.pt`
- 传入包含 combined 文件的目录：`--ptq_dir /path/to/ptq_dir`
- 旧版拆分目录：`/path/to/ptq_dir/low_noise_model/ptq_state.pt` 和 `/path/to/ptq_dir/high_noise_model/ptq_state.pt`

## 3. 环境安装

推荐使用 Python 3.10。可使用 conda 环境文件：

```bash
conda env create -f environment.yml
conda activate wan
```

也可以手动安装 PyTorch 后再安装项目依赖：

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

请根据机器 CUDA 版本安装匹配的 `torch` / `torchvision`。如果环境中已经有可用的 Wan2.2 运行环境，通常只需要确认本仓库根目录在当前工作目录下运行即可。

## 4. 单张图片量化推理

在项目根目录运行：

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

常用参数：

- `--ckpt_dir`: Wan2.2 I2V-A14B bf16 checkpoint 路径。
- `--ptq_dir`: PTQ artifact 文件或目录路径。
- `--image`: 输入首帧图片。
- `--prompt`: 文本提示词。
- `--size`: 输出尺寸 key，默认 `480*832`。
- `--frame_num`: 生成帧数，默认 `61`。
- `--sample_steps`: 采样步数，默认 `40`。
- `--device_id`: CUDA device id。
- `--save_file`: 输出视频路径；不指定时会在当前目录生成带时间戳的 mp4。
- `--act_policy_json`: activation policy JSON 路径；如果不使用可传空字符串 `--act_policy_json ""`。
- `--act_scale_method`: 覆盖激活量化 scale 方法，可选 `ocp_floor` 或 `safe_ceil`。

默认会启用 `--freeze_condition_latent` 实验逻辑，在最后一步后重置 I2V 条件帧 latent，以提升首帧一致性。

## 5. VBench-I2V 批量生成

准备好 VBench-I2V 数据后，可以批量生成评测视频：

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

批量脚本会在 `--output_dir` 下生成 mp4，并写入 `generation_manifest.jsonl`。调试时可先使用：

```bash
python tools/generate_vbench_i2v_svdquant.py \
  --full_info_json /path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --vbench_root /path/to/VBench/vbench2_beta_i2v \
  --dry_run
```

## 6. 可选：保留部分 BF16 模块

为了做消融或避免个别层量化，可用以下参数保留 BF16：

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

支持的表达方式：

- `--low_keep_fp_blocks "0,3-5"` / `--high_keep_fp_blocks "0,3-5"`: 按 block index 保留 BF16。
- `--low_keep_fp_modules` / `--high_keep_fp_modules`: 指定模块名，支持 `blocks.*.xxx` 和 `blocks.3-8.xxx`。
- `--keep_fp_module_regex`: 使用正则匹配两个 expert 中的 `nn.Linear` 模块。

## 7. 提交包不包含的内容

压缩包刻意排除了以下内容：

- Wan 原始 checkpoint、量化 PTQ artifact、OpenS2V/VBench 权重。
- VBench/OpenS2V/MSVD 等数据集。
- `outputs/`、`opens2v_outputs/`、`evaluation_results/` 等生成结果。
- `.pt`、`.pth`、`.safetensors`、`.ckpt`、`.bin` 等权重文件。
- 生成的 mp4、缓存、`__pycache__`、git 元数据。

因此解压后需要通过 `--ckpt_dir` 和 `--ptq_dir` 指定外部权重路径后才能进行量化推理。
