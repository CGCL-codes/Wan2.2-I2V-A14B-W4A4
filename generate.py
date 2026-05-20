# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from quant.quant_linear import QuantLinear, QuantLinearWithBranch
from quant.replace import load_quant_config, replace_wan_dit_linear_with_quant
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.modules.mxfp4_per_tensor import load_quantized_wan_model
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


EXPECTED_PTQ_SCHEME = "mxfp4"
EXPECTED_PTQ_WEIGHT_BLOCK_SIZE = 32
EXPECTED_PTQ_ACT_BLOCK_SIZE = 32


def _find_parent_and_leaf(root, full_name):
    parts = full_name.split(".")
    if not parts:
        raise ValueError("empty module path")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _set_module_by_name(root, full_name, module):
    parent, leaf = _find_parent_and_leaf(root, full_name)
    parent._modules[leaf] = module


def _validate_ptq_meta(meta, path, expected_model_name):
    if not isinstance(meta, dict):
        raise TypeError(f"PTQ state __meta__ must be dict at {path}")

    model_name = str(meta.get("model_name", ""))
    if model_name != expected_model_name:
        raise ValueError(
            f"PTQ state model_name mismatch at {path}: expected {expected_model_name}, got {model_name}"
        )

    quant_scheme = str(meta.get("quant_scheme", ""))
    if quant_scheme != EXPECTED_PTQ_SCHEME:
        raise ValueError(
            f"PTQ state quant_scheme mismatch at {path}: expected {EXPECTED_PTQ_SCHEME}, got {quant_scheme}"
        )

    weight_block_size = int(meta.get("weight_block_size", meta.get("weight_group_size", -1)))
    act_block_size = int(meta.get("activation_block_size", meta.get("act_group_size", -1)))
    if weight_block_size != EXPECTED_PTQ_WEIGHT_BLOCK_SIZE:
        raise ValueError(
            f"PTQ state weight block mismatch at {path}: expected {EXPECTED_PTQ_WEIGHT_BLOCK_SIZE}, got {weight_block_size}"
        )
    if act_block_size != EXPECTED_PTQ_ACT_BLOCK_SIZE:
        raise ValueError(
            f"PTQ state activation block mismatch at {path}: expected {EXPECTED_PTQ_ACT_BLOCK_SIZE}, got {act_block_size}"
        )


def _validate_quant_module_state(state, path, model_name, module_name):
    qstate = state.get("quant_linear", None)
    if not isinstance(qstate, dict):
        raise ValueError(f"Missing quant_linear state at {path}:{model_name}.{module_name}")

    scheme = str(qstate.get("scheme", ""))
    if scheme != EXPECTED_PTQ_SCHEME:
        raise ValueError(
            f"Module quant scheme mismatch at {path}:{model_name}.{module_name}: expected {EXPECTED_PTQ_SCHEME}, got {scheme}"
        )

    weight_block_size = int(qstate.get("w_group_size", qstate.get("weight_group_size", -1)))
    act_block_size = int(qstate.get("act_group_size", -1))
    if weight_block_size != EXPECTED_PTQ_WEIGHT_BLOCK_SIZE:
        raise ValueError(
            f"Module weight block mismatch at {path}:{model_name}.{module_name}: expected {EXPECTED_PTQ_WEIGHT_BLOCK_SIZE}, got {weight_block_size}"
        )
    if act_block_size != EXPECTED_PTQ_ACT_BLOCK_SIZE:
        raise ValueError(
            f"Module activation block mismatch at {path}:{model_name}.{module_name}: expected {EXPECTED_PTQ_ACT_BLOCK_SIZE}, got {act_block_size}"
        )


def _build_quant_module_from_state(state):
    qstate = state["quant_linear"]
    bias = qstate.get("bias", None)
    qmod = QuantLinear(
        in_features=int(qstate["in_features"]),
        out_features=int(qstate["out_features"]),
        bias=bias is not None,
        weight_group_size=int(qstate.get("w_group_size", qstate.get("weight_group_size", 32))),
        act_group_size=int(qstate.get("act_group_size", 32)),
        weight_clip_ratio=float(qstate.get("weight_clip_ratio", 1.0)),
        act_clip_ratio=float(qstate.get("act_clip_ratio", 1.0)),
        quantize_input=bool(qstate.get("quantize_input", True)),
        skip_input_quant=bool(qstate.get("skip_input_quant", False)),
        compute_dtype=qstate.get("compute_dtype", "bfloat16"),
        scheme=qstate.get("scheme", EXPECTED_PTQ_SCHEME),
    )
    out = QuantLinearWithBranch(quant_linear=qmod, branch=None)
    out.load_state(state)
    out.eval().requires_grad_(False)
    return out


def _load_ptq_state(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected dict PTQ state at {path}, got {type(state).__name__}")
    return state


def _apply_ptq_state_to_model(model, ptq_state, model_name, path):
    meta = ptq_state.get("__meta__", None)
    if meta is None:
        raise KeyError(f"PTQ state missing __meta__ at {path}")
    _validate_ptq_meta(meta, path, expected_model_name=model_name)

    replaced = 0
    for module_name, state in ptq_state.items():
        if module_name == "__meta__":
            continue
        if not isinstance(state, dict) or "quant_linear" not in state:
            raise ValueError(f"Invalid PTQ state entry at {model_name}.{module_name}")
        _validate_quant_module_state(state, path=path, model_name=model_name, module_name=module_name)
        qmod = _build_quant_module_from_state(state)
        _set_module_by_name(model, module_name, qmod)
        replaced += 1
    logging.info(
        "%s loaded PTQ modules: %d | scheme=%s | weight_block=%d | act_block=%d",
        model_name,
        replaced,
        EXPECTED_PTQ_SCHEME,
        EXPECTED_PTQ_WEIGHT_BLOCK_SIZE,
        EXPECTED_PTQ_ACT_BLOCK_SIZE,
    )
    return replaced


def _load_i2v_ptq_states(wan_i2v, ptq_dir):
    low_ptq_path = os.path.join(ptq_dir, "low_noise_model", "ptq_state.pt")
    high_ptq_path = os.path.join(ptq_dir, "high_noise_model", "ptq_state.pt")
    assert os.path.exists(low_ptq_path), f"PTQ state not found: {low_ptq_path}"
    assert os.path.exists(high_ptq_path), f"PTQ state not found: {high_ptq_path}"

    low_state = _load_ptq_state(low_ptq_path)
    high_state = _load_ptq_state(high_ptq_path)
    _apply_ptq_state_to_model(wan_i2v.low_noise_model, low_state, "low_noise_model", low_ptq_path)
    _apply_ptq_state_to_model(wan_i2v.high_noise_model, high_state, "high_noise_model", high_ptq_path)
    logging.info("SVDQuant PTQ states loaded from %s", ptq_dir)


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."
    if args.ptq_dir is not None:
        assert args.task == "i2v-A14B", \
            "--ptq_dir currently only supports i2v-A14B."
        assert not args.dit_fsdp, \
            "--ptq_dir is not supported with --dit_fsdp."
    if args.load_mxfp4 is not None:
        assert args.task == "i2v-A14B", \
            "--load_mxfp4 currently only supports i2v-A14B."
        assert not args.dit_fsdp, \
            "--load_mxfp4 is not supported with --dit_fsdp."
    if args.dit_quant_type != "none":
        assert args.task == "i2v-A14B", \
            "Currently only i2v-A14B supports --dit_quant_type."
        assert not args.dit_fsdp, \
            "--dit_quant_type is not supported with --dit_fsdp."
        assert args.dit_quant_ckpt_dir is not None, \
            "Please provide --dit_quant_ckpt_dir when using --dit_quant_type."
        assert args.load_mxfp4 is None, \
            "--load_mxfp4 and --dit_quant_type cannot be used together."
    if args.ptq_dir is not None:
        assert args.load_mxfp4 is None, \
            "--ptq_dir and --load_mxfp4 cannot be used together."
        assert args.dit_quant_type == "none", \
            "--ptq_dir and --dit_quant_type cannot be used together."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--dit_quant_type",
        type=str,
        default="none",
        choices=["none", "mxfp4_per_tensor"],
        help="DiT quantization runtime type. Set to mxfp4_per_tensor to load custom quantized I2V DiT checkpoints."
    )
    parser.add_argument(
        "--dit_quant_ckpt_dir",
        type=str,
        default=None,
        help="Directory containing quantized DiT checkpoints, e.g. low_noise_model/mxfp4_per_tensor_model.pt and high_noise_model/mxfp4_per_tensor_model.pt."
    )
    parser.add_argument(
        "--quant_config",
        type=str,
        default=None,
        help="Path to W4A4 MXFP4 quant config yaml/json. Used with --load_mxfp4."
    )
    parser.add_argument(
        "--load_mxfp4",
        type=str,
        default=None,
        help="Path to exported W4A4 MXFP4 checkpoint dir. Expect low_noise_model/mxfp4_state.pt and high_noise_model/mxfp4_state.pt."
    )
    parser.add_argument(
        "--ptq_dir",
        type=str,
        default=None,
        help="Path to exported SVDQuant PTQ dir. Expect low_noise_model/ptq_state.pt and high_noise_model/ptq_state.pt."
    )

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "animate" in args.task:
        logging.info("Creating Wan-Animate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )

        logging.info(f"Generating video ...")
        video = wan_animate.generate(
            src_root_path=args.src_root_path,
            replace_flag=args.replace_flag,
            refert_num = args.refert_num,
            clip_len=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        if args.ptq_dir is not None:
            logging.info("Loading SVDQuant PTQ states ...")
            _load_i2v_ptq_states(wan_i2v, args.ptq_dir)
        elif args.load_mxfp4 is not None:
            low_q = os.path.join(args.load_mxfp4, "low_noise_model",
                                 "mxfp4_state.pt")
            high_q = os.path.join(args.load_mxfp4, "high_noise_model",
                                  "mxfp4_state.pt")
            assert os.path.exists(low_q), f"Quant checkpoint not found: {low_q}"
            assert os.path.exists(high_q), f"Quant checkpoint not found: {high_q}"

            qcfg = load_quant_config(args.quant_config)
            logging.info("Applying W4A4 MXFP4 module replacement ...")
            replace_wan_dit_linear_with_quant(
                wan_i2v.low_noise_model,
                config=qcfg,
                log_dir=os.path.join(args.load_mxfp4, "runtime_replace_logs"),
                model_tag="low_noise_model_runtime")
            replace_wan_dit_linear_with_quant(
                wan_i2v.high_noise_model,
                config=qcfg,
                log_dir=os.path.join(args.load_mxfp4, "runtime_replace_logs"),
                model_tag="high_noise_model_runtime")
            logging.info("Loading W4A4 MXFP4 states ...")
            wan_i2v.low_noise_model.load_state_dict(
                torch.load(low_q, map_location="cpu"), strict=False)
            wan_i2v.high_noise_model.load_state_dict(
                torch.load(high_q, map_location="cpu"), strict=False)
            logging.info("W4A4 MXFP4 models loaded.")
        elif args.dit_quant_type == "mxfp4_per_tensor":
            low_q = os.path.join(args.dit_quant_ckpt_dir, "low_noise_model",
                                 "mxfp4_per_tensor_model.pt")
            high_q = os.path.join(args.dit_quant_ckpt_dir, "high_noise_model",
                                  "mxfp4_per_tensor_model.pt")
            assert os.path.exists(low_q), f"Quant checkpoint not found: {low_q}"
            assert os.path.exists(high_q), f"Quant checkpoint not found: {high_q}"

            logging.info(
                "Loading quantized DiT checkpoints (mxfp4_per_tensor) ...")
            wan_i2v.low_noise_model = load_quantized_wan_model(
                wan_i2v.low_noise_model, low_q)
            wan_i2v.high_noise_model = load_quantized_wan_model(
                wan_i2v.high_noise_model, high_q)
            logging.info("Quantized DiT checkpoints loaded.")
        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
