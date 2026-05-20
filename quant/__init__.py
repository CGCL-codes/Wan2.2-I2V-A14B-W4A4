from .dit_struct import DiTBlockStruct, DiTLinearRef, DiTModelStruct
from .observers import DynamicActQuantizer
from .quant_linear import QuantLinear, QuantLinearWithBranch
from .replace import (
    QuantReplaceConfig,
    load_quant_config,
    replace_wan_dit_linear_with_branch_quant,
    replace_wan_dit_linear_with_quant,
)
try:
    from . import wan_smooth as _wan_smooth
    _HAS_WAN_SMOOTH = True
except (ModuleNotFoundError, RuntimeError):
    _wan_smooth = None
    _HAS_WAN_SMOOTH = False

try:
    from .dit_ptq import (
        DiTActivationQuantConfig,
        DiTQuantCachePaths,
        DiTQuantConfig,
        DiTWeightQuantConfig,
        load_dit_weights_state_dict,
        ptq,
        quantize_dit_activations,
        quantize_dit_weights,
    )
    _HAS_DIT_PTQ = True
except ModuleNotFoundError:
    _HAS_DIT_PTQ = False

_OPTIONAL_WAN_SMOOTH_EXPORTS = []
if _HAS_WAN_SMOOTH:
    # New act/stat + smoothed-weight export API.
    WanActStatConfig = _wan_smooth.WanActStatConfig
    LinearActStatCollector = _wan_smooth.LinearActStatCollector
    collect_wan_linear_modules = _wan_smooth.collect_wan_linear_modules
    calibrate_wan_acts = _wan_smooth.calibrate_wan_acts
    save_wan_act_stats = _wan_smooth.save_wan_act_stats
    build_wan_smoothed_weights = _wan_smooth.build_wan_smoothed_weights
    save_wan_smoothed_weights = _wan_smooth.save_wan_smoothed_weights

    # Optional backward-compat smooth API (export only when present).
    _OPTIONAL_WAN_SMOOTH_EXPORTS = [
        "WanSmoothCalibConfig",
        "WanSmoothTargetSpec",
        "apply_wan_smooth_hooks",
        "build_low_rank_branch_from_smoothed_weight",
        "calibrate_wan_smooth",
        "collect_wan_smooth_targets",
        "export_svdquant_artifacts",
        "export_wan_smoothed_weights",
        "get_smoothed_weight",
        "load_and_apply_wan_smooth",
        "materialize_wan_smoothed_weights",
        "prepare_svdquant_inputs",
        "quantize_smoothed_main_branch",
        "remove_wan_smooth_hooks",
        "svd_decompose_smoothed_weight",
    ]
    for _name in _OPTIONAL_WAN_SMOOTH_EXPORTS:
        if hasattr(_wan_smooth, _name):
            globals()[_name] = getattr(_wan_smooth, _name)

__all__ = [
    "DiTBlockStruct",
    "DiTLinearRef",
    "DiTModelStruct",
    "DynamicActQuantizer",
    "QuantLinear",
    "QuantLinearWithBranch",
    "QuantReplaceConfig",
    "load_quant_config",
    "replace_wan_dit_linear_with_branch_quant",
    "replace_wan_dit_linear_with_quant",
]

if _HAS_WAN_SMOOTH:
    __all__.extend([
        "WanActStatConfig",
        "LinearActStatCollector",
        "collect_wan_linear_modules",
        "calibrate_wan_acts",
        "save_wan_act_stats",
        "build_wan_smoothed_weights",
        "save_wan_smoothed_weights",
    ])

if _HAS_DIT_PTQ:
    __all__.extend([
        "DiTActivationQuantConfig",
        "DiTQuantCachePaths",
        "DiTQuantConfig",
        "DiTWeightQuantConfig",
        "load_dit_weights_state_dict",
        "ptq",
        "quantize_dit_activations",
        "quantize_dit_weights",
    ])

for _name in _OPTIONAL_WAN_SMOOTH_EXPORTS:
    if _name in globals():
        __all__.append(_name)
