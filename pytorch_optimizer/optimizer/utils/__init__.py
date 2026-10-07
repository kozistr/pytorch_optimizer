from pytorch_optimizer.base.compatibility import (
    HAS_TRANSFORMERS,
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_8,
    compare_versions,
    is_deepspeed_zero3_enabled,
    parse_pytorch_version,
)
from pytorch_optimizer.optimizer.utils.gradient import (
    clip_grad_norm,
    get_global_gradient_norm,
    normalize_gradient,
    unit_norm,
)
from pytorch_optimizer.optimizer.utils.graft import (
    AdaGradGraft,
    Graft,
    LayerWiseGrafting,
    RMSPropGraft,
    SGDGraft,
    SQRTNGraft,
    build_graft,
)
from pytorch_optimizer.optimizer.utils.matrix import (
    NS_COEFFICIENTS,
    DTensor,
    NewtonSchulzWeight,
    NewtonSchulzWeights,
    batched_power_iteration,
    compute_power_newton_db,
    compute_power_schur_newton,
    compute_power_svd,
    get_newton_schulz_weights,
    has_dtensor,
    power_iteration,
    zero_power_via_newton_schulz_5,
)
from pytorch_optimizer.optimizer.utils.model import (
    disable_running_stats,
    enable_running_stats,
    is_valid_parameters,
    reg_noise,
)
from pytorch_optimizer.optimizer.utils.offload import CPUOffloadOptimizer
from pytorch_optimizer.optimizer.utils.partition import BlockPartitioner, PreConditionerType
from pytorch_optimizer.optimizer.utils.precision import (
    StochasticAccumulator,
    copy_stochastic,
    has_overflow,
    to_real,
)
from pytorch_optimizer.optimizer.utils.preconditioner import PreConditioner
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims

__all__ = [
    'HAS_TRANSFORMERS',
    'NS_COEFFICIENTS',
    'TORCH_VERSION_AT_LEAST_2_4',
    'TORCH_VERSION_AT_LEAST_2_8',
    'AdaGradGraft',
    'BlockPartitioner',
    'CPUOffloadOptimizer',
    'DTensor',
    'Graft',
    'LayerWiseGrafting',
    'NewtonSchulzWeight',
    'NewtonSchulzWeights',
    'PreConditioner',
    'PreConditionerType',
    'RMSPropGraft',
    'SGDGraft',
    'SQRTNGraft',
    'StochasticAccumulator',
    'batched_power_iteration',
    'build_graft',
    'clip_grad_norm',
    'compare_versions',
    'compute_power_newton_db',
    'compute_power_schur_newton',
    'compute_power_svd',
    'copy_stochastic',
    'disable_running_stats',
    'enable_running_stats',
    'get_global_gradient_norm',
    'get_newton_schulz_weights',
    'has_dtensor',
    'has_overflow',
    'is_deepspeed_zero3_enabled',
    'is_valid_parameters',
    'merge_small_dims',
    'normalize_gradient',
    'parse_pytorch_version',
    'power_iteration',
    'reg_noise',
    'to_real',
    'unit_norm',
    'zero_power_via_newton_schulz_5',
]
