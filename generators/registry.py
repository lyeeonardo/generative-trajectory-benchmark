"""Joint proposal/world-model construction.

Builders receive a configuration dict and normalization statistics and return an
nn.Module with loss(batch), propose_joint(...), predict(...), and generate(batch).
Every model must support imposed-action prediction with NULL quality, shared
public conditioning, action bounds, observation/event output, and seeded sampling.
"""
from dataclasses import asdict
from collections.abc import Mapping
from generators.joint_world_model import JointConfig, JointWorldModel
from generators.cvae import CVAEConfig, JointCVAE

DISPLAY_NAMES = {
    'diffusion': 'Diffusion',
    'flow_matching': 'Flow Matching',
    'autoregressive': 'Autoregressive Transformer',
}

def _joint_builder(config, normalization):
    return JointWorldModel(JointConfig(**config), normalization)

_BUILDERS = {method: _joint_builder for method in DISPLAY_NAMES}
DISPLAY_NAMES['cvae'] = 'Joint CVAE'
_BUILDERS['cvae'] = lambda config, normalization: JointCVAE(CVAEConfig(**config), normalization)

def register_model(method, builder, *, display_name):
    """Explicit extension point; duplicate registrations fail rather than replace a model."""
    if not method or method in _BUILDERS:
        raise ValueError('Model name must be nonempty and unregistered')
    if not callable(builder):
        raise TypeError('Model builder must be callable')
    _BUILDERS[method] = builder
    DISPLAY_NAMES[method] = display_name

def available_models():
    return tuple(_BUILDERS)

def create_model(config, normalization):
    config = dict(config) if isinstance(config, Mapping) else asdict(config)
    method = config.get('method')
    if method not in _BUILDERS:
        raise ValueError(f'Unknown joint model {method!r}; available: {available_models()}.')
    model = _BUILDERS[method](config, normalization)
    for name in ('loss', 'propose_joint', 'predict', 'generate'):
        if not callable(getattr(model, name, None)):
            raise TypeError(f'Joint model must implement {name}()')
    return model
