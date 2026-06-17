from brax.training.distribution import NormalTanhDistribution, _NormalDistribution
import jax.numpy as jnp

class BoundedNormalTanhDistribution(NormalTanhDistribution):
    def __init__(self, event_size, min_log_std=-10.0, max_log_std=2.0):
        super().__init__(event_size)  # still sets up TanhBijector, param_size, etc.
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std

    def create_dist(self, parameters):
        loc, log_std = jnp.split(parameters, 2, axis=-1)
        log_std = self._min_log_std + 0.5 * (self._max_log_std - self._min_log_std) * (jnp.tanh(log_std) + 1)
        return _NormalDistribution(loc=loc, scale=jnp.exp(log_std))