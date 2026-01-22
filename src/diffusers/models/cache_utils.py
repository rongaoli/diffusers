from contextlib import contextmanager

from ..utils.logging import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

class CacheMixin:


    _cache_config = None

    @property
    def is_cache_enabled(self) -> bool:
        return self._cache_config is not None

    def enable_cache(self, config) -> None:
        
        """r"""

        from ..hooks import (
            FasterCacheConfig,
            FirstBlockCacheConfig,
            PyramidAttentionBroadcastConfig,
            TaylorSeerCacheConfig,
            apply_faster_cache,
            apply_first_block_cache,
            apply_pyramid_attention_broadcast,
            apply_taylorseer_cache,
        )

        if self.is_cache_enabled:
            raise ValueError(
                f"Caching has already been enabled with {type(self._cache_config)}. To apply a new caching technique, please disable the existing one first."
            )

        if isinstance(config, FasterCacheConfig):
            apply_faster_cache(self, config)
        elif isinstance(config, FirstBlockCacheConfig):
            apply_first_block_cache(self, config)
        elif isinstance(config, PyramidAttentionBroadcastConfig):
            apply_pyramid_attention_broadcast(self, config)
        elif isinstance(config, TaylorSeerCacheConfig):
            apply_taylorseer_cache(self, config)
        else:
            raise ValueError(f"Cache config {type(config)} is not supported.")

        self._cache_config = config

    def disable_cache(self) -> None:
        from ..hooks import (
            FasterCacheConfig,
            FirstBlockCacheConfig,
            HookRegistry,
            PyramidAttentionBroadcastConfig,
            TaylorSeerCacheConfig,
        )
        from ..hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK
        from ..hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
        from ..hooks.pyramid_attention_broadcast import _PYRAMID_ATTENTION_BROADCAST_HOOK
        from ..hooks.taylorseer_cache import _TAYLORSEER_CACHE_HOOK

        if self._cache_config is None:
            logger.warning("Caching techniques have not been enabled, so there's nothing to disable.")
            return

        registry = HookRegistry.check_if_exists_or_initialize(self)
        if isinstance(self._cache_config, FasterCacheConfig):
            registry.remove_hook(_FASTER_CACHE_DENOISER_HOOK, recurse=True)
            registry.remove_hook(_FASTER_CACHE_BLOCK_HOOK, recurse=True)
        elif isinstance(self._cache_config, FirstBlockCacheConfig):
            registry.remove_hook(_FBC_LEADER_BLOCK_HOOK, recurse=True)
            registry.remove_hook(_FBC_BLOCK_HOOK, recurse=True)
        elif isinstance(self._cache_config, PyramidAttentionBroadcastConfig):
            registry.remove_hook(_PYRAMID_ATTENTION_BROADCAST_HOOK, recurse=True)
        elif isinstance(self._cache_config, TaylorSeerCacheConfig):
            registry.remove_hook(_TAYLORSEER_CACHE_HOOK, recurse=True)
        else:
            raise ValueError(f"Cache config {type(self._cache_config)} is not supported.")

        self._cache_config = None

    def _reset_stateful_cache(self, recurse: bool = True) -> None:
        from ..hooks import HookRegistry

        HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(recurse=recurse)

    @contextmanager
    def cache_context(self, name: str):
        
        """r"""
        from ..hooks import HookRegistry

        registry = HookRegistry.check_if_exists_or_initialize(self)
        registry._set_context(name)

        yield

        registry._set_context(None)
