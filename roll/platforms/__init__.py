import logging
from .platform import Platform
from .cuda import CudaPlatform
from .npu import NPUPlatform

logger = logging.getLogger(__name__)

def _init_platform() -> Platform:
    """Initialize and return the current platform instance.

    Automatically selects the platform based on environment and availability:
    - If torch_npu is installed, use NPUPlatform.
    - Otherwise, fall back to CudaPlatform.
    """
    try:
        import torch_npu  # noqa: F401
        logger.info("Detected torch_npu. Initializing NPU platform.")
        return NPUPlatform()
    except ImportError:
        logger.info("Initializing ROLL default device backend: Cuda platform.")
        return CudaPlatform()

# Global singleton representing the current platform in use.
current_platform: Platform = _init_platform()

__all__ = [
    "Platform",
    "current_platform",
]