import torch
from enum import Enum


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_device_name() -> str:
    """Get device type name"""
    if is_cuda_available:
        return torch.cuda.get_device_name().upper()
    elif is_npu_available:
        return torch.npu.get_device_name().upper()
    else:
        raise RuntimeError("Can not find device type.")


class DeviceType(Enum):
    NONE = "NONE"
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    ASCEND = "ASCEND"
    UNKNOWN = "UNKNOWN"


class GPUUtils:
    @staticmethod
    def get_device_type() -> DeviceType:
        if not is_cuda_available and not is_npu_available:
            return DeviceType.NONE

        device_name = get_device_name()
        if "NVIDIA" in device_name:
            return DeviceType.NVIDIA
        elif "AMD" in device_name:
            return DeviceType.AMD
        elif "ASCEND" in device_name:
            return DeviceType.ASCEND

        return DeviceType.UNKNOWN
