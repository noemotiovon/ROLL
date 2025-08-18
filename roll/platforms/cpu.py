from .platform import Platform
from roll.utils.logging import get_logger


logger = get_logger()


class CpuPlatform(Platform):
    device_name: str = "CPU"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "CPU"
    communication_backend: str = "gloo"

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        pass

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        return {}

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank:str) -> dict:
        return {}