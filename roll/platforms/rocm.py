from .platform import Platform
from roll.utils.logging import get_logger

import torch

logger = get_logger()


class RocmPlatform(Platform):
    device_name: str = "AMD"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "HIP_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def get_vllm_worker_class(clas):
        try:
            from vllm import envs

            if envs.VLLM_USE_V1:
                from vllm.v1.worker.gpu_worker import Worker

                logger.info("Successfully imported vLLM V1 Worker.")
                return Worker
            else:
                from vllm.worker.worker import Worker

                logger.info("Successfully imported vLLM V0 Worker.")
                return Worker
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. " "Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def set_allocator_settings(cls) -> None:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        }
        return env_vars

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF": "",
            "HIP_VISIBLE_DEVICES": f"{gpu_rank}",
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            # "NCCL_DEBUG_SUBSYS":"INIT,COLL",
            # "NCCL_DEBUG":"INFO",
            # "NCCL_DEBUG_FILE":"rccl.%h.%p.log",
            # "NCCL_P2P_DISABLE":"1",
        }
        return env_vars