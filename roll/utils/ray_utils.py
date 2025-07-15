import os

import ray


@ray.remote
def get_visible_gpus(ray_device_key: str):
    if ray_device_key == "GPU":
        return ray.get_gpu_ids()
    elif ray_device_key == "NPU":
        visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
        if visible_devices:
            return list(map(int, visible_devices.split(",")))
    else:
        return []


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
