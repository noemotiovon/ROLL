# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import ray


@ray.remote
def get_visible_gpus():
    # 获取 ASCEND_RT_VISIBLE_DEVICES 环境变量
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible_devices:
        return list(map(int, visible_devices.split(',')))
    else:
        return []


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
