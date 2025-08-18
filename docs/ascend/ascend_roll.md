# ROLL x Ascend

Last updated: 08/15/2025.

我们在 ROLL 上增加对华为昇腾设备的支持。

## 硬件支持

Atlas 200T A2 Box16

Atlas 900 A2 PODc


## 安装


### 基础环境准备

| software  | version     |
|-----------|-------------|
| Python    |  3.10     |
| CANN      |  8.1.RC1  |

### 创建 conda 环境


使用以下命令在 Miniconda 中创建新的 conda 环境：

```
conda create --name roll python=3.10
conda activate roll
```

### 安装 torch & torch_npu:


为了能在 ROLL 中正常使用 torch 和 torch_npu，需使用以下命令安装 torch 和 torch_npu。请注意根据机器类型区分安装方式。

```
# 安装 torch 的 CPU 版本
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 安装 torch_npu
pip install torch_npu==2.5.1
```


### 安装vllm & vllm-ascend:

为了能够在 ROLL 中正常使用 vllm，需使用以下命令编译安装 vllm 和 vllm-ascend。请注意根据机器类型区分安装方式。

```
# vllm
git clone -b v0.8.4 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm

VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..
```

``` 
# vllm-ascend
git clone -b v0.8.4rc2 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

export COMPILE_CUSTOM_KERNELS=1
pip install -e .
cd ..
```

如果在安装 vllm-ascend 时遇到类似以下问题：

```
RuntimeError: CMake configuration failed: Command '['/pathto/miniconda3/envs/roll/bin/python3.10', '-m', 'pybind11', '--cmake']' returned non-zero exit status 2.
```

可尝试在 vllm-ascend 目录下 setup.py 文件 151-158 行进行如下修改并重新进行编译：

```
try:
    # if pybind11 is installed via pip
    pybind11_cmake_path = (subprocess.check_output(
        [python_executable, "-m", "pybind11",
        "--cmakedir"]).decode().strip())
except subprocess.CalledProcessError as e:
    # else specify pybind11 path installed from source code on CI container
    raise RuntimeError(f"CMake configuration failed: {e}")
```

### 安装 ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install deepspeed==0.16.0
cd ..
```

### 其他三方库说明

| software                       | description   |
|-------------------------------|---------------|
| transformers                  | v4.52.4       |
| flash_attn                    | not supported |
| tensordict                    | 0.8.3 (ARM)   |
| transformer-engine[pytorch]   | not supported |

1. 支持通过 transformers 使能 --flash_attention_2， transformers 需大于等于 4.52.0版本。
2. 不支持通过 flash_attn 使能 flash attention 加速。
3. 针对 ARM 服务器，tensordict 要求 0.8.3，可在依赖安装完成后再手动安装 tensordict。
4. 暂不支持 transformer-engine[pytorch] 

```
pip install transformers==4.52.4
pip install tensordict==0.8.3
```

## 快速开始，单节点部署指引

正式使用前，建议您通过对单节点流水线的训练尝试以检验环境准备和安装的正确性。
由于目前暂不支持 Megatron-LM 训练，请首先将对应文件中 
strategy_args 参数修改为 deepspeed 选项。

1. 使用 shell 执行单节点流水线

```
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh  
```

2. 使用配置文件执行 agentic pipeline

```
# 确保当前位于ROLL项目目录的根目录下
# export PYTHONPATH=$(pwd):$PYTHONPATH

python examples/start_agentic_pipeline.py \
        --config_path qwen2.5-0.5B-agentic \
        --config_name agentic_val_sokoban

- ``--config_path`` – 包含您的YAML配置文件的目录。
- ``--config_name`` – 文件名（不含.yaml后缀）。
```

## 支持现状


**表1** NPU 已通过流水线验证

| pipeline                                                      | hardware            |
|---------------------------------------------------------------|---------------------|
| examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh | Atlas 900 A2 PODc   |
| examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh  | Atlas 900 A2 PODc   |
| examples/qwen2.5-1.5B-distill_ds/run_distill_pipeline.sh      | Atlas 900 A2 PODc   |
| examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh          | Atlas 900 A2 PODc   |
| examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh        | Atlas 900 A2 PODc   |

**表2** NPU 待流水线验证

| pipeline                                                      | hardware            |
|---------------------------------------------------------------|---------------------|
| examples/qwen2.5-vl-7B-distill/run_distill_vl_ds_pipeline.sh  | Atlas 900 A2 PODc   |
| examples/qwen2.5-vl-7B-rlvr/run_rlvr_pipeline.sh              | Atlas 900 A2 PODc   |

## 后续计划


分别按照以下规则进行与 GPU 的精度与吞吐量的对比
精度对比:
根据经验，对于 Agentic 和 RLVR 等 RL 类算法，我们期望在相同配置下华为昇腾设备与 A100 的 rewards 平均绝对误差 <= 4%，计算方式参考下公式。
```
$ Mean Error = \frac{\sum_{i=1}^{N} |reward_i^{npu} - reward_{i}^{gpu}|}{N}  \leq 0.04 $
```
对于 DPO 和 Distill 等类算法，我们期望在相同配置下华为昇腾设备与 A100 的 loss 相对误差 <= 4%，计算方式参考下公式。
```
$ Mean Error = \frac{\sum_{i=1}^{N} |loss_i^{npu} - loss_{i}^{gpu}|}{N}  \leq 0.04 $
```

吞吐对比：Ascend npu 和 A100 分别取日志中前4个 step 的 throughput 的 tpu 值 做平均， tpu ratio = npu 平均值 / A100 平均值。


## 声明
-----------------------------------
ROLL 中提供的 Ascend 支持代码皆为参考样例，商业使用请通过官方正式途径沟通，谢谢。
