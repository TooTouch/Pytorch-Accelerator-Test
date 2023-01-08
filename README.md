# Accelerator Test

Accelerator 실험

# Environments

```
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

# Accelerate Config

`accelerate config`를 설정하면 `accelerate launch`를 통해 사전에 정의한 config를 활용할 수 있음

1. CPU or GPU
2. Distributed Training
3. torch dynamo
4. DeepSpeed
5. GPU device
6. FP16 or BF16

config 가 완료되면 `accelerate env`를 통해 설정한 config를 확인할 수 있음

```
root@cifar_test:/projects/projects/Accerator_Practice# accelerate env

Copy-and-paste the text below in your GitHub issue

- `Accelerate` version: 0.15.0
- Platform: Linux-5.4.0-135-generic-x86_64-with-glibc2.29
- Python version: 3.8.10
- Numpy version: 1.22.2
- PyTorch version (GPU?): 1.14.0a0+410ce96 (True)
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: NO
        - mixed_precision: fp16
        - use_cpu: False
        - dynamo_backend: NO
        - num_processes: 1
        - machine_rank: 0
        - num_machines: 1
        - gpu_ids: all
        - main_process_ip: None
        - main_process_port: None
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - deepspeed_config: {}
        - fsdp_config: {}
        - megatron_lm_config: {}
        - downcast_bf16: no
        - tpu_name: None
        - tpu_zone: None
        - command_file: None
        - commands: None
```

# Experiments

**Setting**

- **Dataset**: CIFAR-100
- **Model**: ResNet-50
- **Optimizer**: SGD(lr=0.1)
- **Learning rate scheduler**: Cosine Annealing
- **Batch size**: 512
- **Epochs**: 200
- **Augmentations**: RandAugment(3,9)

**Test**

1. Default

```
bash scripts/run_default.sh
```

2. FP16

```
bash scripts/run_fp16.sh
```

3. BF16

```
bash scripts/run_bf16.sh
```

4. Gradients Accumulation Steps 4

```
bash scripts/run_grad_accum_steps.sh
```

5. Load Checkpoints

```
bash scripts/run_ckp.sh
```



# Trouble Shooting

## 1. `import accelerate` 시 아래와 같은 ImportError 발생

이유는 torch version 1.13.5 부터 `_LRScheduler` 에서 `LRScheduler`로 바뀜

```
File /usr/local/lib/python3.8/dist-packages/accelerate/accelerator.py:102
    100     from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
    101 else:
--> 102     from torch.optim.lr_scheduler import LRScheduler as LRScheduler
    104 logger = get_logger(__name__)
    107 class Accelerator:

ImportError: cannot import name 'LRScheduler' from 'torch.optim.lr_scheduler' (/usr/local/lib/python3.8/dist-packages/torch/optim/lr_scheduler.py)
```

docker image의 torch version은 1.14.0 지만 `_LRScheduler`에서 수정되어있지 않음.
아래와 같이 `/usr/local/lib/python3.8/dist-packages/accelerate/accelerator.py`의 조건이 1.13.5 이하인 경우로 되어 있어서 version 명 변경

**변경 전**

```
if is_torch_version("<=", "1.13.5"):
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
else:
    from torch.optim.lr_scheduler import LRScheduler as LRScheduler
```

**변경 후**

```
if is_torch_version("<=", "1.14.0"):
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
else:
    from torch.optim.lr_scheduler import LRScheduler as LRScheduler
```