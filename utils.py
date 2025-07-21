"""辅助函数集合，包含分布式初始化、文件路径处理等工具。"""

import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List


def get_open_port():
    """在本机上找到一个空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 绑定到随机端口并立即返回该端口号
        s.bind(('', 0))
        return s.getsockname()[1]


def get_remote_file(remote_path, local_path=None):
    """从远程机器拷贝文件到本地，如已在本地则直接返回路径。"""
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """仅在 rank0 进程打印日志。"""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """根据给定的本地目录前缀，返回当前用户的缓存目录路径。"""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """为当前实验创建唯一的输出目录并返回其路径。"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """将批次切分到不同进程，并移动到指定设备上。"""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    """在指定维度上将张量填充到给定长度。"""
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """在多进程环境下收集并拼接所有进程的张量。"""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """将字典中的浮点数格式化为更简洁的字符串表示。"""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """关闭模型中所有 Dropout 层，避免推理/评测阶段的随机性。"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """打印当前进程占用的 GPU 显存，方便调试内存问题。"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """根据名称在模型中查找对应的模块类。"""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    """根据模型类文件动态加载并返回指定的模块类。"""
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    """初始化分布式环境，设置地址与端口并指定 GPU。"""
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    """上下文管理器：在进入时设置随机种子，退出时恢复原状态。"""

    def __init__(self, seed):
        # 记录种子及原始随机状态
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # 保存进入前的随机状态
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # 设置新的随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复之前保存的随机状态
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
