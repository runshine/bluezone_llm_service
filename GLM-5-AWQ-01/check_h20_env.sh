#!/bin/bash
# H20-3e 环境检查脚本

echo "=========================================="
echo "  H20-3e 环境诊断"
echo "=========================================="
echo ""

# 检查 NVIDIA 驱动和 GPU
echo "[1/6] NVIDIA GPU 信息:"
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,memory.free,driver_version --format=csv
if [ $? -ne 0 ]; then
    echo "  ERROR: nvidia-smi 执行失败"
else
    echo "  OK"
fi
echo ""

# 检查 CUDA 版本
echo "[2/6] CUDA 版本:"
which nvcc > /dev/null 2>&1
if [ $? -eq 0 ]; then
    nvcc --version 2>/dev/null | grep "release"
else
    echo "  nvcc not in PATH, checking /usr/local/cuda..."
    if [ -f /usr/local/cuda/bin/nvcc ]; then
        /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release"
    else
        echo "  CUDA toolkit not found"
    fi
fi

# 检查 PyTorch CUDA 版本
echo ""
echo "[3/6] PyTorch CUDA 信息:"
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  cuDNN version: {torch.backends.cudnn.version()}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory // (1024**3)} GB)')
        print(f'    Compute capability: {props.major}.{props.minor}')
        print(f'    Multi-process count: {props.multi_processor_count}')
"

echo ""
echo "[4/6] vLLM 版本:"
python3 -c "
try:
    import vllm
    print(f'  vLLM version: {vllm.__version__}')
except:
    try:
        import subprocess
        result = subprocess.run(['vllm', '--version'], capture_output=True, text=True)
        print(f'  {result.stdout.strip()}')
    except:
        print('  vLLM 未安装或不在 PATH 中')
"

echo ""
echo "[5/6] 检查 AWQ 量化支持:"
python3 -c "
try:
    import vllm
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print(f'  支持的量化方法:')
    for method in QUANTIZATION_METHODS.keys():
        print(f'    - {method}')
except Exception as e:
    print(f'  Error: {e}')
"

echo ""
echo "[6/6] 内存检查:"
free -h 2>/dev/null || echo "  free command not available"

echo ""
echo "=========================================="
echo "  诊断完成"
echo "=========================================="

