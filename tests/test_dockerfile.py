import re
from pathlib import Path
import pytest

def test_dockerfile_unsloth_version_match():
    """
    Validates that the PyTorch and CUDA versions in the Dockerfile base image
    match the versions expected by the Unsloth pip package.
    This prevents Dependabot from silently breaking the image by upgrading
    PyTorch without updating the unsloth installation tag.
    """
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        pytest.skip("Dockerfile not found in current directory")

    content = dockerfile_path.read_text()

    # Find the base image version
    # e.g., FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
    from_match = re.search(r"FROM\s+pytorch/pytorch:([0-9]+)\.([0-9]+)[.0-9]*-cuda([0-9]+)\.([0-9]+)-", content)
    assert from_match is not None, "Could not parse PyTorch and CUDA versions from FROM instruction in Dockerfile"

    torch_major, torch_minor, cuda_major, cuda_minor = from_match.groups()
    base_cuda_str = f"{cuda_major}{cuda_minor}"     # 12.1 -> 121

    # Find the unsloth install extra
    # e.g., "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
    unsloth_match = re.search(r"unsloth\[cu([0-9]+)-torch([0-9]+)\]", content)
    assert unsloth_match is not None, "Could not parse CUDA and PyTorch versions from unsloth pip install in Dockerfile"

    unsloth_cuda_str, unsloth_torch_str = unsloth_match.groups()

    # Assert CUDA matches exactly
    assert base_cuda_str == unsloth_cuda_str, (
        f"CUDA version mismatch in Dockerfile! Base image has CUDA {cuda_major}.{cuda_minor} (cu{base_cuda_str}), "
        f"but unsloth is configured to install for cu{unsloth_cuda_str}."
    )
    
    # Assert PyTorch major and minor match (e.g., base '2.4.x' matches unsloth 'torch240')
    expected_torch_prefix = f"{torch_major}{torch_minor}"
    assert unsloth_torch_str.startswith(expected_torch_prefix), (
        f"PyTorch version mismatch in Dockerfile! Base image is PyTorch {torch_major}.{torch_minor}.x, "
        f"but unsloth is configured for PyTorch {unsloth_torch_str[0]}.{unsloth_torch_str[1]} (torch{unsloth_torch_str}). "
        f"If you update the base image, you MUST update the unsloth extra (e.g., to torch{expected_torch_prefix}0)."
    )
