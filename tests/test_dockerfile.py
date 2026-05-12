import re
from pathlib import Path
import pytest

def test_dockerfile_axolotl_base_image():
    """
    Validates that the Dockerfile uses an official Axolotl base image.
    """
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        pytest.skip("Dockerfile not found in current directory")

    content = dockerfile_path.read_text()

    # Find the base image version
    # e.g., FROM winglian/axolotl:main-py3.10-cu121-2.1.2
    from_match = re.search(r"FROM\s+winglian/axolotl:main-py", content)
    assert from_match is not None, "Could not parse Axolotl base image from FROM instruction in Dockerfile"

def test_dockerfile_axolotl_entrypoint():
    """
    Validates that the Dockerfile entrypoint correctly invokes axolotl.
    """
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        pytest.skip("Dockerfile not found in current directory")

    content = dockerfile_path.read_text()
    
    assert "ENTRYPOINT [\"accelerate\", \"launch\", \"-m\", \"axolotl.cli.train\"" in content, "Dockerfile ENTRYPOINT does not invoke axolotl.cli.train correctly via accelerate launch"
