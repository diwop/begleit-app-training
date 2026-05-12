from pathlib import Path
import pytest

def test_dockerfile_axolotl_entrypoint():
    """
    Validates that the Dockerfile entrypoint correctly invokes axolotl.
    """
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        pytest.skip("Dockerfile not found in current directory")

    content = dockerfile_path.read_text()
    
    assert "ENTRYPOINT [\"accelerate\", \"launch\", \"-m\", \"axolotl.cli.train\"" in content, "Dockerfile ENTRYPOINT does not invoke axolotl.cli.train correctly via accelerate launch"
