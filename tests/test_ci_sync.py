import re
from pathlib import Path

def test_dockerfile_and_pr_workflow_sync():
    # Dynamically resolve the project root (assuming this file is in tests/)
    project_root = Path(__file__).parent.parent
    
    dockerfile_path = project_root / "Dockerfile"
    pr_workflow_path = project_root / ".github" / "workflows" / "pull-request.yml" 

    assert dockerfile_path.exists(), f"Missing Dockerfile at {dockerfile_path}"
    assert pr_workflow_path.exists(), f"Missing PR workflow at {pr_workflow_path}"

    # Extract base image from Dockerfile
    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
    match = re.search(r"^FROM\s+(?:--platform=\S+\s+)?([^\s]+)", dockerfile_content, re.MULTILINE)
    
    assert match is not None, "Could not find a valid 'FROM' statement in the Dockerfile."
    base_image = match.group(1)

    # Verify it exists in the Pull Request workflow
    pr_workflow_content = pr_workflow_path.read_text(encoding="utf-8")
    
    error_msg = (
        f"\n[CRITICAL MISMATCH] \n"
        f"Dockerfile uses base image : {base_image}\n"
        f"However, this image was not found in {pr_workflow_path.name}.\n"
        f"Please update the 'container: image:' line in your PR workflow to match!"
    )
    
    assert base_image in pr_workflow_content, error_msg

    print(f"Success! Both CI environments are perfectly synced to: {base_image}")

if __name__ == "__main__":
    test_dockerfile_and_pr_workflow_sync()