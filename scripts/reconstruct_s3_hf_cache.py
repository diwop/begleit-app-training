#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import tempfile
import tarfile
import shutil
from pathlib import Path
import boto3
from huggingface_hub import HfApi

def get_sha256(file_path: Path) -> str:
    print(f"Calculating SHA256 for {file_path.name}...")
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024) # 1MB
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Reconstruct S3 flat model files into a proper Hugging Face cache tarball on S3.")
    parser.add_argument("--bucket", required=True, help="S3 Bucket name (e.g. diwop-leichte-sprache)")
    parser.add_argument("--model-id", default="google/gemma-4-26b-a4b-it", help="Hugging Face Model ID")
    parser.add_argument("--s3-prefix", default="hf_cache", help="S3 prefix under which models are stored")
    parser.add_argument("--profile", default=None, help="AWS Profile name to use")
    
    args = parser.parse_args()
    
    # Initialize boto3 session
    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")
    
    model_id = args.model_id
    model_dir_name = f"models--{model_id.replace('/', '--')}"
    
    # S3 source prefix e.g. hf_cache/google/gemma-4-26b-a4b-it/
    s3_src_prefix = f"{args.s3_prefix}/{model_id}/"
    s3_dst_tar = f"{args.s3_prefix}/{model_dir_name}.tar"
    
    print(f"Connecting to S3 bucket: {args.bucket}")
    print(f"Source prefix: s3://{args.bucket}/{s3_src_prefix}")
    print(f"Target tarball: s3://{args.bucket}/{s3_dst_tar}")
    
    # Fetch model commit SHA from Hugging Face Hub
    print(f"Fetching latest commit SHA for {model_id} from Hugging Face Hub...")
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        commit_sha = model_info.sha
        print(f"Commit SHA: {commit_sha}")
    except Exception as e:
        print(f"Error fetching model info from Hugging Face: {e}")
        sys.exit(1)
        
    # List files in S3 source prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=args.bucket, Prefix=s3_src_prefix)
    
    s3_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Skip folders and check-tar artifacts if they exist
            if not key.endswith('/'):
                s3_keys.append(key)
                
    if not s3_keys:
        print("No files found in S3 under the source prefix!")
        sys.exit(1)
        
    print(f"Found {len(s3_keys)} files to process.")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Local paths for cache reconstruction
        local_cache_root = tmp_path / "cache"
        local_model_dir = local_cache_root / model_dir_name
        
        blobs_dir = local_model_dir / "blobs"
        refs_dir = local_model_dir / "refs"
        snapshots_dir = local_model_dir / "snapshots" / commit_sha
        
        blobs_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Write refs/main
        with open(refs_dir / "main", "w") as f:
            f.write(commit_sha)
            
        # Download and process files
        for key in s3_keys:
            filename = os.path.basename(key)
            local_file = tmp_path / filename
            
            print(f"Downloading s3://{args.bucket}/{key} -> {local_file}")
            s3.download_file(args.bucket, key, str(local_file))
            
            # Compute SHA-256
            sha = get_sha256(local_file)
            
            # Move to blobs
            blob_dest = blobs_dir / sha
            print(f"Moving file to blobs/{sha}")
            shutil.move(str(local_file), str(blob_dest))
            
            # Create relative symlink in snapshots: ../../blobs/<sha>
            symlink_dest = snapshots_dir / filename
            print(f"Creating symlink: {symlink_dest.name} -> ../../blobs/{sha}")
            os.symlink(f"../../blobs/{sha}", str(symlink_dest))
            
        # Create tar file preserving symlinks
        local_tar = tmp_path / f"{model_dir_name}.tar"
        print(f"Creating tarball: {local_tar}")
        with tarfile.open(local_tar, "w") as tar:
            # We add the model_dir_name directory into the tarball
            tar.add(str(local_model_dir), arcname=model_dir_name)
            
        # Upload tar file to S3
        print(f"Uploading tarball to S3: s3://{args.bucket}/{s3_dst_tar}")
        s3.upload_file(str(local_tar), args.bucket, s3_dst_tar)
        print("Success! S3 cache has been reconstructed and uploaded.")

if __name__ == "__main__":
    main()
