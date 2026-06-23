#!/usr/bin/env python3
import os
import sys
import hashlib
import tempfile
import boto3
from huggingface_hub import HfApi

def main():
    bucket_name = "diwop-leichte-sprache"
    model_id = "google/gemma-4-26b-a4b-it"
    profile_name = "diwop-production"
    
    model_dir_name = f"models--{model_id.replace('/', '--')}"
    s3_src_prefix = f"hf_cache/{model_id}/"
    s3_dst_prefix = f"hf_cache/{model_dir_name}/"
    
    print(f"Connecting to S3 bucket '{bucket_name}' using profile '{profile_name}'...")
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")
    
    # 1. Fetch model info from Hugging Face Hub
    print(f"Fetching metadata for '{model_id}' from Hugging Face Hub...")
    try:
        api = HfApi()
        model_info = api.model_info(model_id, files_metadata=True)
        commit_sha = model_info.sha
        print(f"Commit SHA: {commit_sha}")
    except Exception as e:
        print(f"Error fetching model metadata: {e}")
        sys.exit(1)
        
    # Map LFS filename -> SHA256
    lfs_map = {}
    for sibling in model_info.siblings:
        if sibling.lfs:
            lfs_map[sibling.rfilename] = sibling.lfs.get("sha256")
            
    # 2. List files in S3 source prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_src_prefix)
    
    s3_objects = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'):
                s3_objects.append(key)
                
    if not s3_objects:
        print(f"No objects found under prefix: s3://{bucket_name}/{s3_src_prefix}")
        sys.exit(0)
        
    print(f"Found {len(s3_objects)} files in S3 source prefix.")
    
    # 3. Process each object
    for key in s3_objects:
        filename = os.path.basename(key)
        
        # Determine target SHA256
        sha256 = None
        if filename in lfs_map:
            sha256 = lfs_map[filename]
            print(f"Found LFS SHA-256 for '{filename}' from HF API: {sha256}")
        
        if sha256:
            # We can perform a server-side copy for the large LFS files!
            dst_key = f"{s3_dst_prefix}blobs/{sha256}"
            copy_source = {'Bucket': bucket_name, 'Key': key}
            
            print(f"Server-side copying large LFS file:\n  Source: s3://{bucket_name}/{key}\n  Dest:   s3://{bucket_name}/{dst_key}")
            try:
                s3.copy(copy_source, bucket_name, dst_key)
                print(f"Deleting source file: s3://{bucket_name}/{key}")
                s3.delete_object(Bucket=bucket_name, Key=key)
            except Exception as e:
                print(f"Error copying/deleting {filename}: {e}")
        else:
            # For small non-LFS files, download, compute hash, upload to blobs
            print(f"Processing small non-LFS file: {filename}")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_name = tmp.name
                
            try:
                s3.download_file(bucket_name, key, tmp_name)
                
                # Compute SHA256
                sha = hashlib.sha256()
                with open(tmp_name, 'rb') as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        sha.update(chunk)
                file_sha = sha.hexdigest()
                
                # Upload to blobs
                dst_key = f"{s3_dst_prefix}blobs/{file_sha}"
                print(f"Uploading small file to blobs/{file_sha}...")
                s3.upload_file(tmp_name, bucket_name, dst_key)
                
                print(f"Deleting source file: s3://{bucket_name}/{key}")
                s3.delete_object(Bucket=bucket_name, Key=key)
            except Exception as e:
                print(f"Error processing non-LFS file {filename}: {e}")
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
                    
    # 4. Write refs/main
    refs_key = f"{s3_dst_prefix}refs/main"
    print(f"Writing commit SHA '{commit_sha}' to s3://{bucket_name}/{refs_key}...")
    try:
        s3.put_object(Bucket=bucket_name, Key=refs_key, Body=commit_sha.encode('utf-8'))
        print("Success!")
    except Exception as e:
        print(f"Error writing refs/main: {e}")

if __name__ == "__main__":
    main()
