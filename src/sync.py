import os
import sys
import subprocess

def download_dataset_from_host(host_url: str, target_dir: str = "./data/processed"):
    """
    Downloads preprocessed dataset from remote WebDAV host using rclone or curl fallback.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"[TenutoSync] Syncing preprocessed dataset from host: {host_url} -> {target_dir}")
    
    # 1. Install rclone binary dynamically if missing in Colab environment
    rclone_bin = "rclone"
    if subprocess.call(["which", "rclone"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("[TenutoSync] Installing rclone binary...")
        try:
            subprocess.run("curl https://rclone.org/install.sh | bash", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # 2. Try rclone sync with WebDAV credentials and live progress stats
    try:
        pass_obs = os.popen("rclone obscure tenuto").read().strip() or "tenuto"
        connection_str = f':webdav,url="{host_url}",user="tenuto",pass="{pass_obs}":'
        cmd = [
            "rclone", "copy",
            connection_str, target_dir,
            "-P",
            "--stats", "2s",
            "--stats-one-line",
            "--transfers", "16",
            "--checkers", "16"
        ]
        print("[TenutoSync] Starting high-speed parallel sync (16 streams)...")
        subprocess.run(cmd, check=True)
        print("\n[TenutoSync] Dataset sync complete!")
    except Exception as e:
        print(f"[TenutoSync] rclone sync notice: {e}. Trying wget fallback with auth...")
        try:
            subprocess.run([
                "wget", "--http-user=tenuto", "--http-password=tenuto",
                "-r", "-np", "-nH", "--cut-dirs=1",
                "--show-progress",
                f"{host_url}/", "-P", target_dir
            ], check=True)
        except Exception as ex:
            print(f"[TenutoSync] Failed to download dataset: {ex}")

def push_dataset_to_hf(dataset_dir: str = "./storage/processed", repo_id: str = "kyleconciso/tenuto-processed", token: str = None):
    """
    Pushes preprocessed dataset directly to private Hugging Face Dataset repository.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[TenutoSync] HF_TOKEN is required to upload dataset to Hugging Face.")
        return False
    
    print(f"[TenutoSync] Uploading preprocessed dataset from '{dataset_dir}' to Hugging Face '{repo_id}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=dataset_dir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="processed"
        )
        print(f"[TenutoSync] 🎉 Successfully uploaded preprocessed dataset to Hugging Face '{repo_id}'!")
        return True
    except Exception as e:
        print(f"[TenutoSync] HF Dataset upload error: {e}")
        return False

def pull_dataset_from_hf(target_dir: str = "./data/processed", repo_id: str = "kyleconciso/tenuto-processed", token: str = None):
    """
    Downloads preprocessed dataset from private Hugging Face Dataset repository in Colab.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    os.makedirs(target_dir, exist_ok=True)
    print(f"[TenutoSync] Downloading preprocessed dataset from Hugging Face '{repo_id}'...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            token=hf_token,
            max_workers=16
        )
        print(f"[TenutoSync] 🎉 Dataset downloaded from Hugging Face!")
        return True
    except Exception as e:
        print(f"[TenutoSync] HF Dataset download notice: {e}")
        return False

def upload_checkpoint_to_hf(checkpoint_path: str, repo_id: str = "kyleconciso/tenuto-model", token: str = None):
    """
    Uploads trained model checkpoint directly to private Hugging Face Model repository.
    """
    if not os.path.exists(checkpoint_path):
        return False

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        return False

    filename = os.path.basename(checkpoint_path)
    print(f"[TenutoSync] Syncing checkpoint '{filename}' to Hugging Face Model Hub '{repo_id}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"[TenutoSync] 🎉 Saved checkpoint '{filename}' to Hugging Face Hub!")
        return True
    except Exception as e:
        print(f"[TenutoSync] HF Checkpoint upload notice: {e}")
        return False

def download_checkpoint_from_hf(filename: str = "latest_transformer_model.pth", target_dir: str = "checkpoints", repo_id: str = "kyleconciso/tenuto-model", token: str = None):
    """
    Downloads latest checkpoint from Hugging Face Model Hub for auto-resuming training in Colab.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, filename)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=target_dir,
            token=hf_token
        )
        return out_path if os.path.exists(out_path) else None
    except Exception:
        return None

def upload_checkpoint_to_host(host_url: str, checkpoint_path: str):
    """
    Uploads trained model checkpoint to remote WebDAV host.
    """
    if not os.path.exists(checkpoint_path):
        return

    print(f"[TenutoSync] Uploading checkpoint to host: {checkpoint_path} -> {host_url}")
    filename = os.path.basename(checkpoint_path)
    
    try:
        # Use curl PUT request for WebDAV upload
        cmd = [
            "curl", "-s", "-X", "PUT",
            "-u", "tenuto:tenuto",
            "--upload-file", checkpoint_path,
            f"{host_url}/{filename}"
        ]
        subprocess.run(cmd, check=True)
        print(f"[TenutoSync] Successfully backed up {filename} to your local host!")
    except Exception as e:
        print(f"[TenutoSync] Failed to upload checkpoint to host: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--download":
        download_dataset_from_host(sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[1] == "--upload":
        upload_checkpoint_to_host(sys.argv[2], sys.argv[3])
