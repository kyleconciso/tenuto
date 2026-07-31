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
