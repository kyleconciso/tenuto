import os
import subprocess
import argparse

ASAP_REPO_URL = "https://github.com/vocaloid-team/asap-dataset.git"

def download_asap_dataset(target_dir: str = "./data/asap", force: bool = False):
    """
    Downloads / clones the ASAP (Aligned Scores and Performances) dataset.
    """
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    if os.path.exists(target_dir) and not force:
        print(f"[TenutoData] ASAP dataset already exists at '{target_dir}'. Skipping download.")
        return target_dir

    print(f"[TenutoData] Cloning ASAP dataset from '{ASAP_REPO_URL}' into '{target_dir}'...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", ASAP_REPO_URL, target_dir], check=True)
        print(f"[TenutoData] Successfully downloaded ASAP dataset to '{target_dir}'.")
    except Exception as e:
        print(f"[TenutoData] Git clone failed: {e}. You can manually download ASAP dataset to '{target_dir}'.")

    return target_dir

def main():
    parser = argparse.ArgumentParser(description="Download ASAP Dataset for Tenuto Training")
    parser.add_argument("--target_dir", type=str, default="./data/asap", help="Target directory for ASAP dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download if dataset exists")
    args = parser.parse_args()

    download_asap_dataset(target_dir=args.target_dir, force=args.force)

if __name__ == "__main__":
    main()
