import os
import subprocess
import argparse

ASAP_REPO_URL = "https://github.com/fosfrancesco/asap-dataset.git"
PIANOCORE_HF_DATASET = "SyMuPe/PianoCoRe"

def download_asap_dataset(target_dir: str = "./data/asap", force: bool = False):
    """Downloads ASAP (Aligned Scores and Performances) dataset."""
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    if os.path.exists(target_dir) and not force:
        print(f"[TenutoData] ASAP dataset already exists at '{target_dir}'. Skipping.")
        return target_dir

    print(f"[TenutoData] Cloning ASAP dataset from '{ASAP_REPO_URL}'...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", ASAP_REPO_URL, target_dir], check=True)
        print(f"[TenutoData] Downloaded ASAP dataset to '{target_dir}'.")
    except Exception as e:
        print(f"[TenutoData] Git clone failed: {e}.")
    return target_dir

def download_pianocore_dataset(target_dir: str = "./data/pianocore", subset: str = "PianoCoRe-A"):
    """Downloads PianoCoRe dataset (SyMuPe/PianoCoRe) from Hugging Face."""
    os.makedirs(target_dir, exist_ok=True)
    print(f"[TenutoData] Fetching PianoCoRe ({subset}) from Hugging Face '{PIANOCORE_HF_DATASET}'...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=PIANOCORE_HF_DATASET, repo_type="dataset", local_dir=target_dir)
        print(f"[TenutoData] Downloaded PianoCoRe dataset to '{target_dir}'.")
    except ImportError:
        print("[TenutoData] `huggingface_hub` not installed. Install with `pip install huggingface_hub`.")
    except Exception as e:
        print(f"[TenutoData] PianoCoRe download notice: {e}")
    return target_dir

def main():
    parser = argparse.ArgumentParser(description="Download Datasets for Tenuto")
    parser.add_argument("--dataset", type=str, default="asap", choices=["asap", "pianocore", "all"])
    parser.add_argument("--target_dir", type=str, default="./data", help="Base target directory")
    args = parser.parse_args()

    if args.dataset in ["asap", "all"]:
        download_asap_dataset(os.path.join(args.target_dir, "asap"))
    if args.dataset in ["pianocore", "all"]:
        download_pianocore_dataset(os.path.join(args.target_dir, "pianocore"))

if __name__ == "__main__":
    main()
