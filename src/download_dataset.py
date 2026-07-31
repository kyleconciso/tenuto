import os
import subprocess
import argparse

ASAP_REPO_URL = "https://github.com/fosfrancesco/asap-dataset.git"
PIANOCORE_HF_DATASET = "SyMuPe/PianoCoRe"

def download_asap_dataset(target_dir: str = "./data/asap", force: bool = False):
    """Downloads ASAP dataset."""
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

def download_pianocore_dataset(target_dir: str = "./data/pianocore", subset: str = "PianoCoRe-A*", token: str = None):
    """
    Downloads specific subset of PianoCoRe (default: 'PianoCoRe-A*' for aligned high-quality score-performance pairs).
    Supports HF token and multi-threaded fast transfers.
    """
    os.makedirs(target_dir, exist_ok=True)
    hf_token = token or os.environ.get("HF_TOKEN")
    print(f"[TenutoData] Fetching PianoCoRe subset '{subset}' from Hugging Face '{PIANOCORE_HF_DATASET}' (Fast Multithreaded Download)...")
    try:
        from huggingface_hub import snapshot_download
        # Enable high-speed parallel downloads
        snapshot_download(
            repo_id=PIANOCORE_HF_DATASET,
            repo_type="dataset",
            allow_patterns=["data/*.parquet", "*.json", "*.csv", "*.md"],
            local_dir=target_dir,
            token=hf_token,
            max_workers=8
        )
        print(f"[TenutoData] Successfully downloaded PianoCoRe subset '{subset}' to '{target_dir}'.")
    except ImportError:
        print("[TenutoData] `huggingface_hub` not installed. Run `pip install huggingface_hub`.")
    except Exception as e:
        print(f"[TenutoData] PianoCoRe download notice: {e}")
    return target_dir

def main():
    parser = argparse.ArgumentParser(description="Download Datasets for Tenuto")
    parser.add_argument("--dataset", type=str, default="combined", choices=["asap", "pianocore", "combined"])
    parser.add_argument("--pianocore_subset", type=str, default="PianoCoRe-A*", choices=["PianoCoRe-A*", "PianoCoRe-A", "PianoCoRe-B"])
    parser.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face access token for faster rate limits")
    parser.add_argument("--target_dir", type=str, default="./data", help="Base target directory")
    args = parser.parse_args()

    if args.dataset in ["asap", "combined"]:
        download_asap_dataset(os.path.join(args.target_dir, "asap"))
    if args.dataset in ["pianocore", "combined"]:
        download_pianocore_dataset(os.path.join(args.target_dir, "pianocore"), subset=args.pianocore_subset, token=args.hf_token)

if __name__ == "__main__":
    main()
