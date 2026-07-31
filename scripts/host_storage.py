import os
import sys
import time
import re
import subprocess

def start_storage_host(port=8080, storage_dir="./storage"):
    os.makedirs(storage_dir, exist_ok=True)
    abs_storage = os.path.abspath(storage_dir)

    print(f"=================================================================")
    print(f"            TENUTO SELF-HOSTED STORAGE SERVER                   ")
    print(f"=================================================================")
    print(f"  • Local Directory:  {abs_storage}")
    print(f"  • Local WebDAV Port: {port}")
    print(f"  • User / Password:   tenuto / tenuto")
    print(f"-----------------------------------------------------------------")
    print(f"Starting rclone WebDAV server...")

    # 1. Start rclone serve webdav
    rclone_cmd = [
        "rclone", "serve", "webdav", abs_storage,
        "--addr", f"127.0.0.1:{port}",
        "--user", "tenuto",
        "--pass", "tenuto"
    ]
    
    try:
        rclone_proc = subprocess.Popen(rclone_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Failed to start rclone: {e}. Make sure rclone is installed!")
        sys.exit(1)

    time.sleep(1)
    if rclone_proc.poll() is not None:
        print("rclone exited unexpectedly. Check if port is already in use.")
        sys.exit(1)

    print("rclone WebDAV server is active!")
    print("Establishing secure public tunnel via localhost.run...")

    # 2. Start SSH reverse tunnel via localhost.run
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-R", f"80:127.0.0.1:{port}",
        "nokey@localhost.run"
    ]

    ssh_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    public_url = None
    for line in ssh_proc.stdout:
        print("  [Tunnel Log]", line.strip())
        if "admin.localhost.run" in line:
            continue
        match = re.search(r"https://[a-zA-Z0-9-]+\.(?:lhr\.life|localhost\.run|lhr\.rocks)", line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n=================================================================")
        print("             🎉 SUCCESS! YOUR STORAGE IS LIVE!                  ")
        print("=================================================================")
        print(f" Public WebDAV URL:  {public_url}")
        print(f" Target Directory:   {abs_storage}")
        print("-----------------------------------------------------------------")
        print(" Copy and paste this command into Google Colab to connect:")
        print(f" %env HOST_STORAGE_URL={public_url}")
        print("=================================================================\n")
        print("Press Ctrl+C to stop the storage server at any time.")
    else:
        print("Could not retrieve public URL from tunnel log. Check output above.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Tenuto storage server...")
        rclone_proc.terminate()
        ssh_proc.terminate()
        print("Server stopped cleanly.")

if __name__ == "__main__":
    start_storage_host()
