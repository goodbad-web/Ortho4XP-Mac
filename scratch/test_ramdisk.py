import sys
import os
import shutil

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import O4_RAMDisk_Utils
import O4_File_Names as FNAMES

def run_test():
    print("=== Ortho4XP RAM Disk Test ===")
    
    tmp_abs_path = os.path.abspath(FNAMES.Tmp_dir)
    print(f"Target tmp path: {tmp_abs_path}")
    
    # 1. Mount RAM Disk (using 2GB for test)
    print("\n--- 1. Testing Mount ---")
    mounted = O4_RAMDisk_Utils.mount_ram_disk(size_gb=2)
    if not mounted:
        print("ERROR: Mount failed!")
        return False
    print("Mount success.")
    
    # Check status
    ram_disk_path = "/Volumes/Ortho4XP_RAM_Disk"
    if not os.path.exists(ram_disk_path):
        print(f"ERROR: {ram_disk_path} does not exist!")
        return False
        
    if not os.path.islink(tmp_abs_path):
        print(f"ERROR: {tmp_abs_path} is not a symlink!")
        return False
        
    target = os.readlink(tmp_abs_path)
    if target != ram_disk_path:
        print(f"ERROR: symlink points to {target} instead of {ram_disk_path}!")
        return False
        
    print("Mount verification passed.")
    
    # 2. Test writing a file
    print("\n--- 2. Testing Write ---")
    test_file_path = os.path.join(tmp_abs_path, "test_ramdisk_write.txt")
    try:
        with open(test_file_path, "w") as f:
            f.write("Hello from Ortho4XP RAM Disk!")
        
        # Verify file exists and reads correctly
        with open(test_file_path, "r") as f:
            content = f.read()
        print(f"File write/read content: '{content}'")
        if content != "Hello from Ortho4XP RAM Disk!":
            print("ERROR: File content mismatch!")
            return False
        print("Write verification passed.")
    except Exception as e:
        print(f"ERROR: File writing failed: {e}")
        return False
        
    # 3. Test double mounting (should not fail, should report already mounted)
    print("\n--- 3. Testing Double Mount ---")
    double_mounted = O4_RAMDisk_Utils.mount_ram_disk(size_gb=2)
    if not double_mounted:
        print("ERROR: Double mount failed!")
        return False
    print("Double mount verification passed.")
    
    # 4. Test Unmounting
    print("\n--- 4. Testing Unmount ---")
    unmounted = O4_RAMDisk_Utils.unmount_ram_disk()
    if not unmounted:
        print("ERROR: Unmount failed!")
        return False
    print("Unmount success.")
    
    # Check status
    if os.path.exists(ram_disk_path):
        print(f"ERROR: {ram_disk_path} still exists after unmount!")
        return False
        
    if os.path.islink(tmp_abs_path):
        print(f"ERROR: {tmp_abs_path} is still a symlink after unmount!")
        return False
        
    if not os.path.isdir(tmp_abs_path):
        print(f"ERROR: {tmp_abs_path} is not a directory after unmount!")
        return False
        
    print("Unmount verification passed.")
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
