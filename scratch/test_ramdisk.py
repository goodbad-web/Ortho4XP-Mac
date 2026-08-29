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
    ortho_abs_path = os.path.abspath(FNAMES.Imagery_dir)
    print(f"Target tmp path: {tmp_abs_path}")
    print(f"Target Orthophotos path: {ortho_abs_path}")
    
    # 0. Clean up any pre-existing RAM disk states to ensure fresh test boundaries
    print("Cleaning up any pre-existing RAM disk states...")
    O4_RAMDisk_Utils.unmount_ram_disk(use_orthophotos=True)
    
    # 0. Setup dummy orthophoto file to simulate existing cache
    os.makedirs(ortho_abs_path, exist_ok=True)
    dummy_cache_file = os.path.join(ortho_abs_path, "dummy_cache_test.txt")
    with open(dummy_cache_file, "w") as f:
        f.write("Important Cached Aerial Imagery!")
    print(f"Created dummy orthophoto cache file at: {dummy_cache_file}")
    
    # 1. Mount RAM Disk with Orthophotos enabled
    print("\n--- 1. Testing Mount with Orthophotos ---")
    mounted = O4_RAMDisk_Utils.mount_ram_disk(size_gb=2, use_orthophotos=True)
    if not mounted:
        print("ERROR: Mount failed!")
        return False
    print("Mount success.")
    
    # Check status
    ram_disk_path = "/Volumes/Ortho4XP_RAM_Disk"
    if not os.path.exists(ram_disk_path):
        print(f"ERROR: {ram_disk_path} does not exist!")
        return False
        
    # Check tmp symlink
    if not os.path.islink(tmp_abs_path):
        print(f"ERROR: {tmp_abs_path} is not a symlink!")
        return False
    print("Tmp symlink verified.")
        
    # Check Orthophotos symlink
    if not os.path.islink(ortho_abs_path):
        print(f"ERROR: {ortho_abs_path} is not a symlink!")
        return False
    target = os.readlink(ortho_abs_path)
    expected_target = os.path.join(ram_disk_path, "Orthophotos")
    if target != expected_target:
        print(f"ERROR: Orthophotos symlink points to {target} instead of {expected_target}!")
        return False
    print("Orthophotos symlink verified.")
        
    # Check Orthophotos backup existence and contents
    ortho_backup_path = ortho_abs_path + "_backup"
    backup_file = os.path.join(ortho_backup_path, "dummy_cache_test.txt")
    if not os.path.exists(backup_file):
        print(f"ERROR: Backup file {backup_file} does not exist!")
        return False
    with open(backup_file, "r") as f:
        content = f.read()
    if content != "Important Cached Aerial Imagery!":
        print(f"ERROR: Backup file content mismatch: {content}")
        return False
    print("Orthophotos backup content verified successfully.")
    
    # 2. Test writing a temporary download file
    print("\n--- 2. Testing Write to RAM-mounted Orthophotos ---")
    temp_download_file = os.path.join(ortho_abs_path, "temp_download_test.txt")
    try:
        with open(temp_download_file, "w") as f:
            f.write("Temporary download chunk.")
        with open(temp_download_file, "r") as f:
            content = f.read()
        print(f"Written temp file content: '{content}'")
        if content != "Temporary download chunk.":
            print("ERROR: Temp file content mismatch!")
            return False
        print("RAM-mounted Orthophotos write verification passed.")
    except Exception as e:
        print(f"ERROR: Orthophotos write failed: {e}")
        return False
        
    # 3. Test Unmounting
    print("\n--- 3. Testing Unmount and Restore ---")
    unmounted = O4_RAMDisk_Utils.unmount_ram_disk(use_orthophotos=True)
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
        
    if os.path.islink(ortho_abs_path):
        print(f"ERROR: {ortho_abs_path} is still a symlink after unmount!")
        return False
        
    # Verify backup is restored
    if os.path.exists(ortho_backup_path):
        print(f"ERROR: Backup directory {ortho_backup_path} still exists after unmount!")
        return False
        
    restored_cache_file = os.path.join(ortho_abs_path, "dummy_cache_test.txt")
    if not os.path.exists(restored_cache_file):
        print(f"ERROR: Restored cache file {restored_cache_file} does not exist!")
        return False
    with open(restored_cache_file, "r") as f:
        content = f.read()
    if content != "Important Cached Aerial Imagery!":
        print(f"ERROR: Restored cache file content mismatch: {content}")
        return False
    print("Orthophotos restored content verified successfully.")
    
    # Verify RAM-written temp file has been successfully merged back to SSD!
    restored_temp_file = os.path.join(ortho_abs_path, "temp_download_test.txt")
    if not os.path.exists(restored_temp_file):
        print("ERROR: Temporary download file was NOT merged back to SSD after restore!")
        return False
    with open(restored_temp_file, "r") as f:
        content = f.read()
    if content != "Temporary download chunk.":
        print(f"ERROR: Merged temp file content mismatch: {content}")
        return False
    print("Temporary RAM-written file successfully merged to SSD cache verified.")
    
    # Clean up the dummy cache and merged files to keep workspace clean
    os.remove(restored_cache_file)
    os.remove(restored_temp_file)

    # 4. Test Recovery/Self-Healing from Abrupt Termination
    print("\n--- 4. Testing Self-Healing Recovery (Simulation of Abrupt Termination) ---")
    # Simulate a crash: mount the RAM disk, write to it, but DO NOT call unmount_ram_disk
    mounted = O4_RAMDisk_Utils.mount_ram_disk(size_gb=2, use_orthophotos=True)
    if not mounted:
        print("ERROR: Recovery mount failed!")
        return False
        
    # Write a new dummy image inside the RAM-linked Orthophotos to ensure it gets recovered
    temp_recover_file = os.path.join(ortho_abs_path, "recover_me.txt")
    with open(temp_recover_file, "w") as f:
        f.write("Crash recovery data")
        
    # Verify we are in the symlink state
    if not os.path.islink(tmp_abs_path) or not os.path.islink(ortho_abs_path):
        print("ERROR: Setup did not create links for recovery simulation!")
        return False
        
    print("Simulated crash state: Links exist and RAM Disk is active.")
    
    # Trigger our recovery logic!
    print("Executing recover_orphaned_symlinks()...")
    recovered = O4_RAMDisk_Utils.recover_orphaned_symlinks()
    if not recovered:
        print("ERROR: recover_orphaned_symlinks() reported no recovery done or failed!")
        return False
        
    # Check that directories are fully restored to regular SSD folders
    if os.path.islink(tmp_abs_path) or os.path.islink(ortho_abs_path):
        print("ERROR: Symbolic links were not removed during recovery!")
        return False
        
    ortho_backup_path = ortho_abs_path + "_backup"
    if os.path.exists(ortho_backup_path):
        print("ERROR: Orthophotos backup folder still remains after recovery!")
        return False
        
    # Verify the RAM-written recovery file was safely merged to the SSD Orthophotos folder
    ssd_recovered_file = os.path.join(ortho_abs_path, "recover_me.txt")
    if not os.path.exists(ssd_recovered_file):
        print("ERROR: RAM-written recovery file was lost during self-healing!")
        return False
        
    with open(ssd_recovered_file, "r") as f:
        content = f.read()
    if content != "Crash recovery data":
        print(f"ERROR: Recovered file content mismatch: {content}")
        return False
        
    # Clean up recovery test files
    os.remove(ssd_recovered_file)
    print("Recovery / Self-Healing verified successfully.")
    
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
