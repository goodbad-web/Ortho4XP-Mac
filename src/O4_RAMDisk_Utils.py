import os
import sys
import subprocess
import shutil
import O4_File_Names as FNAMES
import O4_UI_Utils as UI

def mount_ram_disk(size_gb=4, use_orthophotos=False):
    if sys.platform != 'darwin':
        UI.vprint(1, "[RAMDisk] RAM disk is only supported on macOS.")
        return False
        
    ram_disk_path = "/Volumes/Ortho4XP_RAM_Disk"
    tmp_path = os.path.abspath(FNAMES.Tmp_dir)
    ortho_path = os.path.abspath(FNAMES.Imagery_dir)
    
    # 1. Check if RAM disk is already mounted
    if os.path.exists(ram_disk_path):
        UI.vprint(1, f"[RAMDisk] RAM disk is already mounted at {ram_disk_path}")
    else:
        UI.vprint(1, f"[RAMDisk] Creating {size_gb}GB RAM disk on macOS...")
        # 1 GB = 2097152 sectors
        sectors = size_gb * 2097152
        try:
            # Attach RAM disk
            result = subprocess.run(
                ["hdiutil", "attach", "-nomount", f"ram://{sectors}"],
                capture_output=True,
                text=True,
                check=True
            )
            device_node = result.stdout.strip()
            UI.vprint(2, f"[RAMDisk] Attached RAM device: {device_node}")
            
            # Format and mount
            subprocess.run(
                ["diskutil", "erasevolume", "HFS+", "Ortho4XP_RAM_Disk", device_node],
                capture_output=True,
                text=True,
                check=True
            )
            UI.vprint(1, f"[RAMDisk] Formatted and mounted RAM disk at {ram_disk_path}")
        except Exception as e:
            UI.vprint(0, f"[RAMDisk] Error creating RAM disk: {e}")
            return False
            
    # 2. Setup the symlink for tmp folder
    try:
        if os.path.islink(tmp_path):
            current_target = os.readlink(tmp_path)
            if current_target == ram_disk_path:
                UI.vprint(2, f"[RAMDisk] symlink already points to {ram_disk_path}")
            else:
                os.unlink(tmp_path)
                os.symlink(ram_disk_path, tmp_path)
        elif os.path.isdir(tmp_path):
            if not os.listdir(tmp_path):
                os.rmdir(tmp_path)
            else:
                backup_path = tmp_path + "_backup"
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path, ignore_errors=True)
                os.rename(tmp_path, backup_path)
                UI.vprint(1, f"[RAMDisk] Non-empty 'tmp' folder backed up to '{backup_path}'.")
            os.symlink(ram_disk_path, tmp_path)
        elif os.path.exists(tmp_path):
            os.remove(tmp_path)
            os.symlink(ram_disk_path, tmp_path)
        else:
            os.symlink(ram_disk_path, tmp_path)
            
        UI.vprint(1, f"[RAMDisk] Linked {tmp_path} -> {ram_disk_path}")
    except Exception as e:
        UI.vprint(0, f"[RAMDisk] Error setting up symbolic link: {e}")
        return False

    # 3. Setup the symlink for Orthophotos folder if requested
    if use_orthophotos:
        try:
            ram_ortho_path = os.path.join(ram_disk_path, "Orthophotos")
            os.makedirs(ram_ortho_path, exist_ok=True)
            
            if os.path.islink(ortho_path):
                current_target = os.readlink(ortho_path)
                if current_target == ram_ortho_path:
                    UI.vprint(2, f"[RAMDisk] Orthophotos symlink already points to {ram_ortho_path}")
                else:
                    os.unlink(ortho_path)
                    os.symlink(ram_ortho_path, ortho_path)
                    UI.vprint(1, f"[RAMDisk] Updated Orthophotos link -> {ram_ortho_path}")
            elif os.path.isdir(ortho_path):
                if not os.listdir(ortho_path):
                    os.rmdir(ortho_path)
                else:
                    ortho_backup = ortho_path + "_backup"
                    if os.path.exists(ortho_backup):
                        shutil.rmtree(ortho_backup, ignore_errors=True)
                    os.rename(ortho_path, ortho_backup)
                    UI.vprint(1, f"[RAMDisk] Non-empty 'Orthophotos' folder backed up to '{ortho_backup}'.")
                os.symlink(ram_ortho_path, ortho_path)
                UI.vprint(1, f"[RAMDisk] Linked Orthophotos -> {ram_ortho_path}")
            else:
                if os.path.exists(ortho_path):
                    os.remove(ortho_path)
                os.symlink(ram_ortho_path, ortho_path)
                UI.vprint(1, f"[RAMDisk] Linked Orthophotos -> {ram_ortho_path}")
        except Exception as e:
            UI.vprint(0, f"[RAMDisk] Error setting up Orthophotos symlink: {e}")
            return False

    return True

def merge_directories(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dest_dir, rel_path) if rel_path != "." else dest_dir
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(target_dir, file)
            try:
                shutil.copy2(src_file, dest_file)
            except Exception as e:
                UI.vprint(2, f"[RAMDisk] Warning: Failed to copy {file} during merge: {e}")

def unmount_ram_disk(use_orthophotos=False):
    if sys.platform != 'darwin':
        return False
        
    ram_disk_path = "/Volumes/Ortho4XP_RAM_Disk"
    tmp_path = os.path.abspath(FNAMES.Tmp_dir)
    ortho_path = os.path.abspath(FNAMES.Imagery_dir)
    ortho_backup = ortho_path + "_backup"
    tmp_backup = tmp_path + "_backup"
    
    # 1. Clean up Orthophotos symlink and restore backup if requested
    if use_orthophotos:
        try:
            if os.path.islink(ortho_path):
                ram_ortho_path = os.path.realpath(ortho_path)
                if os.path.exists(ram_ortho_path) and os.path.exists(ortho_backup):
                    UI.vprint(1, "[RAMDisk] Merging newly downloaded Orthophotos back to SSD cache...")
                    merge_directories(ram_ortho_path, ortho_backup)
                
                os.unlink(ortho_path)
                if os.path.exists(ortho_backup):
                    os.rename(ortho_backup, ortho_path)
                    UI.vprint(1, f"[RAMDisk] Restored and merged 'Orthophotos' from '{ortho_backup}'.")
                else:
                    os.makedirs(ortho_path, exist_ok=True)
                    UI.vprint(1, "[RAMDisk] Restored empty 'Orthophotos' directory.")
            elif os.path.isdir(ortho_path):
                pass
        except Exception as e:
            UI.vprint(0, f"[RAMDisk] Error restoring Orthophotos directory: {e}")
            
    # 2. Remove symlink and restore empty or backed up tmp directory
    try:
        if os.path.islink(tmp_path):
            os.unlink(tmp_path)
            if os.path.exists(tmp_backup):
                os.rename(tmp_backup, tmp_path)
                UI.vprint(1, f"[RAMDisk] Restored 'tmp' directory from '{tmp_backup}'.")
            else:
                os.makedirs(tmp_path, exist_ok=True)
                UI.vprint(1, "[RAMDisk] Restored empty 'tmp' directory.")
        elif os.path.isdir(tmp_path):
            pass # normal directory, no symlink to clean
    except Exception as e:
        UI.vprint(0, f"[RAMDisk] Error removing symlink: {e}")
    except Exception as e:
        UI.vprint(0, f"[RAMDisk] Error removing symlink: {e}")
        
    # 3. Unmount RAM disk
    if os.path.exists(ram_disk_path):
        UI.vprint(1, f"[RAMDisk] Detaching RAM disk from {ram_disk_path}...")
        try:
            subprocess.run(
                ["hdiutil", "detach", ram_disk_path],
                capture_output=True,
                text=True,
                check=True
            )
            UI.vprint(1, "[RAMDisk] RAM disk detached successfully.")
            return True
        except Exception as e:
            UI.vprint(0, f"[RAMDisk] Error detaching RAM disk: {e}")
            return False
    return True
