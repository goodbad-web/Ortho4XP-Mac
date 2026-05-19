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
    if use_orthophotos or os.path.islink(ortho_path) or os.path.exists(ortho_backup):
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
        
    # 3. Unmount RAM disk
    if os.path.exists(ram_disk_path):
        UI.vprint(1, f"[RAMDisk] Detaching RAM disk from {ram_disk_path}...")
        try:
            subprocess.run(
                ["hdiutil", "detach", "-force", ram_disk_path],
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

def recover_orphaned_symlinks():
    """
    Check if we have orphaned symlinks left over from a previous crash/termination.
    If so, restore SSD backups and force detach any orphaned mounted RAM Disk.
    This guarantees a clean, SSD-backed workspace before any new execution.
    """
    if sys.platform != 'darwin':
        return False
        
    ram_disk_path = "/Volumes/Ortho4XP_RAM_Disk"
    tmp_path = os.path.abspath(FNAMES.Tmp_dir)
    ortho_path = os.path.abspath(FNAMES.Imagery_dir)
    ortho_backup = ortho_path + "_backup"
    tmp_backup = tmp_path + "_backup"
    recovered = False
    
    # 1. Recover Orthophotos if symlink exists OR backup exists
    if os.path.islink(ortho_path) or os.path.exists(ortho_backup):
        UI.vprint(1, "[RAMDisk] Orphaned Orthophotos symlink or backup detected! Restoring SSD cache...")
        try:
            if os.path.islink(ortho_path):
                ram_ortho_path = os.path.realpath(ortho_path)
                # If the symlink's target directory exists (e.g. RAM Disk is still mounted) and SSD backup exists, merge
                if os.path.exists(ram_ortho_path) and os.path.exists(ortho_backup):
                    merge_directories(ram_ortho_path, ortho_backup)
                os.unlink(ortho_path)
            
            if os.path.exists(ortho_backup):
                if os.path.exists(ortho_path):
                    if os.path.isdir(ortho_path):
                        # Merge if a regular folder got created somehow
                        merge_directories(ortho_backup, ortho_path)
                        shutil.rmtree(ortho_backup, ignore_errors=True)
                    else:
                        os.remove(ortho_path)
                        os.rename(ortho_backup, ortho_path)
                else:
                    os.rename(ortho_backup, ortho_path)
                UI.vprint(1, f"[RAMDisk] Successfully restored 'Orthophotos' from backup.")
            else:
                os.makedirs(ortho_path, exist_ok=True)
            recovered = True
        except Exception as e:
            UI.vprint(0, f"[RAMDisk] Error recovering Orthophotos symlink: {e}")

    # 2. Recover tmp if symlink exists OR backup exists
    if os.path.islink(tmp_path) or os.path.exists(tmp_backup):
        UI.vprint(1, "[RAMDisk] Orphaned tmp symlink or backup detected! Restoring SSD cache...")
        try:
            if os.path.islink(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(tmp_backup):
                if os.path.exists(tmp_path):
                    if os.path.isdir(tmp_path):
                        merge_directories(tmp_backup, tmp_path)
                        shutil.rmtree(tmp_backup, ignore_errors=True)
                    else:
                        os.remove(tmp_path)
                        os.rename(tmp_backup, tmp_path)
                else:
                    os.rename(tmp_backup, tmp_path)
                UI.vprint(1, f"[RAMDisk] Successfully restored 'tmp' from backup.")
            else:
                os.makedirs(tmp_path, exist_ok=True)
            recovered = True
        except Exception as e:
            UI.vprint(0, f"[RAMDisk] Error recovering tmp symlink: {e}")

    # 3. Clean up the mounted RAM disk to be perfectly sure
    if os.path.exists(ram_disk_path):
        UI.vprint(1, f"[RAMDisk] Orphaned RAM Disk mount detected. Detaching...")
        try:
            subprocess.run(
                ["hdiutil", "detach", "-force", ram_disk_path],
                capture_output=True,
                text=True,
                check=True
            )
            UI.vprint(1, "[RAMDisk] Orphaned RAM Disk detached successfully.")
            recovered = True
        except Exception as e:
            # It's fine if detaching fails here as long as SSD paths are recovered
            UI.vprint(2, f"[RAMDisk] Warning: Failed to force detach RAM Disk (might be unmounted): {e}")

    return recovered

def check_and_restore_cached_image(file_path):
    """
    If the requested file_path (which points inside Orthophotos) doesn't exist on RAM disk,
    but does exist in Orthophotos_backup (the SSD cache), copy it over on-demand.
    Returns True if the file exists on RAM disk (either originally or after restoring).
    """
    if not file_path:
        return False
        
    if os.path.exists(file_path):
        return True
        
    import O4_File_Names as FNAMES
    ortho_dir = os.path.abspath(FNAMES.Imagery_dir)
    
    # RAM Disk is active only when Orthophotos is a symlink pointing to the RAM Disk
    if os.path.islink(ortho_dir):
        backup_dir = ortho_dir + "_backup"
        abs_file_path = os.path.abspath(file_path)
        
        # Ensure the file_path is indeed inside the Orthophotos directory
        if abs_file_path.startswith(ortho_dir + os.sep) or abs_file_path == ortho_dir:
            rel_path = os.path.relpath(abs_file_path, ortho_dir)
            backup_file_path = os.path.join(backup_dir, rel_path)
            
            if os.path.exists(backup_file_path):
                # Double-check file existence to avoid redundant copy steps
                if os.path.exists(abs_file_path):
                    return True
                    
                try:
                    os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
                    
                    # Atomic write utilizing temporary file to eliminate race conditions
                    tmp_file_path = abs_file_path + f".{os.getpid()}.tmp"
                    shutil.copy2(backup_file_path, tmp_file_path)
                    try:
                        os.replace(tmp_file_path, abs_file_path)
                    except Exception as replace_err:
                        # If replacing failed because another process already created the file, clean up and ignore
                        if os.path.exists(abs_file_path):
                            try:
                                os.remove(tmp_file_path)
                            except:
                                pass
                        else:
                            raise replace_err
                            
                    UI.vprint(2, f"[RAMDisk] Restored cached image on-demand: {os.path.basename(file_path)}")
                    return True
                except Exception as e:
                    UI.vprint(2, f"[RAMDisk] Warning: Failed to restore cached image {file_path}: {e}")
                    
    return False


def flush_tile_imagery(lat, lon):
    """
    Copy all cached imagery for the completed tile (lat, lon) from RAM Disk back to SSD,
    and remove them from RAM Disk to free up space during batch builds.
    Handles both normal and grouped folder structures.
    """
    import O4_File_Names as FNAMES
    ortho_dir = os.path.abspath(FNAMES.Imagery_dir)
    
    if not os.path.islink(ortho_dir):
        return False
        
    backup_dir = ortho_dir + "_backup"
    flushed = False
    
    # 1. Handle normal structure: Orthophotos/short_latlon/
    normal_rel = FNAMES.short_latlon(lat, lon)
    ram_normal_dir = os.path.join(ortho_dir, normal_rel)
    ssd_normal_dir = os.path.join(backup_dir, normal_rel)
    if os.path.exists(ram_normal_dir):
        UI.vprint(1, f"[RAMDisk] Flushing tile {normal_rel} imagery cache to SSD to free space...")
        merge_directories(ram_normal_dir, ssd_normal_dir)
        try:
            shutil.rmtree(ram_normal_dir, ignore_errors=True)
            UI.vprint(1, f"[RAMDisk] Successfully freed RAM Disk space for normal tile: {normal_rel}.")
            flushed = True
        except Exception as e:
            UI.vprint(2, f"[RAMDisk] Warning: Failed to clear normal RAM Disk tile directory: {e}")

    # 2. Handle grouped structure: Orthophotos/long_latlon/
    grouped_rel = FNAMES.long_latlon(lat, lon)
    ram_grouped_dir = os.path.join(ortho_dir, grouped_rel)
    ssd_grouped_dir = os.path.join(backup_dir, grouped_rel)
    if os.path.exists(ram_grouped_dir):
        UI.vprint(1, f"[RAMDisk] Flushing tile {grouped_rel} (grouped) imagery cache to SSD to free space...")
        merge_directories(ram_grouped_dir, ssd_grouped_dir)
        try:
            shutil.rmtree(ram_grouped_dir, ignore_errors=True)
            UI.vprint(1, f"[RAMDisk] Successfully freed RAM Disk space for grouped tile: {grouped_rel}.")
            flushed = True
        except Exception as e:
            UI.vprint(2, f"[RAMDisk] Warning: Failed to clear grouped RAM Disk tile directory: {e}")

    return flushed


