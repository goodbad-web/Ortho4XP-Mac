#!/usr/bin/env python3
import sys
import os
os.environ["OPENCV_OPENCL_CACHE_ENABLE"] = "0"
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
except:
    pass
Ortho4XP_dir='..' if getattr(sys,'frozen',False) else '.'
sys.path.append(os.path.join(Ortho4XP_dir,'src'))

import O4_File_Names as FNAMES
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_Tile_Utils as TILE
import O4_GUI_Utils as GUI
import O4_UI_Utils as UI
import O4_Config_Utils as CFG  # CFG imported last because it can modify other modules variables


cmd_line="USAGE: Ortho4XP.py lat lon imagery zl (won't read a tile config)\n  OR:  Ortho4XP.py lat lon (with existing tile config file)"

if __name__ == '__main__':
    if not os.path.isdir(FNAMES.Utils_dir):
        print("Missing ",FNAMES.Utils_dir,"directory, check your install. Exiting.")
        sys.exit(1)
    import signal
    def sig_handler(signum, frame):
        print(f"\n[Ortho4XP] Caught termination signal ({signum}). Exiting cleanly...")
        sys.exit(130)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
        
    import O4_RAMDisk_Utils
    # 1. Recover any orphaned symbolic links from a previous crash/abrupt termination
    O4_RAMDisk_Utils.recover_orphaned_symlinks()
    
    use_ram_disk = getattr(CFG.UI, 'use_ram_disk', False)
    use_ram_disk_for_orthophotos = getattr(CFG.UI, 'use_ram_disk_for_orthophotos', False)
    if use_ram_disk:
        use_ram_disk = O4_RAMDisk_Utils.mount_ram_disk(
            size_gb=getattr(CFG.UI, 'ram_disk_size_gb', 4),
            use_orthophotos=use_ram_disk_for_orthophotos
        )
        if not use_ram_disk:
            print("[Ortho4XP] RAM disk setup failed. Continuing without RAM disk.")
        
    try:
        for directory in (FNAMES.Preview_dir, FNAMES.Provider_dir, FNAMES.Extent_dir, FNAMES.Filter_dir, FNAMES.OSM_dir,
                          FNAMES.Mask_dir,FNAMES.Imagery_dir,FNAMES.Elevation_dir,FNAMES.Geotiff_dir,FNAMES.Patch_dir,
                          FNAMES.Tile_dir,FNAMES.Tmp_dir):
            if not os.path.isdir(directory):
                try: 
                    os.makedirs(directory)
                    print("Creating missing directory",directory)
                except: 
                    print("Could not create required directory",directory,". Exit.")
                    sys.exit(1)
        IMG.initialize_extents_dict()
        IMG.initialize_color_filters_dict()
        IMG.initialize_providers_dict()
        IMG.initialize_combined_providers_dict()
        if len(sys.argv)==1: # switch to the graphical interface
            Ortho4XP = GUI.Ortho4XP_GUI()
    
            Ortho4XP.mainloop()	    
            print("Bon vol!")
        else: # sequel is only concerned with command line 
            if len(sys.argv) not in (3, 5):
                print(cmd_line); sys.exit(2)
            try:
                lat=int(sys.argv[1])
                lon=int(sys.argv[2])
            except:
                print(cmd_line); sys.exit(2)
            if lat < -85 or lat > 84 or lon < -180 or lon > 179:
                print("ERROR: latitude must be in [-85,84] and longitude in [-180,179].")
                sys.exit(2)
            if len(sys.argv)==3:
                try:
                    tile=CFG.Tile(lat,lon,'')
                    if not tile.read_from_config():
                        print("ERROR: could not initialize tile config.")
                        sys.exit(1)
                    known_providers = set(IMG.providers_dict).union(IMG.combined_providers_dict)
                    if tile.default_website not in known_providers:
                        print("ERROR: tile config contains an unknown imagery provider.")
                        sys.exit(2)
                    if int(tile.default_zl) < 12 or int(tile.default_zl) > 18:
                        print("ERROR: tile config zoomlevel must be between 12 and 18.")
                        sys.exit(2)
                except Exception as e:
                    print(e)
                    print("ERROR: could not initialize tile config."); sys.exit(1)
            else:
                try:
                    provider_code=sys.argv[3]
                    zoomlevel=int(sys.argv[4])
                    known_providers = set(IMG.providers_dict).union(IMG.combined_providers_dict)
                    if provider_code not in known_providers:
                        raise ValueError("unknown imagery provider")
                    if zoomlevel < 12 or zoomlevel > 18:
                        raise ValueError("zoomlevel must be between 12 and 18")
                    tile=CFG.Tile(lat,lon,'')
                    tile.default_website=provider_code
                    tile.default_zl=zoomlevel
                except Exception as e:
                    print("ERROR:", e)
                    print(cmd_line); sys.exit(2)
            try:
                stages = (
                    ("vector data", VMAP.build_poly_file),
                    ("mesh", MESH.build_mesh),
                    ("water masks", MASK.build_masks),
                    ("imagery/DSF", TILE.build_tile),
                )
                for stage_name, stage in stages:
                    if not stage(tile) or UI.red_flag:
                        print(f"ERROR: {stage_name} stage failed.")
                        sys.exit(1)
                print("Bon vol!")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Crash! Error: {e}")
                sys.exit(1)
    finally:
        if use_ram_disk:
            O4_RAMDisk_Utils.unmount_ram_disk(use_orthophotos=use_ram_disk_for_orthophotos)
 
        
