import sys
import os
Ortho4XP_dir = '.'
sys.path.append(os.path.join(Ortho4XP_dir, 'src'))

import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_Config_Utils as CFG

print("--- Testing new config parameter ---")
# Create tile instance and check variable existence
tile = CFG.Tile(45, -122, '')
tile.make_dirs()
print("build_overlays_in_all_in_one default:", getattr(tile, 'build_overlays_in_all_in_one', None))

# Set to True and write
tile.build_overlays_in_all_in_one = True
tile.write_to_config()
print("Saved config to file.")

# Create another instance and read
tile2 = CFG.Tile(45, -122, '')
tile2.read_from_config()
print("build_overlays_in_all_in_one after read:", getattr(tile2, 'build_overlays_in_all_in_one', None))

# Cleanup created config files
config_file = os.path.join(tile.build_dir, "Ortho4XP_" + FNAMES.short_latlon(45, -122) + ".cfg")
if os.path.exists(config_file):
    os.remove(config_file)
if os.path.exists(config_file + ".bak"):
    os.remove(config_file + ".bak")
print("Cleaned up config files.")
