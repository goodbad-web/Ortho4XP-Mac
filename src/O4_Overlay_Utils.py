import time
import os
import shutil
import sys
import subprocess
import O4_File_Names as FNAMES
import O4_UI_Utils as UI

# the following is meant to be modified directly by users who need it (in the 
# config window, not here!)
ovl_exclude_pol = [0]
ovl_exclude_net = []

# the following is meant to be modified by the CFG module at run time
custom_overlay_src = ""

if "dar" in sys.platform:
    unzip_cmd = "7z "
    dsftool_cmd = os.path.join(FNAMES.Utils_dir, "mac", "DSFTool")
elif "win" in sys.platform:
    unzip_cmd = os.path.join(FNAMES.Utils_dir, "win", "7z.exe ")
    dsftool_cmd = os.path.join(FNAMES.Utils_dir, "win", "DSFTool.exe")
else:
    unzip_cmd = "7z "
    dsftool_cmd = os.path.join(FNAMES.Utils_dir, "lin", "DSFTool")

################################################################################
def build_overlay(lat, lon):
    if UI.is_working:
        return 0
    UI.is_working = 1
    timer = time.time()
    try:
        UI.logprint("Step 4 for tile lat=", lat, ", lon=", lon, ": starting.")
        UI.vprint(
            0,
            "\nStep 4 : Extracting overlay for tile "
            + FNAMES.short_latlon(lat, lon)
            + " : \n--------\n",
        )
        file_to_sniff = FNAMES.resolve_global_scenery_dsf(
            custom_overlay_src, lat, lon
        )
        if file_to_sniff is None:
            scenery_candidates = FNAMES.global_scenery_dsf_candidates(
                custom_overlay_src, lat, lon
            )
            if not scenery_candidates:
                message = (
                    "   ERROR: Global Scenery DSF was not found below "
                    + (custom_overlay_src or "(empty path)")
                    + ". Expected Earth nav data/"
                    + FNAMES.long_latlon(lat, lon)
                    + ".dsf."
                )
            else:
                message = "   ERROR: Multiple Global Scenery DSFs matched:"
            UI.exit_message_and_bottom_line(message, *scenery_candidates)
            return 0
        if not os.path.isfile(file_to_sniff):
            UI.exit_message_and_bottom_line(
                "   ERROR: file ",
                file_to_sniff,
                "absent. Recall that the overlay source directory needs to be set ",
                "in the config window first.",
            )
            return 0
        file_to_sniff_loc = os.path.join(
            FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + ".dsf"
        )
        UI.vprint(1, "-> Making a copy of the original overlay DSF in tmp dir")
        try:
            shutil.copy(file_to_sniff, file_to_sniff_loc)
        except:
            UI.exit_message_and_bottom_line(
                "   ERROR: could not copy it. Disk full, write permissions, erased",
                " tmp dir ?"
            )
            return 0
        with open(file_to_sniff_loc, "rb") as f:
            dsfid = f.read(2).decode("ascii")
        if dsfid == "7z":
            UI.vprint(1, "-> The original DSF is a 7z archive, uncompressing...")
            archive_path = file_to_sniff_loc + ".7z"
            os.replace(file_to_sniff_loc, archive_path)
            try:
                unzip_res = subprocess.run(
                    [
                        unzip_cmd.strip(),
                        "e",
                        "-y",
                        "-o" + FNAMES.Tmp_dir,
                        archive_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except OSError as error:
                UI.exit_message_and_bottom_line(
                    "   ERROR: could not run 7-Zip:", error
                )
                return 0
            if unzip_res.stdout:
                for line in unzip_res.stdout.splitlines():
                    UI.vprint(1, "     " + line)
            if unzip_res.returncode != 0 or not os.path.isfile(file_to_sniff_loc):
                UI.exit_message_and_bottom_line("   ERROR: could not uncompress overlay DSF.")
                return 0
            try:
                os.remove(archive_path)
            except:
                pass
        UI.vprint(1, "-> Converting the copy to text format")
        dsf2text_out = os.path.join(
            FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + "_tmp_dsf.txt"
        )
        try:
            os.remove(dsf2text_out)
        except OSError:
            pass
        dsfconvertcmd = [
            dsftool_cmd.strip(),
            "-dsf2text",
            file_to_sniff_loc,
            dsf2text_out,
        ]
        try:
            dsf2text_res = subprocess.run(
                dsfconvertcmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as error:
            UI.exit_message_and_bottom_line("   ERROR: could not run DSFTool:", error)
            return 0
        if dsf2text_res.stdout:
            for line in dsf2text_res.stdout.splitlines():
                UI.vprint(1, "     " + line)
        if dsf2text_res.returncode != 0 or not os.path.isfile(dsf2text_out):
            UI.exit_message_and_bottom_line("   ERROR: DSFTool crashed.")
            return 0
        UI.vprint(1, "-> Selecting overlays for copy/paste")
        f = open(dsf2text_out, "r")
        g = open(
            os.path.join(
                FNAMES.Tmp_dir,
                FNAMES.short_latlon(lat, lon) + "_tmp_dsf_without_mesh.txt",
            ),
            "w",
        )
        line = f.readline()
        g.write("PROPERTY sim/overlay 1\n")
        pol_type = 0
        pol_dict = {}
        exclude_set_updated = False
        full_ovl_exclude_pol = set(ovl_exclude_pol)
        while line:
            if "PROPERTY" in line:
                g.write(line)
            elif "POLYGON_DEF" in line:
                level = 2 if "facade" not in line else 3
                pol_dict[pol_type] = line.split()[1]
                UI.vprint(level, pol_type, ":", pol_dict[pol_type])
                pol_type += 1
                g.write(line)
            elif "NETWORK_DEF" in line:
                g.write(line)
            elif "OBJECT_DEF" in line:
                g.write(line)
            elif "STRING_DEF" in line:
                g.write(line)
            elif "BEGIN_POLYGON" in line:
                if not exclude_set_updated:
                    tmp = set()
                    for item in full_ovl_exclude_pol:
                        if isinstance(item, int):
                            tmp.add(item)
                        elif isinstance(item, str):
                            if item and item[0] == "!":
                                item = item[1:]
                                tmp = tmp.union(
                                    [k for k in pol_dict if item not in pol_dict[k]]
                                )
                            else:
                                tmp = tmp.union(
                                    [k for k in pol_dict if item in pol_dict[k]]
                                )
                    full_ovl_exclude_pol = tmp
                    exclude_set_updated = True
                pol_type = int(line.split()[1])
                if pol_type not in full_ovl_exclude_pol:
                    while line and ("END_POLYGON" not in line):
                        g.write(line)
                        line = f.readline()
                    g.write(line)
                else:
                    while line and ("END_POLYGON" not in line):
                        line = f.readline()
            elif "BEGIN_OBJECT" in line:
                while line and ("END_OBJECT" not in line):
                    g.write(line)
                    line = f.readline()
                g.write(line)
            elif "BEGIN_STRING" in line:
                while line and ("END_STRING" not in line):
                    g.write(line)
                    line = f.readline()
                g.write(line)
            elif "BEGIN_SEGMENT" in line:
                road_type = int(line.split()[2])
                if (
                    road_type not in ovl_exclude_net
                    and "" not in ovl_exclude_net
                    and "*" not in ovl_exclude_net
                ):
                    while line and ("END_SEGMENT" not in line):
                        g.write(line)
                        line = f.readline()
                    g.write(line)
                else:
                    while line and ("END_SEGMENT" not in line):
                        line = f.readline()
            line = f.readline()
        f.close()
        g.close()
        UI.vprint(1, "-> Converting back the text DSF to binary format")
        dsfconvertcmd = [
            dsftool_cmd.strip(),
            "-text2dsf",
            os.path.join(
                FNAMES.Tmp_dir,
                FNAMES.short_latlon(lat, lon) + "_tmp_dsf_without_mesh.txt",
            ),
            os.path.join(
                FNAMES.Tmp_dir,
                FNAMES.short_latlon(lat, lon) + "_tmp_dsf_without_mesh.dsf",
            ),
        ]
        output_overlay = os.path.join(
            FNAMES.Tmp_dir,
            FNAMES.short_latlon(lat, lon) + "_tmp_dsf_without_mesh.dsf",
        )
        try:
            os.remove(output_overlay)
        except OSError:
            pass
        try:
            text2dsf_res = subprocess.run(
                dsfconvertcmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as error:
            UI.exit_message_and_bottom_line("   ERROR: could not run DSFTool:", error)
            return 0
        if text2dsf_res.stdout:
            for line in text2dsf_res.stdout.splitlines():
                UI.vprint(1, "     " + line)
        if text2dsf_res.returncode != 0 or not os.path.isfile(output_overlay):
            UI.exit_message_and_bottom_line("   ERROR: DSFTool crashed.")
            return 0
        dest_dir = os.path.join(
            FNAMES.Overlay_dir, "Earth nav data", FNAMES.round_latlon(lat, lon)
        )
        UI.vprint(1, "-> Coping the final overlay DSF in " + dest_dir)
        if not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir)
            except:
                UI.exit_message_and_bottom_line(
                    "   ERROR: could not create destination directory "
                    + str(dest_dir)
                )
                return 0
        try:
            shutil.copy(
                output_overlay,
                os.path.join(dest_dir, FNAMES.short_latlon(lat, lon) + ".dsf"),
            )
        except OSError as error:
            UI.exit_message_and_bottom_line(
                "   ERROR: could not copy final overlay DSF:", error
            )
            return 0
        UI.timings_and_bottom_line(timer)
        return 1
    finally:
        tmp_base = FNAMES.short_latlon(lat, lon)
        tmp_paths = [
            os.path.join(FNAMES.Tmp_dir, tmp_base + ".dsf"),
            os.path.join(FNAMES.Tmp_dir, tmp_base + ".dsf.7z"),
            os.path.join(FNAMES.Tmp_dir, tmp_base + "_tmp_dsf.txt"),
            os.path.join(
                FNAMES.Tmp_dir, tmp_base + "_tmp_dsf_without_mesh.txt"
            ),
            os.path.join(
                FNAMES.Tmp_dir, tmp_base + "_tmp_dsf_without_mesh.dsf"
            ),
            os.path.join(
                FNAMES.Tmp_dir, tmp_base + "_tmp_dsf.txt.elevation.raw"
            ),
            os.path.join(
                FNAMES.Tmp_dir, tmp_base + "_tmp_dsf.txt.sea_level.raw"
            ),
        ]
        for path in tmp_paths:
            try:
                os.remove(path)
            except:
                pass
        UI.is_working = 0
