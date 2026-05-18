import os
import sys
import time
import subprocess

def is_dark_mode():
    if sys.platform != "darwin": return False
    try:
        result = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=1)
        return "Dark" in result.stdout
    except:
        return False

is_dark = is_dark_mode()
BG_COLOR = "#2c2c2c" if is_dark else "light green"
FG_COLOR = "#e0e0e0" if is_dark else "black"
ENTRY_BG = "#3d3d3d" if is_dark else "white"
ENTRY_FG = "#4ea8de" if is_dark else "blue"
ACCENT_BG = "#1a1a1a" if is_dark else "dark green"
BTN_BG = "#3a3a3a" if is_dark else "light green"


Ortho4XP_dir = ".." if getattr(sys, "frozen", False) else "."
verbosity = 1
red_flag = False
is_working = False
cleaning_level = 1
gui = None
log = True
write_build_log = False
build_log_buffer = []
is_building_all = False

# System resource limits adjustment
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # macOS default is often 256, which is too low for Ortho4XP's parallel processing
    if soft < 4096:
        new_soft = min(hard, 65536)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
except:
    pass

################################################################################
def progress_bar(nbr, percentage, message=None):
    if gui:
        gui.pgrb_queue.put((nbr, percentage))
    else:
        # Command line progress bar
        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        prefix = "Progress"
        if nbr == 2: prefix = "Downloads"
        elif nbr == 3: prefix = "DDS Conv "
        sys.stdout.write(f"\r{prefix}: [{bar}] {percentage:3d}%")
        sys.stdout.flush()
        if percentage >= 100:
            sys.stdout.write('\n')


################################################################################
def vprint(min_verbosity, *args):
    msg = " ".join([str(x) for x in args])
    if verbosity >= min_verbosity:
        print(msg)
        if write_build_log:
            build_log_buffer.append(msg)


################################################################################
def logprint(*args):
    try:
        f = open(os.path.join(Ortho4XP_dir, "Ortho4XP.log"), "a")
        f.write(
            time.strftime("%c")
            + " | "
            + " ".join([str(x) for x in args])
            + "\n"
        )
        f.close()
    except:
        pass


################################################################################
def lvprint(min_verbosity, *args):
    msg = " ".join([str(x) for x in args])
    if verbosity >= min_verbosity:
        print(msg)
        if log:
            logprint(msg)
        if write_build_log:
            build_log_buffer.append(msg)


def get_config_summary():
    try:
        import O4_Config_Utils as CFG
        import O4_OSM_Utils as OSM
        import O4_Imagery_Utils as IMG
        import O4_Tile_Utils as TILE
        import O4_Overlay_Utils as OVL
        
        summary = [
            "==================================================",
            "          Ortho4XP Build Configuration            ",
            "=================================================="
        ]
        
        sorted_vars = sorted(CFG.cfg_vars.keys())
        for var in sorted_vars:
            info = CFG.cfg_vars[var]
            module_name = info.get("module")
            val = None
            if module_name == "UI":
                val = getattr(sys.modules.get("O4_UI_Utils"), var, None)
            elif module_name == "OSM":
                val = getattr(OSM, var, None)
            elif module_name == "IMG":
                val = getattr(IMG, var, None)
            elif module_name == "TILE":
                val = getattr(TILE, var, None)
            elif module_name == "OVL":
                val = getattr(OVL, var, None)
            else:
                val = getattr(CFG, var, None)
            
            if val is None:
                val = info.get("default")
            summary.append(f"  {var:<30} : {val}")
            
        summary.append("==================================================\n")
        return "\n".join(summary)
    except Exception as e:
        return f"Failed to dump config parameters: {e}\n"


################################################################################
def initialize_build_log(build_dir):
    global build_log_buffer
    build_log_buffer = []
    if not write_build_log:
        return
    build_log_buffer.append(get_config_summary())
    try:
        os.makedirs(build_dir, exist_ok=True)
        log_path = os.path.join(build_dir, "Ortho4XP_build.log")
        if os.path.exists(log_path):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n{'='*20} New Step Started: {time.strftime('%c')} {'='*20}\n\n")
    except Exception as e:
        logprint("Failed to initialize build log:", e)


################################################################################
def flush_build_log(build_dir):
    global build_log_buffer
    if not write_build_log or not build_log_buffer:
        build_log_buffer = []
        return
    try:
        os.makedirs(build_dir, exist_ok=True)
        log_path = os.path.join(build_dir, "Ortho4XP_build.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(build_log_buffer) + "\n")
    except Exception as e:
        logprint("Failed to flush build log to tile:", e)
    finally:
        build_log_buffer = []


################################################################################
def bug_report(*args):
    logprint(
        "An internal error occured. Please file a bug with lat/lon and cfg"
    )
    if args:
        logprint(*args)


################################################################################
def exit_message_and_bottom_line(*args):
    global is_working
    if not args:
        args = ("Process interrupted",)
    if args[0]:
        logprint(*args)
        print(*args)
    print(
        "_____________________________________________________________"
        + "____________________________________"
    )
    is_working = False


################################################################################
def timings_and_bottom_line(tinit):
    global is_working
    print("\nCompleted in " + nicer_timer(time.time() - tinit) + ".")
    print(
        "_____________________________________________________________"
        + "____________________________________"
    )
    is_working = False


################################################################################
def human_print(num, suffix=""):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "{:.1f}{}{}".format(num, unit, suffix)
        num /= 1024.0
    return "{:.1f}{}{}".format(num, "Y", suffix)


################################################################################
def nicer_timer(elapsed):
    out_string = ""
    hours = elapsed // 3600
    if hours:
        elapsed -= 3600 * hours
        out_string += str(int(hours)) + "h"
    minutes = elapsed // 60
    if hours or minutes:
        elapsed -= 60 * minutes
        out_string += str(int(minutes)) + "m"
    elapsed = "{:.2f}".format(elapsed) if not out_string else int(elapsed)
    out_string += str(elapsed) + "sec"
    return out_string
