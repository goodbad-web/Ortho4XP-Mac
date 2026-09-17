import os
import sys
import time
import subprocess
import locale

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
active_cancel_event = None
last_operation_cancelled = False
cancel_reason = None
active_effective_config = {}
active_write_build_log = None
cleaning_level = 1
gui = None
log = True
write_build_log = False
build_log_buffer = []
is_building_all = False


def begin_operation(cancel_event=None, effective_config=None):
    """Start an operation with a fresh, operation-local cancellation state."""
    global active_cancel_event, last_operation_cancelled, cancel_reason
    global active_effective_config, active_write_build_log, red_flag
    pending_cancel = bool(red_flag and cancel_reason)
    pending_reason = cancel_reason if pending_cancel else None
    active_cancel_event = cancel_event
    last_operation_cancelled = False
    # A GUI Stop can arrive after the token is reserved but before the worker
    # thread enters _start_full_pipeline.  Preserve that reason while
    # re-binding the already-cancelled token to the operation context.
    cancel_reason = pending_reason
    active_effective_config = dict(effective_config or {})
    active_write_build_log = active_effective_config.get("write_build_log")
    red_flag = False
    if pending_cancel and active_cancel_event is not None:
        try:
            active_cancel_event.set()
        except AttributeError:
            pass


def cancel_operation(reason="user"):
    """Request cancellation without relying only on the legacy global flag."""
    global red_flag, cancel_reason
    red_flag = True
    cancel_reason = reason
    event = active_cancel_event
    if event is not None:
        try:
            event.set()
        except AttributeError:
            pass


def update_operation_config(values):
    """Update the effective snapshot for a per-tile batch item."""
    global active_effective_config, active_write_build_log
    active_effective_config.update(dict(values or {}))
    if "write_build_log" in active_effective_config:
        active_write_build_log = active_effective_config["write_build_log"]


def is_cancel_requested():
    event = active_cancel_event
    if event is not None:
        try:
            # Keep legacy writers observable while an operation-local token
            # is active.  Stage code must not clear the token, but a few
            # older helpers still set red_flag directly on internal failure.
            return bool(event.is_set()) or bool(red_flag)
        except AttributeError:
            pass
    return bool(red_flag)


def end_operation():
    """Record and clear an operation state after its result was determined."""
    global active_cancel_event, last_operation_cancelled, red_flag
    global active_effective_config, active_write_build_log
    event = active_cancel_event
    if event is not None:
        try:
            last_operation_cancelled = bool(event.is_set()) or bool(red_flag)
        except AttributeError:
            last_operation_cancelled = bool(cancel_reason)
    else:
        last_operation_cancelled = bool(red_flag)
    active_cancel_event = None
    active_effective_config = {}
    active_write_build_log = None
    red_flag = False


def ui_text(english, japanese):
    """Return English by default and Japanese for Japanese locales."""
    language = (
        os.environ.get("ORTHO4XP_LANG")
        or os.environ.get("LC_ALL")
        or os.environ.get("LANG")
        or (locale.getlocale()[0] or "")
    )
    return japanese if language.lower().startswith("ja") else english

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
        gui.pgrb_queue.put((nbr, percentage, message))
    else:
        # Command line progress bar
        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        prefix = "Progress"
        if nbr == 2: prefix = "Downloads"
        elif nbr == 3: prefix = "DDS Conv "
        suffix = f" - {message}" if message else ""
        sys.stdout.write(f"\r{prefix}: [{bar}] {percentage:3d}%{suffix}")
        sys.stdout.flush()
        if percentage >= 100:
            sys.stdout.write('\n')


################################################################################
def vprint(min_verbosity, *args):
    msg = " ".join([str(x) for x in args])
    if verbosity >= min_verbosity:
        print(msg)
        if active_write_build_log if active_write_build_log is not None else write_build_log:
            build_log_buffer.append(msg)
        if gui:
            gui.status_queue.put(msg)


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
        if active_write_build_log if active_write_build_log is not None else write_build_log:
            build_log_buffer.append(msg)
        if gui:
            gui.status_queue.put(msg)


def get_config_summary(tile=None):
    try:
        import O4_Config_Utils as CFG
        import O4_OSM_Utils as OSM
        import O4_Imagery_Utils as IMG
        import O4_Tile_Utils as TILE
        import O4_Overlay_Utils as OVL

        tile_vars = set(getattr(CFG, "list_tile_vars", ()))

        def global_value(var, info):
            module_name = info.get("module")
            if module_name == "UI":
                return getattr(sys.modules.get("O4_UI_Utils"), var, None)
            if module_name == "OSM":
                return getattr(OSM, var, None)
            if module_name == "IMG":
                return getattr(IMG, var, None)
            if module_name == "TILE":
                return getattr(TILE, var, None)
            if module_name == "OVL":
                return getattr(OVL, var, None)
            return getattr(CFG, var, None)
        
        summary = [
            "==================================================",
            "          Ortho4XP Build Configuration            ",
            "=================================================="
        ]
        
        sorted_vars = sorted(CFG.cfg_vars.keys())
        for var in sorted_vars:
            info = CFG.cfg_vars[var]
            global_val = global_value(var, info)
            if global_val is None:
                global_val = info.get("default")
            if tile is not None and var in tile_vars and hasattr(tile, var):
                summary.append(
                    f"  {var:<30} : {getattr(tile, var)} "
                    f"[tile effective; global={global_val}]"
                )
            else:
                summary.append(f"  {var:<30} : {global_val} [global]")
            
        summary.append("==================================================\n")
        return "\n".join(summary)
    except Exception as e:
        return f"Failed to dump config parameters: {e}\n"


################################################################################
def initialize_build_log(build_dir, tile=None):
    global build_log_buffer
    build_log_buffer = []
    build_log_enabled = (
        active_write_build_log
        if active_write_build_log is not None
        else write_build_log
    )
    if not build_log_enabled:
        return
    build_log_buffer.append(get_config_summary(tile))
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
    build_log_enabled = (
        active_write_build_log
        if active_write_build_log is not None
        else write_build_log
    )
    if not build_log_enabled or not build_log_buffer:
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
