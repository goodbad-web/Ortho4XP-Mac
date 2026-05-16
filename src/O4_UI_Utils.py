import os
import sys
import time

Ortho4XP_dir = ".." if getattr(sys, "frozen", False) else "."
verbosity = 1
red_flag = False
is_working = False
cleaning_level = 1
gui = None
log = True

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
    if verbosity >= min_verbosity:
        print(*args)


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
    if verbosity >= min_verbosity:
        print(*args)
    if log:
        logprint(*args)


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
