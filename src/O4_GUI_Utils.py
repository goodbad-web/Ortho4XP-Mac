import os
import sys
import shutil
import locale
from math import floor, cos, pi
import queue
import threading
import tkinter as tk
from tkinter import (
    RIDGE,
    N,
    S,
    E,
    W,
    NW,
    ALL,
    END,
    LEFT,
    RIGHT,
    CENTER,
    HORIZONTAL,
    filedialog,
)
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import O4_Version
import O4_Imagery_Utils as IMG
import O4_File_Names as FNAMES
import O4_Geo_Utils as GEO
import O4_Vector_Utils as VECT
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_Tile_Utils as TILE
import O4_UI_Utils as UI
import O4_Config_Utils as CFG
import O4_Overlay_Utils as OVL

# Set OsX=True if you prefer the OsX way of drawing existing tiles but 
# are on Linux or Windows.
OsX = "dar" in sys.platform


def _ui_text(english, japanese):
    """Use Japanese for Japanese locales while keeping English as default."""
    language = (
        os.environ.get("ORTHO4XP_LANG")
        or os.environ.get("LC_ALL")
        or os.environ.get("LANG")
        or (locale.getlocale()[0] or "")
    )
    return japanese if language.lower().startswith("ja") else english

################################################################################
class Ortho4XP_GUI(tk.Tk):

    # Constants
    zl_list = ["12", "13", "14", "15", "16", "17", "18"]

    def __init__(self):
        tk.Tk.__init__(self)
        O4 = ttk.Style()
        O4.theme_use("alt")
        O4.configure(
            "Flat.TButton",
            background=UI.BTN_BG,
            foreground=UI.FG_COLOR,
            highlightbackground=UI.BTN_BG,
            selectbackground=UI.BTN_BG,
            highlightcolor=UI.BTN_BG,
            highlightthickness=0,
            relief="flat",
        )
        O4.map(
            "Flat.TButton",
            background=[
                ("disabled", "pressed", "!focus", "active", UI.BTN_BG)
            ],
            foreground=[
                ("disabled", "pressed", "!focus", "active", UI.FG_COLOR)
            ],
        )
        O4.configure(
            "O4.TCombobox",
            selectbackground=UI.ENTRY_BG,
            selectforeground=UI.ENTRY_FG,
            fieldbackground=UI.ENTRY_BG,
            foreground=UI.ENTRY_FG,
            background=UI.ENTRY_BG,
        )
        O4.map(
            "O4.TCombobox",
            fieldbackground=[
                ("disabled", "!focus", "focus", "active", UI.ENTRY_BG)
            ],
        )
        O4.configure(
            "TEntry",
            fieldbackground=UI.ENTRY_BG,
            foreground=UI.ENTRY_FG,
            insertcolor=UI.FG_COLOR,
        )
        O4.configure(
            "Compact.TButton",
            padding=(4, 1),
            background=UI.BTN_BG,
            foreground=UI.FG_COLOR,
            highlightbackground=UI.BTN_BG,
            selectbackground=UI.BTN_BG,
            highlightcolor=UI.BTN_BG,
            highlightthickness=0,
            relief="flat",
        )
        self.option_add("*Font", "TkFixedFont")

        # Let UI know ourself
        UI.gui = self
        self.status_queue = queue.Queue()
        self.stage_result_queue = queue.Queue()
        # Initialize providers combobox entries
        self.map_list = sorted(
            [
                provider_code
                for provider_code in set(IMG.providers_dict)
                if IMG.providers_dict[provider_code]["in_GUI"]
            ]
            + sorted(set(IMG.combined_providers_dict))
        )
        try:
            self.map_list.remove("OSM")
        except:
            pass
        try:
            self.map_list.remove("SEA")
        except:
            pass

        # Grid behaviour
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Resources
        self.title("Ortho4XP " + O4_Version.version)
        self.folder_icon = tk.PhotoImage(
            file=os.path.join(FNAMES.Utils_dir, "Folder.gif")
        )
        self.earth_icon = tk.PhotoImage(
            file=os.path.join(FNAMES.Utils_dir, "Earth.gif")
        )
        self.loupe_icon = tk.PhotoImage(
            file=os.path.join(FNAMES.Utils_dir, "Loupe.gif")
        )
        self.config_icon = tk.PhotoImage(
            file=os.path.join(FNAMES.Utils_dir, "Config.gif")
        )
        self.stop_icon = tk.PhotoImage(
            file=os.path.join(FNAMES.Utils_dir, "Stop.gif")
        )
        self.exit_icon = tk.PhotoImage(
            file=os.path.join(FNAMES.Utils_dir, "Exit.gif")
        )

        # Frame instances and placement
        # Level 0
        self.frame_top = tk.Frame(
            self, border=4, relief=RIDGE, bg=UI.BG_COLOR
        )
        self.frame_top.grid(row=0, column=0, sticky=N + S + W + E)
        self.frame_top.columnconfigure(0, weight=1)
        self.frame_console = tk.Frame(
            self, border=4, relief=RIDGE, bg=UI.BG_COLOR
        )
        self.frame_console.grid(row=1, column=0, sticky=N + S + W + E)
        # Level 1
        self.frame_tile = tk.Frame(
            self.frame_top, border=0, padx=5, pady=5, bg=UI.BG_COLOR
        )
        self.frame_tile.grid(row=0, column=0, sticky=N + S + W + E)
        self.frame_steps = tk.Frame(
            self.frame_top, border=0, padx=5, pady=5, bg=UI.BG_COLOR
        )
        self.frame_steps.grid(row=1, column=0, sticky=N + S + W + E)
        self.frame_aux = tk.Frame(
            self.frame_top, border=0, padx=5, pady=5, bg=UI.BG_COLOR
        )
        self.frame_aux.grid(row=2, column=0, sticky=N + S + W + E)
        self.frame_bars = tk.Frame(
            self.frame_top, border=0, padx=5, pady=5, bg=UI.BG_COLOR
        )
        self.frame_bars.grid(row=3, column=0, sticky=N + S + W + E)
        for i in range(3):
            self.frame_bars.columnconfigure(i, weight=1)
        # Level 2
        self.frame_folder = tk.Frame(
            self.frame_tile, border=0, padx=0, pady=0, bg=UI.BG_COLOR
        )
        self.frame_folder.grid(
            row=2, column=0, columnspan=8, sticky=N + S + W + E
        )

        # Widgets instances and placement
        # First row (Tile data)
        tk.Label(
            self.frame_tile,
            text="Input",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=0, column=0, columnspan=8, sticky=W + E)

        self.lat = tk.StringVar()
        self.lat.trace_add("write", self.tile_change)
        tk.Label(self.frame_tile, text="Latitude:", bg=UI.BG_COLOR, fg=UI.FG_COLOR).grid(
            row=1, column=0, padx=5, pady=5, sticky=E + W
        )
        self.lat_entry = tk.Entry(
            self.frame_tile,
            width=4,
            bg=UI.ENTRY_BG,
            fg=UI.ENTRY_FG,
            textvariable=self.lat,
            insertbackground=UI.FG_COLOR,
        )
        self.lat_entry.grid(row=1, column=1, padx=5, pady=5, sticky=W)

        self.lon = tk.StringVar()
        self.lon.trace_add("write", self.tile_change)
        tk.Label(
            self.frame_tile, anchor=W, text="Longitude:", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=1, column=2, padx=5, pady=5, sticky=E + W)
        self.lon_entry = tk.Entry(
            self.frame_tile,
            width=4,
            bg=UI.ENTRY_BG,
            fg=UI.ENTRY_FG,
            textvariable=self.lon,
            insertbackground=UI.FG_COLOR,
        )
        self.lon_entry.grid(row=1, column=3, padx=5, pady=5, sticky=W)

        self.default_website = tk.StringVar()
        self.default_website.trace_add("write", self.update_cfg)
        tk.Label(
            self.frame_tile, anchor=W, text="Imagery:", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=1, column=4, padx=5, pady=5, sticky=E + W)
        self.img_combo = ttk.Combobox(
            self.frame_tile,
            values=self.map_list,
            textvariable=self.default_website,
            state="readonly",
            width=14,
            style="O4.TCombobox",
        )
        self.img_combo.grid(row=1, column=5, padx=5, pady=5, sticky=W)

        self.default_zl = tk.StringVar()
        self.default_zl.trace_add("write", self.update_cfg)
        tk.Label(
            self.frame_tile, anchor=W, text="Zoomlevel:", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=1, column=6, padx=5, pady=5, sticky=E + W)
        self.zl_combo = ttk.Combobox(
            self.frame_tile,
            values=self.zl_list,
            textvariable=self.default_zl,
            state="readonly",
            width=3,
            style="O4.TCombobox",
        )
        self.zl_combo.grid(row=1, column=7, padx=5, pady=5, sticky=W)

        # Second row (Base Folder)
        self.frame_folder.columnconfigure(1, weight=1)
        tk.Label(
            self.frame_folder, anchor=W, text="Base Folder:", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=0, column=0, padx=5, pady=5, sticky=E + W)
        self.custom_build_dir = tk.StringVar()
        self.custom_build_dir_entry = tk.Entry(
            self.frame_folder,
            bg=UI.ENTRY_BG,
            fg=UI.ENTRY_FG,
            textvariable=self.custom_build_dir,
            insertbackground=UI.FG_COLOR,
        )
        self.custom_build_dir_entry.grid(
            row=0, column=1, padx=0, pady=0, sticky=E + W
        )
        ttk.Button(
            self.frame_folder,
            takefocus=False,
            image=self.folder_icon,
            command=self.choose_custom_build_dir,
            style="Flat.TButton",
        ).grid(row=0, column=2, padx=0, pady=0, sticky=N + S + E + W)

        # Third row (Steps)
        for i in range(5):
            self.frame_steps.columnconfigure(i, weight=1)
        tk.Label(
            self.frame_steps,
            text="Run",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=0, column=0, columnspan=5, sticky=W + E)
        ttk.Button(
            self.frame_steps,
            text="Assemble Vector data",
            command=self.build_poly_file,
        ).grid(row=1, column=0, padx=5, pady=0, sticky=N + S + E + W)
        build_mesh_button = ttk.Button(
            self.frame_steps, text="Triangulate 3D Mesh"
        )  # ,command=self.build_mesh)
        build_mesh_button.grid(
            row=1, column=1, padx=5, pady=0, sticky=N + S + E + W
        )
        build_mesh_button.bind("<ButtonPress-1>", self.build_mesh)
        build_mesh_button.bind("<Shift-ButtonPress-1>", self.sort_mesh)
        mod_key = "<Command-ButtonPress-1>" if OsX else "<Control-ButtonPress-1>"
        build_mesh_button.bind(mod_key, self.community_mesh)
        build_masks_button = ttk.Button(
            self.frame_steps, text=" Draw Water Masks  "
        )  # ,command=self.build_masks)
        build_masks_button.grid(
            row=1, column=2, padx=5, pady=0, sticky=N + S + E + W
        )
        build_masks_button.bind("<ButtonPress-1>", self.build_masks)
        build_masks_button.bind("<Shift-ButtonPress-1>", self.build_masks)
        ttk.Button(
            self.frame_steps,
            text=" Build Imagery/DSF ",
            command=self.build_tile,
        ).grid(row=1, column=3, padx=5, pady=0, sticky=N + S + E + W)
        ttk.Button(
            self.frame_steps, text="    All in one     ", command=self.build_all
        ).grid(row=1, column=4, padx=5, pady=0, sticky=N + S + E + W)

        # Support section
        self.frame_aux.columnconfigure(0, weight=1)
        tk.Label(
            self.frame_aux,
            text="Support",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=0, column=0, sticky=W + E)
        self.frame_aux_main = tk.Frame(
            self.frame_aux, border=0, padx=0, pady=0, bg=UI.BG_COLOR
        )
        self.frame_aux_main.grid(row=1, column=0, sticky=N + S + W + E)
        for i in range(6):
            self.frame_aux_main.columnconfigure(i, weight=1)
        ttk.Button(
            self.frame_aux_main,
            takefocus=False,
            text="Verify",
            command=self.check_dependencies,
            style="Compact.TButton",
        ).grid(row=0, column=0, padx=3, pady=0, sticky=N + S + E + W)
        ttk.Button(
            self.frame_aux_main,
            takefocus=False,
            text="Config",
            command=self.open_config_window,
            style="Compact.TButton",
        ).grid(row=0, column=1, padx=3, pady=0, sticky=N + S + E + W)
        ttk.Button(
            self.frame_aux_main,
            takefocus=False,
            text="Preview",
            command=self.open_earth_window,
            style="Compact.TButton",
        ).grid(row=0, column=2, padx=3, pady=0, sticky=N + S + E + W)
        ttk.Button(
            self.frame_aux_main,
            takefocus=False,
            text="Custom ZL",
            command=self.open_custom_zl_window,
            style="Compact.TButton",
        ).grid(row=0, column=3, padx=3, pady=0, sticky=N + S + E + W)
        ttk.Button(
            self.frame_aux_main,
            takefocus=False,
            text="Stop",
            command=self.set_red_flag,
            style="Compact.TButton",
        ).grid(row=0, column=4, padx=3, pady=0, sticky=N + S + E + W)
        ttk.Button(
            self.frame_aux_main,
            takefocus=False,
            text="Exit",
            command=self.exit_prg,
            style="Compact.TButton",
        ).grid(row=0, column=5, padx=3, pady=0, sticky=N + S + E + W)

        # Fourth row (Progress bars and controls)
        # Label(self.frame_left,anchor=W,text="DSF/Masks progress",
        # bg="light green")
        self.progress_titles = {
            1: "Vector / mesh / masks",
            2: "Downloads",
            3: "DDS conversion",
        }
        self.pgrb1v = tk.IntVar()
        self.pgrb2v = tk.IntVar()
        self.pgrb3v = tk.IntVar()
        self.pgrbv = {1: self.pgrb1v, 2: self.pgrb2v, 3: self.pgrb3v}
        self.pgrb_detail = {
            1: tk.StringVar(value="Idle"),
            2: tk.StringVar(value="Idle"),
            3: tk.StringVar(value="Idle"),
        }
        self.pgrb_pct = {
            1: tk.StringVar(value="0%"),
            2: tk.StringVar(value="0%"),
            3: tk.StringVar(value="0%"),
        }
        for i in range(3):
            tk.Label(
                self.frame_bars,
                text=self.progress_titles[i + 1],
                fg=UI.FG_COLOR,
                bg=UI.BG_COLOR,
            ).grid(row=0, column=i, padx=5, pady=(0, 2), sticky=W + E)
        self.pgrb1 = ttk.Progressbar(
            self.frame_bars,
            mode="determinate",
            orient=HORIZONTAL,
            variable=self.pgrb1v,
        )
        self.pgrb1.grid(row=1, column=0, padx=5, pady=0, sticky=W + E)
        tk.Label(
            self.frame_bars,
            textvariable=self.pgrb_pct[1],
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
        ).grid(row=2, column=0, padx=5, pady=(0, 2), sticky=E)
        tk.Label(
            self.frame_bars,
            textvariable=self.pgrb_detail[1],
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
            anchor=W,
        ).grid(row=3, column=0, padx=5, pady=(0, 2), sticky=W + E)
        self.pgrb2 = ttk.Progressbar(
            self.frame_bars,
            mode="determinate",
            orient=HORIZONTAL,
            variable=self.pgrb2v,
        )
        self.pgrb2.grid(row=1, column=1, padx=5, pady=0, sticky=W + E)
        tk.Label(
            self.frame_bars,
            textvariable=self.pgrb_pct[2],
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
        ).grid(row=2, column=1, padx=5, pady=(0, 2), sticky=E)
        tk.Label(
            self.frame_bars,
            textvariable=self.pgrb_detail[2],
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
            anchor=W,
        ).grid(row=3, column=1, padx=5, pady=(0, 2), sticky=W + E)
        self.pgrb3 = ttk.Progressbar(
            self.frame_bars,
            mode="determinate",
            orient=HORIZONTAL,
            variable=self.pgrb3v,
        )
        self.pgrb3.grid(row=1, column=2, padx=5, pady=0, sticky=W + E)
        tk.Label(
            self.frame_bars,
            textvariable=self.pgrb_pct[3],
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
        ).grid(row=2, column=2, padx=5, pady=(0, 2), sticky=E)
        tk.Label(
            self.frame_bars,
            textvariable=self.pgrb_detail[3],
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
            anchor=W,
        ).grid(row=3, column=2, padx=5, pady=(0, 2), sticky=W + E)
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(
            self.frame_bars,
            textvariable=self.status_var,
            fg=UI.FG_COLOR,
            bg=UI.BG_COLOR,
            anchor=W,
            justify=LEFT,
        ).grid(row=4, column=0, columnspan=3, padx=5, pady=(4, 0), sticky=W + E)

        # Console
        self.frame_console.rowconfigure(0, weight=0)
        self.frame_console.rowconfigure(1, weight=1)
        self.frame_console.columnconfigure(0, weight=1)

        self.btn_clear = ttk.Button(
            self.frame_console,
            text="Clear",
            command=self.clear_console,
            style="Flat.TButton",
            takefocus=False,
        )
        self.btn_clear.grid(row=0, column=0, padx=5, pady=2, sticky=E)

        self.console = tk.Text(self.frame_console, bd=0, bg=UI.ENTRY_BG, fg=UI.FG_COLOR, insertbackground=UI.FG_COLOR)
        self.console.grid(row=1, column=0, sticky=N + S + E + W)

        # Update
        self.console_queue = queue.Queue()
        self.console_update()
        self.pgrb_queue = queue.Queue()
        self.pgrb_update()
        self.status_update()

        # Redirection
        self.stdout_orig = sys.stdout
        sys.stdout = self

        # reinitialization from last visit
        try:
            f = open(
                os.path.join(FNAMES.Ortho4XP_dir, ".last_gui_params.txt"), "r"
            )
            (lat, lon, default_website, default_zl) = f.readline().split()
            custom_build_dir = f.readline().strip()
            self.lat.set(lat)
            self.lon.set(lon)
            self.default_website.set(default_website)
            self.default_zl.set(default_zl)
            self.custom_build_dir.set(custom_build_dir)
            f.close()
        except:
            self.lat.set(48)
            self.lon.set(-6)
            self.default_website.set("BI")
            self.default_zl.set(16)
            self.custom_build_dir.set("")

    # GUI methods
    def clear_console(self):
        self.console_queue.put(None)

    def write(self, line):
        self.console_queue.put(line)

    def flush(self):
        return

    def console_update(self):
        try:
            while 1:
                line = self.console_queue.get_nowait()
                at_bottom = self.console.yview()[1] >= 0.99
                if line is None:
                    self.console.delete(1.0, END)
                else:
                    self.console.insert(END, str(line))
                if at_bottom:
                    self.console.see(END)
                self.console.update_idletasks()
        except queue.Empty:
            pass
        self.callback_console = self.after(100, self.console_update)

    def pgrb_update(self):
        try:
            while 1:
                item = self.pgrb_queue.get_nowait()
                if len(item) == 3:
                    (nbr, value, message) = item
                else:
                    (nbr, value) = item
                    message = None
                self.pgrbv[nbr].set(value)
                self.pgrb_pct[nbr].set(f"{value}%")
                if message:
                    self.pgrb_detail[nbr].set(str(message))
        except queue.Empty:
            pass
        self.callback_pgrb = self.after(100, self.pgrb_update)

    def status_update(self):
        try:
            while 1:
                self.status_var.set(self.status_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            while 1:
                label, result, was_cancelled = self.stage_result_queue.get_nowait()
                if was_cancelled:
                    self.status_var.set(_ui_text(f"Cancelled: {label}", f"キャンセル: {label}"))
                elif result:
                    self.status_var.set(_ui_text(f"Completed: {label}", f"完了: {label}"))
                else:
                    self.status_var.set(_ui_text(f"Failed: {label}", f"失敗: {label}"))
        except queue.Empty:
            pass
        self.callback_status = self.after(100, self.status_update)

    def tile_change(self, *args):
        # HACK : user preference is to not trash custom_dem and zone_list on 
        # tile change. Instead added a new shortcut for trashing all high zl 
        # list in the custom ZL window at once.
        return
        CFG.custom_dem = ""
        try:
            self.config_window.v_["custom_dem"].set("")
        except:
            pass
        CFG.zone_list = []

    def update_cfg(self, *args):
        if self.default_website.get():
            CFG.default_website = str(self.default_website.get())
        if self.default_zl.get():
            CFG.default_zl = int(self.default_zl.get())

    def get_lat_lon(self, check=True):
        error_string = ""
        try:
            lat = int(self.lat.get())
            if lat < -85 or lat > 84:
                error_string += (
                    "Latitude out of range (-85,84) for webmercator grid. "
                )
        except:
            error_string += "Latitude wrongly encoded. "
        try:
            lon = int(self.lon.get())
            if lon < -180 or lon > 179:
                error_string += "Longitude out of range (-180,179)."
        except:
            error_string += "Longitude wrongly encoded."
        if error_string and check:
            UI.vprint(0, "Error: " + error_string)
            return None
        elif error_string:
            return (48, -6)
        return (lat, lon)

    def tile_from_interface(self):
        try:
            (lat, lon) = self.get_lat_lon()
            return CFG.Tile(lat, lon, str(self.custom_build_dir.get()))
        except:
            raise Exception

    def _run_background_stage(self, label, target, *args):
        result = 0
        try:
            result = target(*args)
        except Exception as error:
            UI.vprint(0, f"ERROR: {label} failed unexpectedly: {error}")
        finally:
            # Some legacy auxiliary actions do not reset this flag themselves.
            UI.is_working = 0
            self.stage_result_queue.put((label, result == 1, bool(UI.red_flag)))

    def _start_background_stage(self, label, target, *args):
        if getattr(self, "working_thread", None) is not None and self.working_thread.is_alive():
            message = _ui_text(
                "A process is already running.",
                "すでに別の処理を実行中です。",
            )
            UI.vprint(0, message)
            self.status_var.set(message)
            return 0
        self.status_var.set(_ui_text(f"Running: {label}", f"実行中: {label}"))
        self.working_thread = threading.Thread(
            target=self._run_background_stage,
            args=(label, target, *args),
        )
        self.working_thread.start()
        return 1

    def build_poly_file(self):
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("Vector data", VMAP.build_poly_file, tile)

    def build_mesh(self, event):
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("Mesh", MESH.build_mesh, tile)

    def sort_mesh(self, event):
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("Sort mesh", MESH.sort_mesh, tile)

    def community_mesh(self, event):
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("Community mesh", MESH.community_mesh, tile)

    def build_masks(self, event):
        for_imagery = "Shift" in str(event) or "shift" in str(event)
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("Masks", MASK.build_masks, tile, for_imagery)

    def build_tile(self):
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("Tile", TILE.build_tile, tile)

    def build_all(self):
        try:
            tile = self.tile_from_interface()
            tile.make_dirs()
        except:
            UI.vprint(1, "Process aborted.\n")
            return 0
        return self._start_background_stage("All steps", TILE.build_all, tile)

    def choose_custom_build_dir(self):
        tmp = filedialog.askdirectory()
        if tmp:
            tmp += "/"
        self.custom_build_dir.set(tmp)

    def open_config_window(self):
        try:
            self.config_window.lift()
            return 1
        except:
            try:
                (lat, lon) = self.get_lat_lon()
            except:
                return 0
            self.config_window = CFG.Ortho4XP_Config(self)
            return 1

    def open_earth_window(self):
        try:
            self.earth_window.lift()
            return 1
        except:
            try:
                (lat, lon) = self.get_lat_lon(check=False)
            except:
                (lat, lon) = (48, -6)
            self.earth_window = Ortho4XP_Earth_Preview(self, lat, lon)
            return 1

    def open_custom_zl_window(self):
        try:
            self.custom_zl_window.lift()
            return 1
        except:
            try:
                (lat, lon) = self.get_lat_lon()
            except:
                return 0
            self.custom_zl_window = Ortho4XP_Custom_ZL(self, lat, lon)
            return 1

    def set_red_flag(self):
        UI.red_flag = True
        self.status_var.set(
            _ui_text(
                "Stop requested. Waiting for the current step to stop...",
                "停止を要求しました。現在の処理が停止するまでお待ちください...",
            )
        )

    def check_dependencies(self):
        UI.vprint(0, _ui_text("\nChecking dependencies...", "\n依存関係を確認しています..."))
        all_found = True
        configured_dds_converter = getattr(UI, "dds_converter", "nvcompress")
        needs_ashelper = OsX and (
            configured_dds_converter == "TextureConverter"
            or getattr(UI, "use_neural_upscale", False)
            or getattr(UI, "use_gpu_acceleration", False)
        )

        tools = {
            "gdal_translate": ("gdal_translate", "GDAL (for geotagging/conversion)"),
            "gdalwarp": ("gdalwarp", "GDAL (for reprojecting)"),
            "7z": (OVL.unzip_cmd.strip(), "7-Zip (for DSF archives)"),
            "DSFTool": (OVL.dsftool_cmd.strip(), "bundled DSFTool (for overlays)"),
            "Triangle4XP": (MESH.Triangle4XP_cmd.strip(), "bundled Triangle4XP (for meshes)"),
        }
        required_tools = set(tools)
        if OsX:
            tools["ASHelper"] = (
                os.path.join(FNAMES.Utils_dir, "mac", "ASHelper"),
                "bundled ASHelper (for Apple Silicon conversion/upscale)",
            )
            if configured_dds_converter == "magick":
                tools["magick"] = ("magick", "ImageMagick (configured DDS converter)")
                required_tools.add("magick")
            elif configured_dds_converter == "nvcompress":
                tools["nvcompress"] = (
                    getattr(IMG, "dds_convert_cmd", "nvcompress"),
                    "NVIDIA Texture Tools (configured DDS converter)",
                )
                required_tools.add("nvcompress")
            if needs_ashelper:
                required_tools.add("ASHelper")
        else:
            tools["nvcompress"] = (
                getattr(IMG, "dds_convert_cmd", "nvcompress"),
                "NVIDIA Texture Tools (configured DDS converter)",
            )
            required_tools.add("nvcompress")

        for label, (candidate, desc) in tools.items():
            candidate = str(candidate).strip()
            path = (
                shutil.which(candidate)
                if os.path.basename(candidate) == candidate
                else candidate
            )
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                suffix = "" if label in required_tools else " (optional)"
                UI.vprint(
                    0,
                    _ui_text(
                        f"  [OK] {label} found{suffix}: {path}",
                        f"  [OK] {label} を検出しました{suffix}: {path}",
                    ),
                )
            else:
                suffix = "" if label in required_tools else " (optional)"
                UI.vprint(
                    0,
                    _ui_text(
                        f"  [!!] {label} NOT FOUND{suffix} ({desc})",
                        f"  [!!] {label} が見つかりません{suffix} ({desc})",
                    ),
                )
                if label in required_tools:
                    all_found = False

        python_modules = ("numpy", "PIL", "requests", "shapely", "rtree", "skfmm", "cv2", "osgeo")
        import importlib
        for module_name in python_modules:
            try:
                importlib.import_module(module_name)
            except Exception as error:
                UI.vprint(
                    0,
                    _ui_text(
                        f"  [!!] Python module {module_name} NOT FOUND ({error})",
                        f"  [!!] Pythonモジュール {module_name} が見つかりません ({error})",
                    ),
                )
                all_found = False
            else:
                UI.vprint(
                    0,
                    _ui_text(
                        f"  [OK] Python module {module_name} available",
                        f"  [OK] Pythonモジュール {module_name} を利用できます",
                    ),
                )

        scenery_root = getattr(OVL, "custom_overlay_src", "")
        scenery_roots = FNAMES.global_scenery_roots(scenery_root)
        resolved_scenery_root = FNAMES.resolve_global_scenery_root(scenery_root)
        if scenery_root and os.path.isdir(scenery_root):
            UI.vprint(
                0,
                _ui_text(
                    f"  [OK] Global Scenery root found: {scenery_root}",
                    f"  [OK] Global Sceneryのルートを検出しました: {scenery_root}",
                ),
            )
            if resolved_scenery_root is not None:
                UI.vprint(
                    0,
                    _ui_text(
                        f"  [OK] Global Scenery hierarchy resolved: {resolved_scenery_root}",
                        f"  [OK] Global Sceneryの配置を解決しました: {resolved_scenery_root}",
                    ),
                )
            elif len(scenery_roots) > 1:
                UI.vprint(
                    0,
                    _ui_text(
                        "  [!!] Global Scenery is ambiguous; choose one scenery package:",
                        "  [!!] Global Sceneryの候補が複数あります。1つのシーナリーパッケージを指定してください:",
                    ),
                )
                for root in scenery_roots:
                    UI.vprint(0, f"       {root}")
                all_found = False
            else:
                UI.vprint(
                    0,
                    _ui_text(
                        "  [!!] Global Scenery Earth nav data directory could not be resolved",
                        "  [!!] Global SceneryのEarth nav dataを解決できませんでした",
                    ),
                )
                all_found = False
        else:
            UI.vprint(
                0,
                _ui_text(
                    f"  [!!] Global Scenery root NOT FOUND: {scenery_root or '(empty)'}",
                    f"  [!!] Global Sceneryのルートが見つかりません: {scenery_root or '(空のパス)'}",
                ),
            )
            all_found = False
        
        if all_found:
            UI.vprint(
                0,
                _ui_text(
                    "All essential dependencies are satisfied!\n",
                    "必要な依存関係はすべて揃っています。\n",
                ),
            )
        else:
            UI.vprint(
                0,
                _ui_text(
                    "Some dependencies are missing. Please install them via Homebrew.\n",
                    "不足している依存関係があります。Homebrewでインストールしてください。\n",
                ),
            )
            if OsX:
                UI.vprint(
                    0,
                    _ui_text(
                        "Try: brew install gdal imagemagick\n",
                        "例: brew install gdal imagemagick\n",
                    ),
                )
        return 1 if all_found else 0

    def exit_prg(self):
        try:
            f = open(
                os.path.join(FNAMES.Ortho4XP_dir, ".last_gui_params.txt"), "w"
            )
            f.write(
                self.lat.get()
                + " "
                + self.lon.get()
                + " "
                + self.default_website.get()
                + " "
                + self.default_zl.get()
                + "\n"
            )
            f.write(self.custom_build_dir.get())
            f.close()
        except:
            pass
        self.after_cancel(self.callback_pgrb)
        self.after_cancel(self.callback_status)
        self.after_cancel(self.callback_console)
        sys.stdout = self.stdout_orig
        self.destroy()

################################################################################
class Ortho4XP_Custom_ZL(tk.Toplevel):

    dico_color = {
        15: "cyan",
        16: "green",
        17: "yellow",
        18: "orange",
        19: "red",
    }
    zl_list = ["10", "11", "12", "13"]
    points = []
    coords = []
    polygon_list = []
    polyobj_list = []

    def __init__(self, parent, lat, lon):
        self.parent = parent
        self.lat = lat
        self.lon = lon
        self.map_list = sorted(
            [
                provider_code
                for provider_code in set(IMG.providers_dict)
                if IMG.providers_dict[provider_code]["in_GUI"]
            ]
            + sorted(set(IMG.combined_providers_dict))
        )
        self.map_list = [
            provider_code
            for provider_code in self.map_list
            if provider_code != "SEA"
        ]
        self.reduced_map_list = [
            provider_code
            for provider_code in self.map_list
            if provider_code != "OSM"
        ]
        self.points = []
        self.coords = []
        self.polygon_list = []
        self.polyobj_list = []

        tk.Toplevel.__init__(self)
        self.title("Preview / Custom zoomlevels")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Constants

        self.map_choice = tk.StringVar()
        self.map_choice.set("OSM")
        self.zl_choice = tk.StringVar()
        self.zl_choice.set("11")
        self.progress_preview = tk.IntVar()
        self.progress_preview.set(0)
        self.zmap_choice = tk.StringVar()
        self.zmap_choice.set(self.parent.default_website.get())

        self.zlpol = tk.IntVar()
        try:  # default_zl might still be empty
            self.zlpol.set(
                max(min(int(self.parent.default_zl.get()) + 1, 19), 15)
            )
        except:
            self.zlpol.set(17)
        self.gb = tk.StringVar()
        self.gb.set("0Gb")

        # Frames
        self.frame_left = tk.Frame(
            self, border=4, relief=RIDGE, bg=UI.BG_COLOR
        )
        self.frame_left.grid(row=0, column=0, sticky=N + S + W + E)

        self.frame_right = tk.Frame(
            self, border=4, relief=RIDGE, bg=UI.BG_COLOR
        )
        self.frame_right.grid(row=0, column=1, sticky=N + S + W + E)
        self.frame_right.rowconfigure(0, weight=1)
        self.frame_right.columnconfigure(0, weight=1)

        # Widgets
        row = 0
        tk.Label(
            self.frame_left,
            anchor=W,
            text="Preview params ",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=row, column=0, sticky=W + E)
        row += 1

        tk.Label(
            self.frame_left, anchor=W, text="Source : ", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=row, column=0, padx=5, pady=3, sticky=W)
        self.map_combo = ttk.Combobox(
            self.frame_left,
            textvariable=self.map_choice,
            values=self.map_list,
            width=10,
            state="readonly",
            style="O4.TCombobox",
        )
        self.map_combo.grid(row=row, column=0, padx=5, pady=3, sticky=E)
        row += 1

        tk.Label(
            self.frame_left, anchor=W, text="Zoomlevel : ", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=row, column=0, padx=5, pady=3, sticky=W)
        self.zl_combo = ttk.Combobox(
            self.frame_left,
            textvariable=self.zl_choice,
            values=self.zl_list,
            width=3,
            state="readonly",
            style="O4.TCombobox",
        )
        self.zl_combo.grid(row=2, column=0, padx=5, pady=3, sticky=E)
        row += 1

        ttk.Button(
            self.frame_left,
            text="Preview",
            command=lambda: self.preview_tile(lat, lon),
        ).grid(row=row, padx=5, column=0, sticky=N + S + E + W)
        row += 1
        tk.Label(
            self.frame_left,
            anchor=W,
            text="Zone params ",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=row, column=0, pady=10, sticky=W + E)
        row += 1

        tk.Label(
            self.frame_left, anchor=W, text="Source : ", bg=UI.BG_COLOR, fg=UI.FG_COLOR
        ).grid(row=row, column=0, sticky=W, padx=5, pady=10)
        self.zmap_combo = ttk.Combobox(
            self.frame_left,
            textvariable=self.zmap_choice,
            values=self.reduced_map_list,
            width=8,
            state="readonly",
            style="O4.TCombobox",
        )
        self.zmap_combo.grid(row=row, column=0, padx=5, pady=10, sticky=E)
        row += 1

        self.frame_zlbtn = tk.Frame(self.frame_left, border=0, bg=UI.BG_COLOR)
        for i in range(5):
            self.frame_zlbtn.columnconfigure(i, weight=1)
        self.frame_zlbtn.grid(
            row=row, column=0, columnspan=1, sticky=N + S + W + E
        )
        row += 1
        for zl in range(15, 20):
            col = zl - 15
            tk.Radiobutton(
                self.frame_zlbtn,
                bd=4,
                bg=self.dico_color[zl],
                activebackground=self.dico_color[zl],
                selectcolor=self.dico_color[zl],
                height=2,
                indicatoron=0,
                text="ZL" + str(zl),
                variable=self.zlpol,
                value=zl,
                command=self.redraw_poly,
            ).grid(row=0, column=col, padx=0, pady=0, sticky=N + S + E + W)

        tk.Label(
            self.frame_left,
            anchor=W,
            text="Approx. Add. Size : ",
            bg=UI.BG_COLOR,
            fg=UI.FG_COLOR,
        ).grid(row=row, column=0, padx=5, pady=10, sticky=W)
        tk.Entry(
            self.frame_left,
            width=7,
            justify=RIGHT,
            bg=UI.ENTRY_BG,
            fg=UI.ENTRY_FG,
            textvariable=self.gb,
            insertbackground=UI.FG_COLOR,
        ).grid(row=row, column=0, padx=5, pady=10, sticky=E)
        row += 1

        ttk.Button(
            self.frame_left, text="  Save zone  ", command=self.save_zone_cmd
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        ttk.Button(
            self.frame_left, text="Delete ZL zone", command=self.delete_zone_cmd
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        ttk.Button(
            self.frame_left,
            text="Make GeoTiffs",
            command=self.build_geotiffs_ifc,
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        ttk.Button(
            self.frame_left, text="Extract Mesh ", command=self.extract_mesh_ifc
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        shortcut_text = "Cmd+B1 : add texture\nShift+B1: add zone point\nCmd+B2 : delete zone" if OsX else \
                       "Ctrl+B1 : add texture\nShift+B1: add zone point\nCtrl+B2 : delete zone"
        tk.Label(
            self.frame_left,
            text=shortcut_text,
            bg=UI.BG_COLOR,
            fg=UI.FG_COLOR,
            justify=LEFT,
        ).grid(row=row, column=0, padx=5, pady=20, sticky=N + S + E + W)
        row += 1
        ttk.Button(
            self.frame_left, text="    Apply    ", command=self.save_zone_list
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        ttk.Button(
            self.frame_left, text="    Reset    ", command=self.delAll
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        ttk.Button(
            self.frame_left, text="    Exit     ", command=self.destroy
        ).grid(row=row, column=0, padx=5, pady=3, sticky=N + S + E + W)
        row += 1
        self.canvas = tk.Canvas(self.frame_right, bd=0, height=750, width=750)
        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)

    def preview_tile(self, lat, lon):
        self._tile_preview_request_id = getattr(
            self, "_tile_preview_request_id", 0
        ) + 1
        request_id = self._tile_preview_request_id
        self.zoomlevel = int(self.zl_combo.get())
        zoomlevel = self.zoomlevel
        provider_code = self.map_combo.get()
        (tilxleft, tilytop) = GEO.wgs84_to_gtile(lat + 1, lon, zoomlevel)
        (self.latmax, self.lonmin) = GEO.gtile_to_wgs84(
            tilxleft, tilytop, zoomlevel
        )
        (self.xmin, self.ymin) = GEO.wgs84_to_pix(
            self.latmax, self.lonmin, zoomlevel
        )
        (tilxright, tilybot) = GEO.wgs84_to_gtile(lat, lon + 1, zoomlevel)
        (self.latmin, self.lonmax) = GEO.gtile_to_wgs84(
            tilxright + 1, tilybot + 1, zoomlevel
        )
        (self.xmax, self.ymax) = GEO.wgs84_to_pix(
            self.latmin, self.lonmax, zoomlevel
        )
        filepreview = FNAMES.preview(lat, lon, zoomlevel, provider_code)
        if os.path.isfile(filepreview) != True:
            fargs_ctp = [lat, lon, zoomlevel, provider_code]
            self.ctp_thread = threading.Thread(
                target=IMG.create_tile_preview, args=fargs_ctp
            )
            self.ctp_thread.start()
            self.after(
                50,
                lambda: self._wait_for_tile_preview(
                    request_id, filepreview, lat, lon
                ),
            )
        else:
            self.show_tile_preview(request_id, filepreview, lat, lon)
        return

    def _wait_for_tile_preview(self, request_id, filepreview, lat, lon):
        if request_id != getattr(self, "_tile_preview_request_id", 0):
            return
        try:
            if self.ctp_thread.is_alive():
                self.after(
                    50,
                    lambda: self._wait_for_tile_preview(
                        request_id, filepreview, lat, lon
                    ),
                )
                return
        except:
            pass
        self.show_tile_preview(request_id, filepreview, lat, lon)

    def show_tile_preview(self, request_id, filepreview, lat, lon):
        if request_id != getattr(self, "_tile_preview_request_id", 0):
            return
        for item in self.polyobj_list:
            try:
                self.canvas.delete(item)
            except:
                pass
        try:
            self.canvas.delete(self.img_map)
        except:
            pass
        try:
            self.canvas.delete(self.boundary)
        except:
            pass
        try:
            self.ctp_thread.join()
        except:
            pass
        self.image = Image.open(filepreview)
        self.photo = ImageTk.PhotoImage(self.image)
        self.map_x_res = self.photo.width()
        self.map_y_res = self.photo.height()
        try:
            self.img_map = self.canvas.create_image(
                0, 0, anchor=NW, image=self.photo
            )
            self.canvas.config(scrollregion=self.canvas.bbox(ALL))
        except:
            return
        if OsX:
            self.canvas.bind("<ButtonPress-1>", self.scroll_start)
            self.canvas.bind("<B1-Motion>", self.scroll_move)
            self.canvas.bind("<ButtonRelease-1>", self.scroll_stop)
            self.canvas.bind("<ButtonPress-2>", self.scroll_start)
            self.canvas.bind("<B2-Motion>", self.scroll_move)
            self.canvas.bind("<ButtonRelease-2>", self.scroll_stop)
            self.canvas.bind("<Command-ButtonPress-2>", self.delPol)
            # Mac Trackpad panning & zooming
            self.canvas.bind("<MouseWheel>", self._on_mousewheel)
            self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
            self.canvas.bind("<Command-MouseWheel>", self._on_zoom_wheel)
        else:
            self.canvas.bind("<ButtonPress-3>", self.scroll_start)
            self.canvas.bind("<B3-Motion>", self.scroll_move)
            self.canvas.bind("<ButtonRelease-3>", self.scroll_stop)
            self.canvas.bind("<Control-ButtonPress-3>", self.delPol)
        self.canvas.bind(
            "<ButtonPress-1>", lambda event: self.canvas.focus_set()
        ) if not OsX else None
        self.canvas.bind("<Shift-ButtonPress-1>", self.newPoint)
        mod_shift_key = "<Shift-Command-ButtonPress-1>" if OsX else "<Control-Shift-ButtonPress-1>"
        self.canvas.bind(mod_shift_key, self.newPointGrid)
        mod_key = "<Command-ButtonPress-1>" if OsX else "<Control-ButtonPress-1>"
        self.canvas.bind(mod_key, self.newPol)
        self.canvas.focus_set()
        self.canvas.bind("p", self.newPoint)
        self.canvas.bind("d", self.delete_zone_cmd)
        self.canvas.bind("n", self.save_zone_cmd)
        self.canvas.bind("<BackSpace>", self.delLast)
        self.polygon_list = []
        self.polyobj_list = []
        self.poly_curr = []
        bdpoints = []
        for [latp, lonp] in [
            [lat, lon],
            [lat, lon + 1],
            [lat + 1, lon + 1],
            [lat + 1, lon],
        ]:
            [x, y] = self.latlon_to_xy(latp, lonp, self.zoomlevel)
            bdpoints += [int(x), int(y)]
        self.boundary = self.canvas.create_polygon(
            bdpoints, outline="black", fill="", width=2
        )
        for zone in CFG.zone_list:
            self.coords = zone[0][0:-2]
            self.zlpol.set(zone[1])
            self.zmap_combo.set(zone[2])
            self.points = []
            for idxll in range(0, len(self.coords) // 2):
                latp = self.coords[2 * idxll]
                lonp = self.coords[2 * idxll + 1]
                [x, y] = self.latlon_to_xy(latp, lonp, self.zoomlevel)
                self.points += [int(x), int(y)]
            self.redraw_poly()
            self.save_zone_cmd()
        return

    def scroll_start(self, event):
        self.canvas.config(cursor="closedhand")
        self.canvas.focus_set()
        self.canvas.scan_mark(event.x, event.y)
        return

    def scroll_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        return

    def scroll_stop(self, event):
        self.canvas.config(cursor="")
        return

    def _on_mousewheel(self, event):
        delta = -1 * event.delta
        # Mac scroll delta is small, needs a multiplier for natural speed
        self.canvas.yview_scroll(int(delta), "units")

    def _on_shift_mousewheel(self, event):
        delta = -1 * event.delta
        self.canvas.xview_scroll(int(delta), "units")

    def _on_zoom_wheel(self, event):
        current_zl = int(self.zl_choice.get())
        if event.delta > 0:
            new_zl = min(current_zl + 1, 13)
        else:
            new_zl = max(current_zl - 1, 10)
        if new_zl != current_zl:
            # Calculate current view center to maintain position after zoom
            x_center = self.canvas.canvasx(self.canvas.winfo_width() / 2)
            y_center = self.canvas.canvasy(self.canvas.winfo_height() / 2)
            lat_c, lon_c = GEO.pix_to_wgs84(x_center, y_center, current_zl)
            self.zl_choice.set(str(new_zl))
            self.preview_tile(lat_c, lon_c)

    def redraw_poly(self):
        try:
            self.canvas.delete(self.poly_curr)
        except:
            pass
        try:
            color = self.dico_color[self.zlpol.get()]
            if len(self.points) >= 4:
                self.poly_curr = self.canvas.create_polygon(
                    self.points, outline=color, fill="", width=2
                )
            else:
                self.poly_curr = self.canvas.create_polygon(
                    self.points, outline=color, fill="", width=5
                )
        except:
            pass
        return

    def newPoint(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.points += [x, y]
        [latp, lonp] = self.xy_to_latlon(x, y, self.zoomlevel)
        self.coords += [latp, lonp]
        self.redraw_poly()
        return

    def newPointGrid(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        [latp, lonp] = self.xy_to_latlon(x, y, self.zoomlevel)
        [a, b] = GEO.wgs84_to_orthogrid(latp, lonp, self.zlpol.get())
        [aa, bb] = GEO.wgs84_to_gtile(latp, lonp, self.zlpol.get())
        a = a + 16 if aa - a >= 8 else a
        b = b + 16 if bb - b >= 8 else b
        [latp, lonp] = GEO.gtile_to_wgs84(a, b, self.zlpol.get())
        self.coords += [latp, lonp]
        [x, y] = self.latlon_to_xy(latp, lonp, self.zoomlevel)
        self.points += [int(x), int(y)]
        self.redraw_poly()
        return

    def newPol(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        [latp, lonp] = self.xy_to_latlon(x, y, self.zoomlevel)
        [a, b] = GEO.wgs84_to_orthogrid(latp, lonp, self.zlpol.get())
        [latmax, lonmin] = GEO.gtile_to_wgs84(a, b, self.zlpol.get())
        [latmin, lonmax] = GEO.gtile_to_wgs84(a + 16, b + 16, self.zlpol.get())
        self.coords = [
            latmin,
            lonmin,
            latmin,
            lonmax,
            latmax,
            lonmax,
            latmax,
            lonmin,
        ]
        self.points = []
        for i in range(4):
            [x, y] = self.latlon_to_xy(
                self.coords[2 * i], self.coords[2 * i + 1], self.zoomlevel
            )
            self.points += [int(x), int(y)]
        self.redraw_poly()
        self.save_zone_cmd()
        return

    def delPol(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        copy = self.polygon_list[:]
        for poly in copy:
            if poly[2] != self.zlpol.get():
                continue
            if VECT.point_in_polygon([x, y], poly[0]):
                idx = self.polygon_list.index(poly)
                self.polygon_list.pop(idx)
                self.canvas.delete(self.polyobj_list[idx])
                self.polyobj_list.pop(idx)
        return

    def delAll(self):
        copy = self.polygon_list[:]
        for poly in copy:
            idx = self.polygon_list.index(poly)
            self.polygon_list.pop(idx)
            self.canvas.delete(self.polyobj_list[idx])
            self.polyobj_list.pop(idx)
        try:
            self.canvas.delete(self.poly_curr)
        except:
            pass
        self.compute_size()
        return

    def xy_to_latlon(self, x, y, zoomlevel):
        pix_x = x + self.xmin
        pix_y = y + self.ymin
        return GEO.pix_to_wgs84(pix_x, pix_y, zoomlevel)

    def latlon_to_xy(self, lat, lon, zoomlevel):
        [pix_x, pix_y] = GEO.wgs84_to_pix(lat, lon, zoomlevel)
        return [pix_x - self.xmin, pix_y - self.ymin]

    def delLast(self, event):
        self.points = self.points[0:-2]
        self.coords = self.coords[0:-2]
        self.redraw_poly()
        return

    def compute_size(self):
        total_size = 0
        for polygon in self.polygon_list:
            polyp = polygon[0] + polygon[0][0:2]
            area = 0
            x1 = polyp[0]
            y1 = polyp[1]
            for j in range(1, len(polyp) // 2):
                x2 = polyp[2 * j]
                y2 = polyp[2 * j + 1]
                area += (x2 - x1) * (y2 + y1)
                x1 = x2
                y1 = y2
            total_size += (
                abs(area)
                / 2
                * (
                    (
                        40000
                        * cos(pi / 180 * polygon[1][0])
                        / 2 ** (int(self.zl_combo.get()) + 8)
                    )
                    ** 2
                )
                * 2 ** (2 * (int(polygon[2]) - 17))
                / 1024
            )
        self.gb.set("{:.1f}".format(total_size) + "Gb")
        return

    def save_zone_cmd(self):
        if len(self.points) < 6:
            return
        self.polyobj_list.append(self.poly_curr)
        self.polygon_list.append(
            [self.points, self.coords, self.zlpol.get(), self.zmap_combo.get()]
        )
        self.compute_size()
        self.poly_curr = []
        self.points = []
        self.coords = []
        return

    def build_geotiffs_ifc(self):
        texture_attributes_list = []
        fake_zone_list = []
        for polygon in self.polygon_list:
            lat_bar = (polygon[1][0] + polygon[1][4]) / 2
            lon_bar = (polygon[1][1] + polygon[1][3]) / 2
            zoomlevel = int(polygon[2])
            provider_code = polygon[3]
            til_x_left, til_y_top = GEO.wgs84_to_orthogrid(
                lat_bar, lon_bar, zoomlevel
            )
            texture_attributes_list.append(
                (til_x_left, til_y_top, zoomlevel, provider_code)
            )
            fake_zone_list.append(("", "", provider_code))
        UI.vprint(1, "\nBuilding geotiffs.\n------------------\n")
        tile = CFG.Tile(self.lat, self.lon, "")
        tile.zone_list = fake_zone_list
        if not IMG.initialize_local_combined_providers_dict(tile):
            UI.exit_message_and_bottom_line(
                "ERROR: Could not initialize local imagery providers."
            )
            return 0
        return self.parent._start_background_stage(
            "GeoTIFFs", IMG.build_geotiffs, tile, texture_attributes_list
        )

    def extract_mesh_ifc(self):
        polygon = self.polygon_list[0]
        lat_bar = (polygon[1][0] + polygon[1][4]) / 2
        lon_bar = (polygon[1][1] + polygon[1][3]) / 2
        zoomlevel = int(polygon[2])
        provider_code = polygon[3]
        til_x_left, til_y_top = GEO.wgs84_to_orthogrid(
            lat_bar, lon_bar, zoomlevel
        )
        build_dir = FNAMES.build_dir(
            self.lat, self.lon, self.parent.custom_build_dir.get()
        )
        mesh_file = FNAMES.mesh_file(build_dir, self.lat, self.lon)
        UI.vprint(
            1,
            "Extracting part of ",
            mesh_file,
            "to",
            FNAMES.obj_file(til_x_left, til_y_top, zoomlevel, provider_code),
            "(Wavefront)",
        )
        fargs_extract_mesh = [
            mesh_file,
            til_x_left,
            til_y_top,
            zoomlevel,
            provider_code,
        ]
        return self.parent._start_background_stage(
            "OBJ mesh", MESH.extract_mesh_to_obj, *fargs_extract_mesh
        )

    def delete_zone_cmd(self):
        try:
            self.canvas.delete(self.poly_curr)
            self.poly_curr = self.polyobj_list[-1]
            self.points = self.polygon_list[-1][0]
            self.coords = self.polygon_list[-1][1]
            self.zlpol.set(self.polygon_list[-1][2])
            self.zmap_combo.set(self.polygon_list[-1][3])
            self.polygon_list.pop(-1)
            self.polyobj_list.pop(-1)
            self.compute_size()
        except:
            self.points = []
            self.coords = []
        return

    def save_zone_list(self):
        ordered_list = sorted(
            self.polygon_list, key=lambda item: item[2], reverse=True
        )
        zone_list = []
        for item in ordered_list:
            tmp = []
            for pt in item[1]:
                tmp.append(pt)
            for pt in item[1][
                0:2
            ]:  # repeat first point for point_in_polygon algo
                tmp.append(pt)
            zone_list.append([tmp, item[2], item[3]])
        CFG.zone_list = zone_list
        # self.destroy()
        return

################################################################################
class Ortho4XP_Earth_Preview(tk.Toplevel):

    earthzl = 6
    resolution = 2 ** earthzl * 256

    list_del_ckbtn = [
        "OSM data",
        "Mask data",
        "Jpeg imagery",
        "Tile (whole)",
        "Tile (textures)",
    ]
    list_do_ckbtn = [
        "Assemble vector data",
        "Triangulate 3D mesh",
        "Draw water masks",
        "Build imagery/DSF",
        "Extract overlays",
        "Read per tile cfg",
    ]

    canvas_min_x = 900
    canvas_min_y = 700

    def __init__(self, parent, lat, lon):
        tk.Toplevel.__init__(self)
        self.title("Tiles collection and management")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Parent derived data
        self.parent = parent
        self.set_working_dir()

        # Constants/Variable
        self.dico_tiles_todo = {}
        self.dico_tiles_done = {}
        self.v_ = {}
        for item in self.list_del_ckbtn + self.list_do_ckbtn:
            self.v_[item] = tk.IntVar()
        self.latlon = tk.StringVar()

        # Frames
        self.frame_left = tk.Frame(
            self, border=4, relief=RIDGE, bg=UI.BG_COLOR
        )
        self.frame_left.grid(row=0, column=0, sticky=N + S + W + E)
        self.frame_right = tk.Frame(
            self, border=4, relief=RIDGE, bg=UI.BG_COLOR
        )
        self.frame_right.grid(row=0, rowspan=60, column=1, sticky=N + S + W + E)
        self.frame_right.rowconfigure(0, weight=1, minsize=self.canvas_min_y)
        self.frame_right.columnconfigure(0, weight=1, minsize=self.canvas_min_x)

        # Widgets
        row = 0
        tk.Label(
            self.frame_left,
            anchor=W,
            text="Active tile",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=row, column=0, sticky=W + E)
        row += 1
        self.latlon_entry = tk.Entry(
            self.frame_left,
            width=8,
            bg=UI.ENTRY_BG,
            fg=UI.ENTRY_FG,
            textvariable=self.latlon,
            insertbackground=UI.FG_COLOR,
        )
        self.latlon_entry.grid(row=row, column=0, padx=5, pady=5, sticky=N + S)
        row += 1
        # Trash
        tk.Label(
            self.frame_left,
            anchor=W,
            text="Erase cached data",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=row, column=0, sticky=W + E)
        row += 1
        for item in self.list_del_ckbtn:
            tk.Checkbutton(
                self.frame_left,
                text=item,
                anchor=W,
                variable=self.v_[item],
                bg=UI.BG_COLOR,
                fg=UI.FG_COLOR,
                activebackground=UI.BG_COLOR,
                activeforeground=UI.FG_COLOR,
                selectcolor=UI.ENTRY_BG,
                highlightthickness=0,
            ).grid(row=row, column=0, padx=5, pady=5, sticky=N + S + E + W)
            row += 1
        ttk.Button(
            self.frame_left, text="  Delete    ", command=self.trash
        ).grid(row=row, column=0, padx=5, pady=5, sticky=N + S + E + W)
        row += 1
        # Batch build
        tk.Label(
            self.frame_left,
            anchor=W,
            text="Batch build tiles",
            fg=UI.BG_COLOR,
            bg=UI.ACCENT_BG,
            font="Helvetica 16 bold italic",
        ).grid(row=row, column=0, sticky=W + E)
        row += 1
        for item in self.list_do_ckbtn:
            tk.Checkbutton(
                self.frame_left,
                text=item,
                anchor=W,
                variable=self.v_[item],
                bg=UI.BG_COLOR,
                fg=UI.FG_COLOR,
                activebackground=UI.BG_COLOR,
                activeforeground=UI.FG_COLOR,
                selectcolor=UI.ENTRY_BG,
                highlightthickness=0,
            ).grid(row=row, column=0, padx=5, pady=5, sticky=N + S + E + W)
            row += 1
        ttk.Button(
            self.frame_left, text="  Batch Build   ", command=self.batch_build
        ).grid(row=row, column=0, padx=5, pady=5, sticky=N + S + E + W)
        row += 1
        # Refresh window
        ttk.Button(
            self.frame_left, text="    Refresh     ", command=self.refresh
        ).grid(row=row, column=0, padx=5, pady=5, sticky=N + S + E + W)
        row += 1
        # Exit
        ttk.Button(
            self.frame_left, text="      Exit      ", command=self.exit
        ).grid(row=row, column=0, padx=5, pady=5, sticky=N + S + E + W)
        row += 1
        shortcut_text = "Shortcuts :\n-----------------\nScroll/B2-hold=move map\n" + \
                        "B1-double-click=select active\n" + \
                        "Shift+B1=add to batch build\nCmd+B1=link in Custom Scenery" if OsX else \
                        "Shortcuts :\n-----------------\nB2-press+hold=move map\n" + \
                        "B1-double-click=select active\n" + \
                        "Shift+B1=add to batch build\nCtrl+B1=link in Custom Scenery"
        tk.Label(
            self.frame_left,
            text=shortcut_text,
            bg=UI.BG_COLOR,
            fg=UI.FG_COLOR,
        ).grid(row=row, column=0, padx=0, pady=5, sticky=N + S + E + W)
        row += 1

        self.canvas = tk.Canvas(self.frame_right, bd=0)
        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)

        self.canvas.config(
            scrollregion=(
                1,
                1,
                2 ** self.earthzl * 256 - 1,
                2 ** self.earthzl * 256 - 1,
            )
        )  # self.canvas.bbox(ALL))
        (x0, y0) = GEO.wgs84_to_pix(lat + 0.5, lon + 0.5, self.earthzl)
        x0 = max(1, x0 - self.canvas_min_x / 2)
        y0 = max(1, y0 - self.canvas_min_y / 2)
        self.canvas.xview_moveto(x0 / self.resolution)
        self.canvas.yview_moveto(y0 / self.resolution)
        self.nx0 = int((8 * x0) // self.resolution)
        self.ny0 = int((8 * y0) // self.resolution)
        self._redraw_id = None
        if OsX:
            self.canvas.bind("<ButtonPress-1>", self.scroll_start)
            self.canvas.bind("<B1-Motion>", self.scroll_move)
            self.canvas.bind("<ButtonRelease-1>", self.scroll_stop)
            self.canvas.bind("<ButtonPress-2>", self.scroll_start)
            self.canvas.bind("<B2-Motion>", self.scroll_move)
            self.canvas.bind("<ButtonRelease-2>", self.scroll_stop)
            # Mac Trackpad panning
            self.canvas.bind("<MouseWheel>", self._on_mousewheel)
            self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        else:
            self.canvas.bind("<ButtonPress-3>", self.scroll_start)
            self.canvas.bind("<B3-Motion>", self.scroll_move)
            self.canvas.bind("<ButtonRelease-3>", self.scroll_stop)
        self.canvas.bind("<Double-Button-1>", self.select_tile)
        self.canvas.bind("<Shift-ButtonPress-1>", self.add_tile)
        mod_key = "<Command-ButtonPress-1>" if OsX else "<Control-ButtonPress-1>"
        self.canvas.bind(mod_key, self.toggle_to_custom)
        self.canvas.focus_set()
        self.draw_canvas(self.nx0, self.ny0)
        self.active_lat = lat
        self.active_lon = lon
        self.latlon.set(FNAMES.short_latlon(self.active_lat, self.active_lon))
        [x0, y0] = GEO.wgs84_to_pix(
            self.active_lat + 1, self.active_lon, self.earthzl
        )
        [x1, y1] = GEO.wgs84_to_pix(
            self.active_lat, self.active_lon + 1, self.earthzl
        )
        self.active_tile = self.canvas.create_rectangle(
            x0, y0, x1, y1, fill="", outline="yellow", width=3
        )
        self.threaded_preview()
        return

    def set_working_dir(self):
        self.custom_build_dir = self.parent.custom_build_dir.get()
        self.grouped = (
            self.custom_build_dir and self.custom_build_dir[-1] != "/"
        )
        self.working_dir = (
            self.custom_build_dir if self.custom_build_dir else FNAMES.Tile_dir
        )

    def refresh(self):
        self.set_working_dir()
        self.threaded_preview()
        return

    def threaded_preview(self):
        self._preview_request_id = getattr(self, "_preview_request_id", 0) + 1
        request_id = self._preview_request_id
        self._preview_result = None
        tiles_todo = list(self.dico_tiles_todo.keys())
        working_dir = self.working_dir
        grouped = self.grouped
        self._preview_thread = threading.Thread(
            target=self.preview_existing_tiles,
            args=(request_id, working_dir, grouped, tiles_todo),
        )
        self._preview_thread.start()
        self.after(50, lambda: self._poll_preview_existing_tiles(request_id))

    def preview_existing_tiles(
        self, request_id, working_dir, grouped, tiles_todo
    ):
        dico_color = {
            11: "blue",
            12: "blue",
            13: "blue",
            14: "blue",
            15: "cyan",
            16: "green",
            17: "yellow",
            18: "orange",
            19: "red",
        }
        tiles_done = {}
        if not os.path.isdir(working_dir):
            self._preview_result = (request_id, tiles_done, tiles_todo)
            return
        if not grouped:
            for dir_name in os.listdir(working_dir):
                if "XP_" in dir_name:
                    try:
                        lat = int(dir_name.split("XP_")[1][:3])
                        lon = int(dir_name.split("XP_")[1][3:7])
                    except:
                        continue
                    # With the enlarged accepetance rule for directory name 
                    # there might be more than one tile for the same (lat,lon),
                    # we skip all but the first encountered.
                    if (lat, lon) in tiles_done:
                        continue
                    if os.path.isfile(
                        os.path.join(
                            working_dir,
                            dir_name,
                            "Earth nav data",
                            FNAMES.long_latlon(lat, lon) + ".dsf",
                        )
                    ):
                        color = "blue"
                        content = ""
                        try:
                            tmpf = open(
                                os.path.join(
                                    working_dir,
                                    dir_name,
                                    "Ortho4XP_"
                                    + FNAMES.short_latlon(lat, lon)
                                    + ".cfg",
                                ),
                                "r",
                            )
                            found_config = True
                        except:
                            try:
                                tmpf = open(
                                    os.path.join(
                                        working_dir,
                                        dir_name,
                                        "Ortho4XP.cfg",
                                    ),
                                    "r",
                                )
                                found_config = True
                            except:
                                found_config = False
                        if found_config:
                            prov = zl = ""
                            for line in tmpf.readlines():
                                if line[:15] == "default_website":
                                    prov = line.strip().split("=")[1][:4]
                                elif line[:10] == "default_zl":
                                    zl = int(line.strip().split("=")[1])
                                    break
                            tmpf.close()
                            if not prov:
                                prov = "?"
                            if zl:
                                color = dico_color[zl]
                            else:
                                zl = "?"
                            content = prov + "\n" + str(zl)
                        else:
                            content = "?"
                        linked = False
                        link = os.path.join(
                            CFG.custom_scenery_dir,
                            "zOrtho4XP_" + FNAMES.short_latlon(lat, lon),
                        )
                        if os.path.isdir(link):
                            if os.path.samefile(
                                os.path.realpath(link),
                                os.path.realpath(os.path.join(working_dir, dir_name)),
                            ):
                                linked = True
                        tiles_done[(lat, lon)] = (color, content, dir_name, linked)
        elif grouped and os.path.isdir(
            os.path.join(working_dir, "Earth nav data")
        ):
            for dir_name in os.listdir(os.path.join(working_dir, "Earth nav data")):
                for file_name in os.listdir(
                    os.path.join(working_dir, "Earth nav data", dir_name)
                ):
                    try:
                        lat = int(file_name[0:3])
                        lon = int(file_name[3:7])
                    except:
                        continue
                    if (lat, lon) in tiles_done:
                        continue
                    color = "blue"
                    content = ""
                    try:
                        tmpf = open(
                            os.path.join(
                                working_dir,
                                "Ortho4XP_"
                                + FNAMES.short_latlon(lat, lon)
                                + ".cfg",
                            ),
                            "r",
                        )
                        found_config = True
                    except:
                        found_config = False
                    if found_config:
                        prov = zl = ""
                        for line in tmpf.readlines():
                            if line[:15] == "default_website":
                                prov = line.strip().split("=")[1][:4]
                            elif line[:10] == "default_zl":
                                zl = int(line.strip().split("=")[1])
                                break
                        tmpf.close()
                        if not prov:
                            prov = "?"
                        if zl:
                            color = dico_color[zl]
                        else:
                            zl = "?"
                        content = prov + "\n" + str(zl)
                    else:
                        content = "?"
                    linked = False
                    link = os.path.join(
                        CFG.custom_scenery_dir,
                        "zOrtho4XP_" + os.path.basename(working_dir),
                    )
                    if os.path.isdir(link) and os.path.samefile(
                        os.path.realpath(link), os.path.realpath(working_dir)
                    ):
                        linked = True
                    tiles_done[(lat, lon)] = (color, content, dir_name, linked)
        self._preview_result = (request_id, tiles_done, tiles_todo)
        return

    def _poll_preview_existing_tiles(self, request_id):
        if request_id != getattr(self, "_preview_request_id", 0):
            return
        try:
            if self._preview_thread.is_alive():
                self.after(
                    50, lambda: self._poll_preview_existing_tiles(request_id)
                )
                return
        except:
            return
        if self._preview_result is None:
            return
        result_request_id, tiles_done, tiles_todo = self._preview_result
        if result_request_id != request_id:
            return
        self._preview_result = None
        self.apply_preview_existing_tiles(tiles_done, tiles_todo)

    def apply_preview_existing_tiles(self, tiles_done, tiles_todo):
        if self.dico_tiles_done:
            for tile in self.dico_tiles_done:
                for objid in self.dico_tiles_done[tile][:2]:
                    self.canvas.delete(objid)
        self.dico_tiles_done = {}
        for (lat, lon), item in tiles_done.items():
            color, content, dir_name, link = item
            [x0, y0] = GEO.wgs84_to_pix(lat + 1, lon, self.earthzl)
            [x1, y1] = GEO.wgs84_to_pix(lat, lon + 1, self.earthzl)
            rect = (
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill=color, stipple="gray12"
                )
                if not OsX
                else self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="black"
                )
            )
            text = self.canvas.create_text(
                (x0 + x1) // 2,
                (y0 + y1) // 2,
                justify=CENTER,
                text=content,
                fill="black",
                font=("Helvetica", "12", "normal"),
            )
            self.dico_tiles_done[(lat, lon)] = (rect, text, dir_name)
            if link:
                if not OsX:
                    self.canvas.itemconfig(rect, stipple="gray50")
                else:
                    self.canvas.itemconfig(
                        text, font=("Helvetica", "12", "bold underline")
                    )
        for (lat, lon) in tiles_todo:
            [x0, y0] = GEO.wgs84_to_pix(lat + 1, lon, self.earthzl)
            [x1, y1] = GEO.wgs84_to_pix(lat, lon + 1, self.earthzl)
            objid = self.dico_tiles_todo.get((lat, lon))
            if objid:
                self.canvas.delete(objid)
            self.dico_tiles_todo[(lat, lon)] = (
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill="red", stipple="gray12"
                )
                if not OsX
                else self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="red", width=2
                )
            )
        return

    def trash(self):
        if self.v_["OSM data"].get():
            try:
                shutil.rmtree(FNAMES.osm_dir(self.active_lat, self.active_lon))
            except Exception as e:
                UI.vprint(3, e)
        if self.v_["Mask data"].get():
            try:
                shutil.rmtree(FNAMES.mask_dir(self.active_lat, self.active_lon))
            except Exception as e:
                UI.vprint(3, e)
        if self.v_["Jpeg imagery"].get():
            try:
                shutil.rmtree(
                    os.path.join(
                        FNAMES.Imagery_dir,
                        FNAMES.long_latlon(self.active_lat, self.active_lon),
                    )
                )
            except Exception as e:
                UI.vprint(3, e)
        if self.v_["Tile (whole)"].get() and not self.grouped:
            try:
                shutil.rmtree(
                    FNAMES.build_dir(
                        self.active_lat, self.active_lon, self.custom_build_dir
                    )
                )
            except Exception as e:
                UI.vprint(3, e)
            if (self.active_lat, self.active_lon) in self.dico_tiles_done:
                for objid in self.dico_tiles_done[
                    (self.active_lat, self.active_lon)
                ][:2]:
                    self.canvas.delete(objid)
                del self.dico_tiles_done[(self.active_lat, self.active_lon)]
        if self.v_["Tile (textures)"].get() and not self.grouped:
            try:
                shutil.rmtree(
                    os.path.join(
                        FNAMES.build_dir(
                            self.active_lat,
                            self.active_lon,
                            self.custom_build_dir,
                        ),
                        "textures",
                    )
                )
            except Exception as e:
                UI.vprint(3, e)
        return

    def select_tile(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        (lat, lon) = [floor(t) for t in GEO.pix_to_wgs84(x, y, self.earthzl)]
        self.active_lat = lat
        self.active_lon = lon
        self.latlon.set(FNAMES.short_latlon(lat, lon))
        try:
            self.canvas.delete(self.active_tile)
        except:
            pass
        [x0, y0] = GEO.wgs84_to_pix(lat + 1, lon, self.earthzl)
        [x1, y1] = GEO.wgs84_to_pix(lat, lon + 1, self.earthzl)
        self.active_tile = self.canvas.create_rectangle(
            x0, y0, x1, y1, fill="", outline="yellow", width=3
        )
        self.parent.lat.set(lat)
        self.parent.lon.set(lon)
        return

    def toggle_to_custom(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        (lat, lon) = [floor(t) for t in GEO.pix_to_wgs84(x, y, self.earthzl)]
        if (lat, lon) not in self.dico_tiles_done:
            return
        if not self.grouped:
            link = os.path.join(
                CFG.custom_scenery_dir,
                "zOrtho4XP_" + FNAMES.short_latlon(lat, lon),
            )
            # target=os.path.realpath(os.path.join(self.working_dir,
            # 'zOrtho4XP_'+FNAMES.short_latlon(lat,lon)))
            target = os.path.realpath(
                os.path.join(
                    self.working_dir, self.dico_tiles_done[(lat, lon)][-1]
                )
            )
            if os.path.isdir(link) and os.path.samefile(
                os.path.realpath(link), target
            ):
                os.remove(link)
                if not OsX:
                    self.canvas.itemconfig(
                        self.dico_tiles_done[(lat, lon)][0], stipple="gray12"
                    )
                else:
                    self.canvas.itemconfig(
                        self.dico_tiles_done[(lat, lon)][1],
                        font=("Helvetica", "12", "normal"),
                    )
                return
        elif self.grouped:
            link = os.path.join(
                CFG.custom_scenery_dir,
                "zOrtho4XP_" + os.path.basename(self.working_dir),
            )
            target = os.path.realpath(self.working_dir)
            if os.path.isdir(link) and os.path.samefile(
                os.path.realpath(link), os.path.realpath(self.working_dir)
            ):
                os.remove(link)
                for (lat0, lon0) in self.dico_tiles_done:
                    if not OsX:
                        self.canvas.itemconfig(
                            self.dico_tiles_done[(lat0, lon0)][0],
                            stipple="gray12",
                        )
                    else:
                        self.canvas.itemconfig(
                            self.dico_tiles_done[(lat0, lon0)][1],
                            font=("Helvetica", "12", "normal"),
                        )
                return
        # in case this was a broken link
        try:
            os.remove(link)
        except:
            pass
        if ("dar" in sys.platform) or (
            "win" not in sys.platform
        ):  # Mac and Linux
            os.system("ln -s " + ' "' + target + '" "' + link + '"')
        else:
            os.system('MKLINK /J "' + link + '" "' + target + '"')
        if not self.grouped:
            if not OsX:
                self.canvas.itemconfig(
                    self.dico_tiles_done[(lat, lon)][0], stipple="gray50"
                )
            else:
                self.canvas.itemconfig(
                    self.dico_tiles_done[(lat, lon)][1],
                    font=("Helvetica", "12", "bold underline"),
                )
        else:
            for (lat0, lon0) in self.dico_tiles_done:
                if not OsX:
                    self.canvas.itemconfig(
                        self.dico_tiles_done[(lat0, lon0)][0], stipple="gray50"
                    )
                else:
                    self.canvas.itemconfig(
                        self.dico_tiles_done[(lat0, lon0)][1],
                        font=("Helvetica", "12", "bold underline"),
                    )
        return

    def add_tile(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        (lat, lon) = [floor(t) for t in GEO.pix_to_wgs84(x, y, self.earthzl)]
        if (lat, lon) not in self.dico_tiles_todo:
            [x0, y0] = GEO.wgs84_to_pix(lat + 1, lon, self.earthzl)
            [x1, y1] = GEO.wgs84_to_pix(lat, lon + 1, self.earthzl)
            if not OsX:
                self.dico_tiles_todo[(lat, lon)] = self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill="red", stipple="gray12"
                )
            else:
                self.dico_tiles_todo[(lat, lon)] = self.canvas.create_rectangle(
                    x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="red", width=1
                )
        else:
            self.canvas.delete(self.dico_tiles_todo[(lat, lon)])
            self.dico_tiles_todo.pop((lat, lon), None)
        return

    def batch_build(self):
        list_lat_lon = sorted(self.dico_tiles_todo.keys())
        if not list_lat_lon:
            return
        (lat, lon) = list_lat_lon[0]
        try:
            tile = CFG.Tile(lat, lon, self.custom_build_dir)
        except:
            return 0
        args = [
            tile,
            list_lat_lon,
            self.v_["Assemble vector data"].get(),
            self.v_["Triangulate 3D mesh"].get(),
            self.v_["Draw water masks"].get(),
            self.v_["Build imagery/DSF"].get(),
            self.v_["Extract overlays"].get(),
            self.v_["Read per tile cfg"].get(),
        ]
        return self.parent._start_background_stage(
            "Batch build", TILE.build_tile_list, *args
        )

    def scroll_start(self, event):
        self.canvas.config(cursor="closedhand")
        self.canvas.focus_set()
        self.canvas.scan_mark(event.x, event.y)
        return

    def scroll_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.queue_redraw()
        return

    def scroll_stop(self, event):
        self.canvas.config(cursor="")
        return

    def queue_redraw(self):
        if self._redraw_id:
            self.after_cancel(self._redraw_id)
        self._redraw_id = self.after(100, self.redraw_canvas)

    def _on_mousewheel(self, event):
        delta = -1 * event.delta
        self.canvas.yview_scroll(int(delta), "units")
        self.queue_redraw()

    def _on_shift_mousewheel(self, event):
        delta = -1 * event.delta
        self.canvas.xview_scroll(int(delta), "units")
        self.queue_redraw()

    def redraw_canvas(self):
        x0 = self.canvas.canvasx(0)
        y0 = self.canvas.canvasy(0)
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        nx0 = int((8 * x0) // self.resolution)
        ny0 = int((8 * y0) // self.resolution)
        if nx0 == self.nx0 and ny0 == self.ny0:
            return
        else:
            self.nx0 = nx0
            self.ny0 = ny0
            try:
                self.canvas.delete(self.canv_imgNW)
            except:
                pass
            try:
                self.canvas.delete(self.canv_imgNE)
            except:
                pass
            try:
                self.canvas.delete(self.canv_imgSW)
            except:
                pass
            try:
                self.canvas.delete(self.canv_imgSE)
            except:
                pass
            fargs_rc = [nx0, ny0]
            self.rc_thread = threading.Thread(
                target=self.draw_canvas, args=fargs_rc
            )
            self.rc_thread.start()
            return

    def draw_canvas(self, nx0, ny0):
        fileprefix = os.path.join(
            FNAMES.Utils_dir, "Earth", "Earth2_ZL" + str(self.earthzl) + "_"
        )
        filepreviewNW = fileprefix + str(nx0) + "_" + str(ny0) + ".jpg"
        try:
            self.imageNW = Image.open(filepreviewNW)
            self.photoNW = ImageTk.PhotoImage(self.imageNW)
            self.canv_imgNW = self.canvas.create_image(
                nx0 * 2 ** self.earthzl * 256 / 8,
                ny0 * 2 ** self.earthzl * 256 / 8,
                anchor=NW,
                image=self.photoNW,
            )
            self.canvas.tag_lower(self.canv_imgNW)
        except:
            UI.lvprint(
                0,
                "Could not find Earth preview file",
                filepreviewNW,
                ", please update your installation from a fresh copy.",
            )
            return
        if nx0 < 2 ** (self.earthzl - 3) - 1:
            filepreviewNE = fileprefix + str(nx0 + 1) + "_" + str(ny0) + ".jpg"
            self.imageNE = Image.open(filepreviewNE)
            self.photoNE = ImageTk.PhotoImage(self.imageNE)
            self.canv_imgNE = self.canvas.create_image(
                (nx0 + 1) * 2 ** self.earthzl * 256 / 8,
                ny0 * 2 ** self.earthzl * 256 / 8,
                anchor=NW,
                image=self.photoNE,
            )
            self.canvas.tag_lower(self.canv_imgNE)
        if ny0 < 2 ** (self.earthzl - 3) - 1:
            filepreviewSW = fileprefix + str(nx0) + "_" + str(ny0 + 1) + ".jpg"
            self.imageSW = Image.open(filepreviewSW)
            self.photoSW = ImageTk.PhotoImage(self.imageSW)
            self.canv_imgSW = self.canvas.create_image(
                nx0 * 2 ** self.earthzl * 256 / 8,
                (ny0 + 1) * 2 ** self.earthzl * 256 / 8,
                anchor=NW,
                image=self.photoSW,
            )
            self.canvas.tag_lower(self.canv_imgSW)
        if (
            nx0 < 2 ** (self.earthzl - 3) - 1
            and ny0 < 2 ** (self.earthzl - 3) - 1
        ):
            filepreviewSE = (
                fileprefix + str(nx0 + 1) + "_" + str(ny0 + 1) + ".jpg"
            )
            self.imageSE = Image.open(filepreviewSE)
            self.photoSE = ImageTk.PhotoImage(self.imageSE)
            self.canv_imgSE = self.canvas.create_image(
                (nx0 + 1) * 2 ** self.earthzl * 256 / 8,
                (ny0 + 1) * 2 ** self.earthzl * 256 / 8,
                anchor=NW,
                image=self.photoSE,
            )
            self.canvas.tag_lower(self.canv_imgSE)
        return

    def exit(self):
        self.destroy()
