"""
eeg2bids-unify -- Node-Based GUI
Built with Dear PyGui + EEG waveform preview
Run: uv pip install dearpygui mne && python eeg2bids_gui.py
"""

import dearpygui.dearpygui as dpg
import os
import threading
import time
import numpy as np
from pathlib import Path

# ── State ─────────────────────────────────────────────────────────────────────
links            = {}
nodes            = {}
node_count       = 0
log_lines        = []
pipeline_running = False
EEG_TEX          = "eeg_texture"
PREVIEW_W        = 860
PREVIEW_H        = 340

HARDWARE_DEVICES = [
    ("BrainProducts", "ActiChamp Plus", ".vhdr / .vmrk / .eeg"),
    ("Neuroscan",     "NuAmps 40ch",    ".cnt"),
    ("OpenBCI",       "Cyton 8-ch",     ".txt"),
    ("Muse 2",        "InteraXon",      ".csv / .xdf"),
    ("Emotiv EPOC+",  "14 channels",    ".edf"),
]

NODE_COLORS = {
    "hardware":    (6,  90, 130),
    "harmonizer":  (180, 90,  10),
    "config":      (30, 130,  80),
    "bids_output": (90,  40, 160),
}

CH_COLORS = [
    (0.00, 0.78, 0.81, 1.0),
    (0.00, 0.78, 0.59, 1.0),
    (0.96, 0.65, 0.14, 1.0),
    (0.61, 0.43, 1.00, 1.0),
    (0.30, 0.65, 1.00, 1.0),
    (1.00, 0.36, 0.42, 1.0),
    (0.00, 0.88, 0.70, 1.0),
    (1.00, 0.78, 0.30, 1.0),
]

# ── EEG waveform renderer ─────────────────────────────────────────────────────
def blank_texture():
    return [14/255, 21/255, 32/255, 1.0] * (PREVIEW_W * PREVIEW_H)


def render_eeg(raw, n_ch=8, dur=10.0):
    sfreq  = raw.info["sfreq"]
    n_samp = int(min(dur, raw.times[-1]) * sfreq)
    n_ch   = min(n_ch, len(raw.ch_names))
    data, _= raw[:n_ch, :n_samp]

    # Normalize each channel
    normed = np.zeros_like(data)
    for i in range(n_ch):
        ch  = data[i]
        rng = ch.max() - ch.min()
        if rng < 1e-12: rng = 1.0
        normed[i] = (ch - ch.min()) / rng

    W, H   = PREVIEW_W, PREVIEW_H
    pixels = np.full((H, W, 4), [14/255, 21/255, 32/255, 1.0], dtype=np.float32)
    row_h  = H / n_ch

    for i in range(n_ch):
        y_c   = (i + 0.5) * row_h
        amp   = row_h * 0.40
        color = CH_COLORS[i % len(CH_COLORS)]

        # Divider line between channels
        dy = int((i + 1) * row_h)
        if 0 < dy < H:
            pixels[dy, :] = [0.14, 0.22, 0.32, 1.0]

        xs = np.linspace(6, W - 6, n_samp).astype(int)
        ys = (y_c - (normed[i] - 0.5) * 2 * amp).astype(int)
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)

        for dy2 in range(-1, 2):
            yy = np.clip(ys + dy2, 0, H - 1)
            pixels[yy, xs] = color

    return pixels.flatten().tolist()


def load_preview(filepath):
    try:
        import mne
        mne.set_log_level("WARNING")
        ext = Path(filepath).suffix.lower()
        log(f"Loading preview: {Path(filepath).name}...", "INFO")

        if   ext == ".vhdr": raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        elif ext == ".cnt":  raw = mne.io.read_raw_cnt(filepath, preload=True, verbose=False)
        elif ext == ".edf":  raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        elif ext == ".xdf":
            import pyxdf
            streams, _ = pyxdf.load_xdf(filepath)
            s    = next((x for x in streams if "EEG" in x["info"]["type"][0].upper()), streams[0])
            d    = np.array(s["time_series"]).T
            info = mne.create_info([f"CH{i+1}" for i in range(d.shape[0])],
                                   float(s["info"]["nominal_srate"][0]), ch_types="eeg")
            raw  = mne.io.RawArray(d * 1e-6, info)
        elif ext == ".csv":
            import pandas as pd
            df   = pd.read_csv(filepath)
            cols = [c for c in df.columns if c.startswith("RAW_")] or \
                   df.select_dtypes("number").columns.tolist()[:8]
            raw  = mne.io.RawArray(df[cols].values.T * 1e-6,
                                   mne.create_info(list(cols), 256.0, ch_types="eeg"))
        elif ext == ".txt":
            import pandas as pd
            df   = pd.read_csv(filepath, comment="%", header=None)
            data = df.iloc[:, 1:9].values.T * 1e-6
            raw  = mne.io.RawArray(data,
                                   mne.create_info([f"CH{i+1}" for i in range(data.shape[0])],
                                                   250.0, ch_types="eeg"))
        else:
            log(f"Preview not supported for {ext}", "WARN"); return

        n_ch = min(8, len(raw.ch_names))
        dur  = min(10.0, raw.times[-1])
        dpg.set_value(EEG_TEX, render_eeg(raw, n_ch=n_ch, dur=dur))

        # Update channel labels
        if dpg.does_item_exist("ch_labels"):
            dpg.delete_item("ch_labels", children_only=True)
            rh = PREVIEW_H / n_ch
            for i, ch in enumerate(raw.ch_names[:n_ch]):
                col = [int(c * 255) for c in CH_COLORS[i % len(CH_COLORS)][:3]]
                dpg.add_text(
                    ch[:12],
                    color=tuple(col),
                    parent="ch_labels")
                if i < n_ch - 1:
                    dpg.add_spacer(height=int(rh) - 18, parent="ch_labels")

        if dpg.does_item_exist("preview_info"):
            dpg.set_value("preview_info",
                f"  {Path(filepath).name}   |   "
                f"{len(raw.ch_names)} channels   |   "
                f"{raw.info['sfreq']:.0f} Hz   |   "
                f"{raw.times[-1]:.1f} sec total   |   "
                f"Showing first {n_ch} channels, first {dur:.0f}s")

        log(f"Preview ready: {n_ch} ch @ {raw.info['sfreq']:.0f} Hz, {raw.times[-1]:.1f}s", "OK")

    except ImportError:
        log("MNE not installed. Run: uv pip install mne", "ERR")
    except Exception as e:
        log(f"Preview error: {e}", "ERR")


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    colors = {"INFO":(180,220,255), "OK":(100,230,150), "WARN":(255,200,80), "ERR":(255,100,100)}
    log_lines.append((f"[{ts}] [{level}] {msg}", colors.get(level,(220,220,220))))
    if dpg.does_item_exist("log_panel"):
        refresh_log()

def refresh_log():
    dpg.delete_item("log_panel", children_only=True)
    for text, color in log_lines[-60:]:
        dpg.add_text(text, color=color, parent="log_panel")
    if dpg.does_item_exist("log_scroll"):
        dpg.set_y_scroll("log_scroll", dpg.get_y_scroll_max("log_scroll"))


# ── Node themes ───────────────────────────────────────────────────────────────
def node_theme(ntype):
    r, g, b = NODE_COLORS.get(ntype, (60,60,60))
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,         (r,g,b,220),       category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,  (r+30,g+30,b+30,220), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (r+50,g+50,b+50,220), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,   (22,30,42,235),    category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (30,42,56,235), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline,      (r,g,b,160),       category=dpg.mvThemeCat_Nodes)
    return t


# ── Nodes ─────────────────────────────────────────────────────────────────────
def add_hardware_node(device_name, model, fmt, pos=(100, 100)):
    global node_count
    node_count += 1
    nid = f"hw_{node_count}"

    with dpg.node(label=f"[HW]  {device_name}",
                  parent="node_editor", pos=pos, tag=nid):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_text(f"Device : {model}", color=(150,190,230))
            dpg.add_text(f"Format : {fmt}",   color=(120,160,210))
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(height=4)
            dpg.add_input_text(tag=f"{nid}_filepath", hint="Path to EEG file...", width=235)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Browse...", width=114,
                               callback=lambda: browse_file(nid))
                dpg.add_button(label="Preview EEG", width=117,
                               callback=lambda n=nid: _trigger_preview(n))
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("Subject:")
                dpg.add_input_text(tag=f"{nid}_sub", default_value="01", width=52)
                dpg.add_spacer(width=8)
                dpg.add_text("Session:")
                dpg.add_input_text(tag=f"{nid}_ses", default_value="01", width=52)
        with dpg.node_attribute(tag=f"{nid}_out", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Raw EEG  >>", color=(100,230,150))

    dpg.bind_item_theme(nid, node_theme("hardware"))
    nodes[nid] = {"type":"hardware", "device":device_name, "out":f"{nid}_out",
                  "file":f"{nid}_filepath", "sub":f"{nid}_sub", "ses":f"{nid}_ses"}
    log(f"Added {device_name} node", "OK")
    return nid


def _trigger_preview(nid):
    fp = dpg.get_value(f"{nid}_filepath")
    if not fp or not os.path.exists(fp):
        log("Set a valid file path first, then click Preview EEG", "WARN")
        return
    if dpg.does_item_exist("tab_preview"):
        dpg.set_value("tab_bar", "tab_preview")
    threading.Thread(target=load_preview, args=(fp,), daemon=True).start()


def add_harmonizer_node(pos=(420, 120)):
    global node_count
    node_count += 1
    nid = f"harm_{node_count}"
    with dpg.node(label="[PROC]  EventHarmonizer",
                  parent="node_editor", pos=pos, tag=nid):
        with dpg.node_attribute(tag=f"{nid}_in", attribute_type=dpg.mvNode_Attr_Input):
            dpg.add_text("<< Raw Events", color=(255,180,80))
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(height=4)
            dpg.add_text("Event Mappings  (key -> trial_type):", color=(200,200,200))
            for d in ['"1"    -> stimulus_6hz', '"2"    -> stimulus_8hz',
                      '"S  1" -> stimulus_6hz', '"99"   -> rest']:
                dpg.add_input_text(default_value=d, width=245)
        with dpg.node_attribute(tag=f"{nid}_out", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Harmonized Events  >>", color=(100,230,150))
    dpg.bind_item_theme(nid, node_theme("harmonizer"))
    nodes[nid] = {"type":"harmonizer", "in":f"{nid}_in", "out":f"{nid}_out"}
    log("Added EventHarmonizer node", "OK")
    return nid


def add_config_node(pos=(420, 390)):
    global node_count
    node_count += 1
    nid = f"cfg_{node_count}"
    with dpg.node(label="[CFG]  YAML Config",
                  parent="node_editor", pos=pos, tag=nid):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_text("Dataset Name:")
            dpg.add_input_text(tag=f"{nid}_name", default_value="My EEG Dataset", width=245)
            dpg.add_text("Institution:")
            dpg.add_input_text(tag=f"{nid}_inst", default_value="NTU Singapore", width=245)
            dpg.add_text("Task Label:")
            dpg.add_input_text(tag=f"{nid}_task", default_value="ssvep", width=245)
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("Power Line:")
                dpg.add_combo(tag=f"{nid}_pwr", items=["50 Hz","60 Hz"],
                              default_value="50 Hz", width=100)
            dpg.add_checkbox(tag=f"{nid}_validate", label="Run BIDS Validator", default_value=True)
            dpg.add_checkbox(tag=f"{nid}_overwrite", label="Overwrite Existing",  default_value=True)
        with dpg.node_attribute(tag=f"{nid}_out", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Config  >>", color=(100,230,150))
    dpg.bind_item_theme(nid, node_theme("config"))
    nodes[nid] = {"type":"config", "out":f"{nid}_out"}
    log("Added YAML Config node", "OK")
    return nid


def add_bids_output_node(pos=(730, 220)):
    global node_count
    node_count += 1
    nid = f"bids_{node_count}"
    with dpg.node(label="[OUT]  BIDS Output",
                  parent="node_editor", pos=pos, tag=nid):
        with dpg.node_attribute(tag=f"{nid}_events_in", attribute_type=dpg.mvNode_Attr_Input):
            dpg.add_text("<< Harmonized Events", color=(255,180,80))
        with dpg.node_attribute(tag=f"{nid}_config_in", attribute_type=dpg.mvNode_Attr_Input):
            dpg.add_text("<< Config", color=(255,180,80))
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(height=4)
            dpg.add_text("Output Directory:", color=(200,200,200))
            dpg.add_input_text(tag=f"{nid}_path", default_value="./bids_output", width=245)
            dpg.add_button(label="Browse...", width=245, callback=lambda: browse_folder(nid))
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_spacer(height=6)
            dpg.add_text("Status: Waiting...", tag=f"{nid}_status_text", color=(180,180,180))
            dpg.add_progress_bar(tag=f"{nid}_progress", default_value=0.0, width=245)
    dpg.bind_item_theme(nid, node_theme("bids_output"))
    nodes[nid] = {"type":"bids_output", "events_in":f"{nid}_events_in",
                  "config_in":f"{nid}_config_in", "path":f"{nid}_path",
                  "status_text":f"{nid}_status_text", "progress":f"{nid}_progress"}
    log("Added BIDS Output node", "OK")
    return nid


# ── Link callbacks ────────────────────────────────────────────────────────────
def link_callback(sender, app_data):
    lid = dpg.add_node_link(app_data[0], app_data[1], parent=sender)
    links[lid] = (app_data[0], app_data[1])
    log("Nodes connected", "OK")

def delink_callback(sender, app_data):
    dpg.delete_item(app_data)
    links.pop(app_data, None)
    log("Connection removed", "WARN")


# ── File browse ───────────────────────────────────────────────────────────────
def browse_file(nid):
    try:
        import tkinter as tk; from tkinter import filedialog
        tk.Tk().withdraw()
        p = filedialog.askopenfilename(title="Select EEG File",
            filetypes=[("EEG Files","*.vhdr *.cnt *.txt *.csv *.xdf *.edf *.bdf"),("All","*.*")])
        if p and dpg.does_item_exist(f"{nid}_filepath"):
            dpg.set_value(f"{nid}_filepath", p)
            log(f"File: {Path(p).name}", "OK")
    except Exception as e:
        log(f"Browse error: {e}", "ERR")

def browse_folder(nid):
    try:
        import tkinter as tk; from tkinter import filedialog
        tk.Tk().withdraw()
        p = filedialog.askdirectory(title="Select BIDS Output Directory")
        if p and dpg.does_item_exist(f"{nid}_path"):
            dpg.set_value(f"{nid}_path", p)
            log(f"Output dir: {p}", "OK")
    except Exception as e:
        log(f"Browse error: {e}", "ERR")


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline():
    global pipeline_running
    if pipeline_running: log("Already running!", "WARN"); return
    hw_nodes   = [n for n,d in nodes.items() if d["type"]=="hardware"]
    bids_nodes = [n for n,d in nodes.items() if d["type"]=="bids_output"]
    if not hw_nodes:   log("No hardware node found", "ERR"); return
    if not bids_nodes: log("No BIDS Output node found", "ERR"); return

    def _run():
        global pipeline_running
        pipeline_running = True
        bids_nid     = bids_nodes[0]
        status_tag   = nodes[bids_nid]["status_text"]
        progress_tag = nodes[bids_nid]["progress"]
        try:
            dpg.set_value(status_tag, "Status: Running...")
            dpg.configure_item(status_tag, color=(255,200,80))
            for hw_nid in hw_nodes:
                filepath = dpg.get_value(f"{hw_nid}_filepath")
                subject  = dpg.get_value(f"{hw_nid}_sub")
                session  = dpg.get_value(f"{hw_nid}_ses")
                device   = nodes[hw_nid]["device"]
                out_path = dpg.get_value(nodes[bids_nid]["path"])
                if not filepath or not os.path.exists(filepath):
                    log(f"[{device}] File not found -- skipping", "WARN"); continue
                log(f"[{device}] Processing {Path(filepath).name}...")
                dpg.set_value(progress_tag, 0.1)
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent / "src"))
                    from eeg2bids_unify.core.config import load_config
                    from eeg2bids_unify.core.converter import EEGConverter
                    log(f"[{device}] Backend loaded", "OK")
                    dpg.set_value(progress_tag, 0.3)
                    cfg = load_config()
                    converter = EEGConverter(cfg)
                    dpg.set_value(progress_tag, 0.5)
                    converter.convert(filepath=filepath, bids_root=out_path,
                                      subject=subject, session=session, task="ssvep")
                    dpg.set_value(progress_tag, 1.0)
                    log(f"[{device}] BIDS output -> {out_path}", "OK")
                except ImportError:
                    log(f"[{device}] Demo mode", "WARN")
                    for p in [0.3, 0.5, 0.7, 0.9, 1.0]:
                        dpg.set_value(progress_tag, p); time.sleep(0.22)
                    log(f"[{device}] Simulated BIDS -> {out_path}", "OK")
                except Exception as be:
                    log(f"[{device}] Error: {be}", "ERR")
                    dpg.set_value(progress_tag, 0.0)
            dpg.set_value(status_tag, "Status: Complete!")
            dpg.configure_item(status_tag, color=(100,230,150))
            log("Pipeline complete!", "OK")
        except Exception as e:
            log(f"Error: {e}", "ERR")
            dpg.set_value(status_tag, "Status: Error")
            dpg.configure_item(status_tag, color=(255,100,100))
        finally:
            pipeline_running = False

    threading.Thread(target=_run, daemon=True).start()


def clear_pipeline():
    for nid in list(nodes.keys()):
        if dpg.does_item_exist(nid): dpg.delete_item(nid)
    nodes.clear(); links.clear()
    log("Canvas cleared", "WARN")


def load_demo():
    clear_pipeline(); time.sleep(0.05)
    hw   = add_hardware_node("BrainProducts","ActiChamp Plus",".vhdr/.vmrk/.eeg",pos=(60,80))
    harm = add_harmonizer_node(pos=(400,80))
    cfg  = add_config_node(pos=(400,390))
    bids = add_bids_output_node(pos=(740,220))
    dpg.add_node_link(nodes[hw]["out"],   nodes[harm]["in"],       parent="node_editor")
    dpg.add_node_link(nodes[harm]["out"], nodes[bids]["events_in"],parent="node_editor")
    dpg.add_node_link(nodes[cfg]["out"],  nodes[bids]["config_in"],parent="node_editor")
    log("Demo pipeline loaded. Set file path -> click Run or Preview EEG", "OK")


# ── Global theme ──────────────────────────────────────────────────────────────
def apply_theme():
    with dpg.theme() as gt:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (15,18,25))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        (20,24,33))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (28,36,50))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (38,50,68))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        (12,16,24))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (18,26,40))
            dpg.add_theme_color(dpg.mvThemeCol_Button,         (6,90,130))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (10,115,165))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (14,140,195))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,      (0,220,160))
            dpg.add_theme_color(dpg.mvThemeCol_Text,           (210,220,235))
            dpg.add_theme_color(dpg.mvThemeCol_Tab,            (18,26,40))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,     (6,90,130))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,      (6,90,130))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,    (12,16,22))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  10, 10)
        with dpg.theme_component(dpg.mvNodeEditor):
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (12,15,22),    category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine,       (28,38,52),    category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link,           (0,190,160),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered,    (0,220,190),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected,   (0,240,200),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Pin,            (0,190,160),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_PinHovered,     (0,230,190),   category=dpg.mvThemeCat_Nodes)
    dpg.bind_theme(gt)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dpg.create_context()
    dpg.create_viewport(title="eeg2bids-unify  |  Node Pipeline Editor",
                        width=1340, height=880, min_width=900, min_height=640)
    apply_theme()

    with dpg.texture_registry():
        dpg.add_raw_texture(width=PREVIEW_W, height=PREVIEW_H,
                            default_value=blank_texture(),
                            format=dpg.mvFormat_Float_rgba,
                            tag=EEG_TEX)

    with dpg.window(tag="main_win", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True):

        # Toolbar
        with dpg.group(horizontal=True):
            dpg.add_text("eeg2bids-unify", color=(0,200,180))
            dpg.add_text("  Node Pipeline Editor", color=(110,140,170))
            dpg.add_spacer(width=18)
            dpg.add_button(label="  Run Pipeline  ", callback=run_pipeline,   width=130)
            dpg.add_button(label="  Load Demo  ",    callback=load_demo,       width=110)
            dpg.add_button(label="  Clear  ",        callback=clear_pipeline,  width=75)
            dpg.add_spacer(width=14)
            dpg.add_text("Add nodes -> connect pins -> Run   |   Click 'Preview EEG' on any hardware node to view signal",
                         color=(75,100,130))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            # Sidebar
            with dpg.child_window(width=208, border=True):
                dpg.add_text("HARDWARE INPUT", color=(0,190,160))
                dpg.add_separator(); dpg.add_spacer(height=4)
                for i,(name,model,fmt) in enumerate(HARDWARE_DEVICES):
                    def _add(n=name,m=model,f=fmt,idx=i):
                        add_hardware_node(n,m,f,pos=(60,60+idx*175))
                    dpg.add_button(label=f"+ {name}", width=192, callback=_add)
                    dpg.add_text(f"  {fmt}", color=(85,115,150)); dpg.add_spacer(height=5)

                dpg.add_separator()
                dpg.add_text("PROCESSING", color=(0,190,160))
                dpg.add_separator(); dpg.add_spacer(height=4)
                dpg.add_button(label="+ EventHarmonizer", width=192,
                               callback=lambda: add_harmonizer_node())
                dpg.add_text("  Normalize events", color=(85,115,150)); dpg.add_spacer(height=5)
                dpg.add_button(label="+ YAML Config", width=192,
                               callback=lambda: add_config_node())
                dpg.add_text("  Dataset settings", color=(85,115,150)); dpg.add_spacer(height=5)

                dpg.add_separator()
                dpg.add_text("OUTPUT", color=(0,190,160))
                dpg.add_separator(); dpg.add_spacer(height=4)
                dpg.add_button(label="+ BIDS Output", width=192,
                               callback=lambda: add_bids_output_node())
                dpg.add_text("  BIDS-EEG writer", color=(85,115,150))

                dpg.add_spacer(height=18); dpg.add_separator()
                dpg.add_text("HOW TO USE", color=(0,190,160))
                dpg.add_separator()
                for tip in ["1. Click hardware button","2. Add Harmonizer + Config",
                            "3. Add BIDS Output","4. Drag pins to connect",
                            "5. Set file paths","6. Click Run Pipeline",
                            "","TIP: Click 'Preview EEG'","   to see raw signal"]:
                    dpg.add_text(tip, color=(95,120,150), wrap=195)
                    dpg.add_spacer(height=2)

            # Tab bar
            with dpg.group():
                with dpg.tab_bar(tag="tab_bar"):

                    # Pipeline tab
                    with dpg.tab(label="  Pipeline Editor  ", tag="tab_pipeline"):
                        with dpg.child_window(border=True, height=-195, no_scrollbar=True):
                            with dpg.node_editor(tag="node_editor",
                                                 callback=link_callback,
                                                 delink_callback=delink_callback,
                                                 minimap=True,
                                                 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                                pass
                        with dpg.child_window(tag="log_scroll", border=True, height=188):
                            dpg.add_text("Console", color=(0,190,160))
                            dpg.add_separator()
                            with dpg.child_window(tag="log_panel", border=False, auto_resize_y=True):
                                pass

                    # EEG Preview tab
                    with dpg.tab(label="  EEG Signal Preview  ", tag="tab_preview"):
                        dpg.add_text(
                            "  No file loaded. Set a file in a hardware node, then click 'Preview EEG'.",
                            tag="preview_info", color=(110,140,170))
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            with dpg.child_window(tag="ch_labels", width=92,
                                                  height=PREVIEW_H, border=False, no_scrollbar=True):
                                dpg.add_text("Channels", color=(0,190,160))
                            dpg.add_image(EEG_TEX, width=PREVIEW_W, height=PREVIEW_H)
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            dpg.add_text("Tip:", color=(0,190,160))
                            dpg.add_text("First 8 channels, first 10 seconds. Each channel amplitude-normalized independently.",
                                         color=(110,140,170))

    dpg.set_primary_window("main_win", True)

    def on_resize(s, a):
        w = dpg.get_viewport_client_width()
        h = dpg.get_viewport_client_height()
        dpg.set_item_width("main_win",  w)
        dpg.set_item_height("main_win", h)
    with dpg.item_handler_registry(tag="vp_h"):
        dpg.add_item_resize_handler(callback=on_resize)
    dpg.bind_item_handler_registry("main_win", "vp_h")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    w = dpg.get_viewport_client_width()
    h = dpg.get_viewport_client_height()
    dpg.set_item_width("main_win",  w)
    dpg.set_item_height("main_win", h)

    log("eeg2bids-unify GUI ready", "OK")
    log("No emojis -- all labels use text for Windows compatibility", "INFO")
    log("Click 'Load Demo' for pre-wired pipeline", "INFO")
    log("Click 'Preview EEG' on any hardware node after setting file path", "INFO")

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()