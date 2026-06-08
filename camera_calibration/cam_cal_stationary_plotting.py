"""
thermal_viewer_zoom.py

Interactive viewer for thermal CSV data.

Controls:
Mouse hover       - live temperature readout at cursor position
Left drag         - pan heatmap or layer plot
Shift + click     - choose two times on the peaks graph and zoom to that interval
Double-click      - reset view on the clicked plot
Frame slider      - scrub through time
Space             - play / pause
Left / Right      - step one frame
Scroll wheel      - step one frame
a                 - save the minimum peak temp into a CSV
b                 - save detected peaks and minima into a CSV
Q / Escape        - quit
"""

CMAP = 'inferno'
INTERP = 'nearest'
TEMP_MIN = None
TEMP_MAX = None
PLAYBACK_FPS = 10.0
LAYER_SIGNAL_MODE = 'max'
LAYER_SMOOTH_SECONDS = 0.0
LAYER_MIN_PEAK_GAP_SECONDS = 3
LAYER_BASELINE_PERCENTILE = 35.0
LAYER_ACTIVE_PERCENTILE = 95.0
LAYER_THRESHOLD_FRACTION = 0.45

import os
import re
import sys

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


FILE_PICKER_INITIALDIR = 'automation_test/data/pa6cf/cal_21_high_gain/cal_r2_r8_test'

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
CSV_PATH = filedialog.askopenfilename(
    title='Select thermal CSV',
    filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
    initialdir=FILE_PICKER_INITIALDIR,
)
root.destroy()

if not CSV_PATH:
    print('No file selected - exiting.')
    sys.exit(0)

print(f'Selected: {CSV_PATH}')
print('Loading...')

with open(CSV_PATH, encoding='utf-8-sig') as f:
    headers = [h.strip() for h in f.readline().rstrip('\r\n').split(',')]

NEW_FORMAT = 'nozzle_cx' in headers
if NEW_FORMAT:
    px_cols = [h for h in headers if re.match(r'px_dr[+-]?\d+_dc[+-]?\d+', h)]
    dr_vals = sorted({int(re.search(r'px_dr([+-]?\d+)_dc', h).group(1)) for h in px_cols})
    dc_vals = sorted({int(re.search(r'px_dr[+-]?\d+_dc([+-]?\d+)', h).group(1)) for h in px_cols})
    ts_col = 'elapsed_s' if 'elapsed_s' in headers else 'timestamp'
    has_elapsed = 'elapsed_s' in headers
else:
    px_cols = [h for h in headers if re.match(r'px_r\d+_c\d+', h)]
    dr_vals = list(range(len({int(re.search(r'px_r(\d+)_c', h).group(1)) for h in px_cols})))
    dc_vals = list(range(len({int(re.search(r'px_r\d+_c(\d+)', h).group(1)) for h in px_cols})))
    ts_col = 'timestamp'
    has_elapsed = False

GRID_H = len(dr_vals)
GRID_W = len(dc_vals)
DR_MIN, DR_MAX = dr_vals[0], dr_vals[-1]
DC_MIN, DC_MAX = dc_vals[0], dc_vals[-1]
dr_axis = np.array(dr_vals)
dc_axis = np.array(dc_vals)

file_size = os.path.getsize(CSV_PATH)
print(f'  Parsing {file_size / 1e9:.1f} GB CSV...')
needed_cols = [ts_col] + px_cols

chunk_ts = []
chunk_frames = []
for chunk in pd.read_csv(
    CSV_PATH,
    sep=',',
    decimal='.',
    encoding='utf-8-sig',
    usecols=needed_cols,
    chunksize=100_000,
):
    valid = chunk[px_cols[0]].notna()
    if not valid.all():
        chunk = chunk.loc[valid]
    if chunk.empty:
        continue
    chunk_ts.append(chunk[ts_col].values.astype(np.float64))
    chunk_frames.append(chunk[px_cols].values.astype(np.float32).reshape(-1, GRID_H, GRID_W))

timestamps = np.concatenate(chunk_ts)
frames = np.concatenate(chunk_frames)

if not has_elapsed:
    timestamps -= timestamps[0]

N = len(frames)
print(f'  {N} frames, {GRID_H}x{GRID_W} grid, duration {timestamps[-1]:.1f} s')

all_vals = frames.ravel()
vmin = TEMP_MIN if TEMP_MIN is not None else float(np.nanpercentile(all_vals, 1))
vmax = TEMP_MAX if TEMP_MAX is not None else float(np.nanpercentile(all_vals, 99))
del all_vals


def _odd_window_from_seconds(seconds, dt, n):
    if not np.isfinite(dt) or dt <= 0:
        return 1
    window = int(round(seconds / dt))
    window = max(1, min(window, n))
    if window % 2 == 0:
        window += 1 if window < n else -1
    return max(1, window)


def _smooth_signal(values, window):
    if window <= 1 or len(values) < 3:
        return values.astype(np.float64, copy=True)
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values).astype(np.float64)
    filled = np.where(np.isfinite(values), values, 0.0)
    numer = np.convolve(filled, kernel, mode='same')
    denom = np.convolve(finite, kernel, mode='same')
    return numer / np.maximum(denom, 1.0)


def _find_peaks(values, threshold, min_gap):
    if len(values) < 3:
        return np.array([], dtype=int)
    candidates = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] >= threshold)
    ) + 1
    if len(candidates) == 0:
        return np.array([], dtype=int)

    peaks = []
    for idx in candidates:
        if not peaks:
            peaks.append(int(idx))
            continue
        if idx - peaks[-1] >= min_gap:
            peaks.append(int(idx))
        elif values[idx] > values[peaks[-1]]:
            peaks[-1] = int(idx)
    return np.array(peaks, dtype=int)


def _find_pass_minima(values, peak_idx):
    if len(values) == 0 or len(peak_idx) < 2:
        return {
            'min_idx': np.array([], dtype=int),
            'min_value': np.array([], dtype=np.float64),
            'layer_number': np.array([], dtype=int),
            'perimeter_pass': np.array([], dtype=int),
        }

    min_idx = []
    min_value = []
    layer_number = []
    perimeter_pass = []

    for pass_i in range(len(peak_idx) - 1):
        start_peak = int(peak_idx[pass_i])
        end_peak = int(peak_idx[pass_i + 1])
        start = start_peak + 1
        end = end_peak
        segment = values[start:end]

        if len(segment) == 0:
            start = start_peak
            end = end_peak + 1
            segment = values[start:end]

        finite = np.isfinite(segment)
        if not np.any(finite):
            continue

        local_idx = int(np.nanargmin(segment))
        idx = start + local_idx
        min_idx.append(idx)
        min_value.append(float(values[idx]))
        pass_sequence = pass_i + 1
        layer_number.append(pass_sequence // 2 + 1)
        perimeter_pass.append(pass_sequence % 2 + 1)

    return {
        'min_idx': np.array(min_idx, dtype=int),
        'min_value': np.array(min_value, dtype=np.float64),
        'layer_number': np.array(layer_number, dtype=int),
        'perimeter_pass': np.array(perimeter_pass, dtype=int),
    }


def build_layer_model(frames, timestamps):
    flat = frames.reshape(len(frames), -1)
    if LAYER_SIGNAL_MODE == 'max':
        raw_signal = np.nanmax(flat, axis=1).astype(np.float64)
    else:
        raise ValueError(f'Unsupported LAYER_SIGNAL_MODE: {LAYER_SIGNAL_MODE}')

    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt = float(np.median(diffs)) if diffs.size else 1.0

    smooth_window = _odd_window_from_seconds(LAYER_SMOOTH_SECONDS, dt, len(raw_signal))
    smooth_signal = _smooth_signal(raw_signal, smooth_window)

    baseline = float(np.nanpercentile(smooth_signal, LAYER_BASELINE_PERCENTILE))
    active = float(np.nanpercentile(smooth_signal, LAYER_ACTIVE_PERCENTILE))
    span = max(active - baseline, 1e-6)
    threshold = baseline + LAYER_THRESHOLD_FRACTION * span
    min_gap = max(1, int(round(LAYER_MIN_PEAK_GAP_SECONDS / max(dt, 1e-6))))
    peak_idx = _find_peaks(smooth_signal, threshold, min_gap)
    pass_minima = _find_pass_minima(raw_signal, peak_idx)

    peak_counts = np.searchsorted(peak_idx, np.arange(len(raw_signal)), side='right')
    layer_number = peak_counts // 2 + 1
    perimeter_pass = peak_counts % 2 + 1
    peak_sequence = np.arange(len(peak_idx), dtype=int)
    peak_layer_number = peak_sequence // 2 + 1
    peak_perimeter_pass = peak_sequence % 2 + 1

    return {
        'raw_signal': raw_signal,
        'smooth_signal': smooth_signal,
        'peak_idx': peak_idx,
        'peak_value': smooth_signal[peak_idx],
        'peak_layer_number': peak_layer_number.astype(int),
        'peak_perimeter_pass': peak_perimeter_pass.astype(int),
        'min_idx': pass_minima['min_idx'],
        'min_value': pass_minima['min_value'],
        'min_layer_number': pass_minima['layer_number'],
        'min_perimeter_pass': pass_minima['perimeter_pass'],
        'layer_number': layer_number.astype(int),
        'perimeter_pass': perimeter_pass.astype(int),
        'threshold': threshold,
        'smooth_window': smooth_window,
    }


def axes_to_grid(ax_x, ax_y):
    col = int(round((ax_x - DC_MIN) / max(DC_MAX - DC_MIN, 1) * (GRID_W - 1)))
    row = int(round((ax_y - DR_MIN) / max(DR_MAX - DR_MIN, 1) * (GRID_H - 1)))
    return np.clip(col, 0, GRID_W - 1), np.clip(row, 0, GRID_H - 1)


layer_model = build_layer_model(frames, timestamps)

fig = plt.figure(figsize=(16, 9), facecolor='#ffffff')
fig.canvas.manager.set_window_title(f'Thermal Viewer - {os.path.basename(CSV_PATH)}')

gs = gridspec.GridSpec(
    2,
    2,
    width_ratios=[1.1, 1.25],
    height_ratios=[1, 0.06],
    hspace=0.30,
    wspace=0.32,
    left=0.07,
    right=0.97,
    top=0.94,
    bottom=0.06,
)

ax_heat = fig.add_subplot(gs[0, 0])
ax_layer = fig.add_subplot(gs[0, 1])
ax_slider = fig.add_subplot(gs[1, :])

for ax in (ax_heat, ax_layer):
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#333333', labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')

im = ax_heat.imshow(
    frames[0],
    cmap=CMAP,
    vmin=vmin,
    vmax=vmax,
    interpolation=INTERP,
    aspect='equal',
    origin='upper',
    extent=[DC_MIN - 0.5, DC_MAX + 0.5, DR_MAX + 0.5, DR_MIN - 0.5],
)
cb = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
cb.set_label('degC', color='#333333', fontsize=14)
cb.ax.yaxis.set_tick_params(color='#333333', labelcolor='#333333')
ax_heat.set_xlabel('dc  (left  |  right)', color='#555555', fontsize=14)
ax_heat.set_ylabel('dr  (up above  |  below down)', color='#555555', fontsize=14)
ax_heat.axhline(0, color='#cc9900', lw=0.7, linestyle='--', alpha=0.5)
ax_heat.axvline(0, color='#cc9900', lw=0.7, linestyle='--', alpha=0.5)
ax_heat.set_title('Thermal heatmap', color='#333333', fontsize=14, pad=5)
ax_heat.set_xticks(np.arange(DC_MIN - 0.5, DC_MAX + 1.0, 1.0), minor=True)
ax_heat.set_yticks(np.arange(DR_MIN - 0.5, DR_MAX + 1.0, 1.0), minor=True)
ax_heat.grid(which='minor', color='#ffffff', linestyle='-', linewidth=0.35, alpha=0.8)
ax_heat.tick_params(which='minor', bottom=False, left=False)

hcross_h = ax_heat.axhline(0, color='#0077cc', lw=0.8, linestyle='-', alpha=0.7, visible=False)
hcross_v = ax_heat.axvline(0, color='#0077cc', lw=0.8, linestyle='-', alpha=0.7, visible=False)

hover_text = ax_heat.text(
    0.02,
    0.02,
    '',
    transform=ax_heat.transAxes,
    color='#0077cc',
    fontsize=14,
    va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='#0077cc', alpha=0.85),
)
frame_text = ax_heat.text(
    0.02,
    0.97,
    '',
    transform=ax_heat.transAxes,
    color='#333333',
    fontsize=14,
    va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='#cccccc', alpha=0.85),
)
pixel_count_text = ax_heat.text(
    0.02,
    0.89,
    '',
    transform=ax_heat.transAxes,
    color='#333333',
    fontsize=11,
    va='top',
    bbox=dict(boxstyle='round,pad=0.25', fc='#ffffff', ec='#cccccc', alpha=0.85),
)
layer_text = ax_heat.text(
    0.98,
    0.97,
    '',
    transform=ax_heat.transAxes,
    color='#7a3d00',
    fontsize=13,
    va='top',
    ha='right',
    bbox=dict(boxstyle='round,pad=0.3', fc='#fff4d6', ec='#cc9900', alpha=0.92),
)

ax_layer.plot(
    timestamps,
    layer_model['raw_signal'],
    color='#f4b266',
    lw=1.0,
    alpha=0.45,
    label='Max-pixel signal',
)
ax_layer.plot(
    timestamps,
    layer_model['smooth_signal'],
    color='#c55a11',
    lw=1.8,
    label='Smoothed signal',
)
if len(layer_model['peak_idx']) > 0:
    ax_layer.scatter(
        timestamps[layer_model['peak_idx']],
        layer_model['smooth_signal'][layer_model['peak_idx']],
        color='#cc0000',
        s=28,
        zorder=5,
        label='Detected perimeter peaks',
    )
if len(layer_model['min_idx']) > 0:
    ax_layer.scatter(
        timestamps[layer_model['min_idx']],
        layer_model['raw_signal'][layer_model['min_idx']],
        color='#1f77ff',
        s=26,
        zorder=5,
        label='Detected pass minima',
    )

ax_layer.axhline(layer_model['threshold'], color='#999999', lw=1.0, linestyle='--', alpha=0.8)
layer_cursor_line = ax_layer.axvline(timestamps[0], color='#0066aa', lw=1.2, linestyle='-', alpha=0.9)
zoom_start_line = ax_layer.axvline(timestamps[0], color='#2ca02c', lw=1.4, linestyle='--', alpha=0.95, visible=False)
zoom_end_line = ax_layer.axvline(timestamps[0], color='#d62728', lw=1.4, linestyle='--', alpha=0.95, visible=False)
layer_status_text = ax_layer.text(
    0.02,
    0.02,
    '',
    transform=ax_layer.transAxes,
    color='#7a3d00',
    fontsize=11,
    va='bottom',
    bbox=dict(boxstyle='round,pad=0.25', fc='#fffaf0', ec='#d7b98d', alpha=0.9),
)
total_layers = int(layer_model['layer_number'][-1]) if N else 0
ax_layer.set_title(
    f'Corner activity / estimated layers  ({len(layer_model["min_idx"])} minima, ~{total_layers} layers)',
    color='#333333',
    fontsize=14,
    pad=4,
)
ax_layer.set_xlabel('Time [s]', color='#555555', fontsize=12)
ax_layer.set_ylabel('Temperature [C]', color='#555555', fontsize=12)
ax_layer.set_xlim(float(timestamps[0]), float(timestamps[-1]) if N > 1 else float(timestamps[0]) + 1.0)
ax_layer.legend(facecolor='#ffffff', labelcolor='#333333', fontsize=10, loc='lower right')

ax_slider.set_facecolor('#f0f0f0')
slider = Slider(ax_slider, 'Frame', 0, N - 1, valinit=0, valstep=1, color='#2299aa', initcolor='none')
slider.label.set_color('#333333')
slider.valtext.set_color('#333333')

fig.suptitle(os.path.basename(CSV_PATH), color='#333333', fontsize=14, y=0.99)

initial_limits = {
    ax_heat: (ax_heat.get_xlim(), ax_heat.get_ylim()),
    ax_layer: (ax_layer.get_xlim(), ax_layer.get_ylim()),
}

state = {
    'fi': 0,
    'hover_col': GRID_W // 2,
    'hover_row': GRID_H // 2,
    'playing': False,
    'timer': None,
    'updating': False,
    'pan': None,
    'zoom_selecting': False,
    'zoom_t0': None,
    'zoom_t1': None,
}


def update_frame(fi):
    fi = int(np.clip(fi, 0, N - 1))
    state['fi'] = fi
    grid = frames[fi]
    im.set_data(grid)
    frame_text.set_text(f'Frame {fi + 1}/{N}  |  t = {timestamps[fi]:.1f} s')
    pixel_count_text.set_text(f'Pixels shown: {GRID_W} x {GRID_H} = {GRID_W * GRID_H}')
    layer_no = int(layer_model['layer_number'][fi])
    pass_no = int(layer_model['perimeter_pass'][fi])
    layer_text.set_text(f'Estimated layer {layer_no}\nPerimeter pass {pass_no}/2')
    layer_cursor_line.set_xdata([timestamps[fi], timestamps[fi]])
    layer_status_text.set_text(
        f'Layer {layer_no}  |  pass {pass_no}/2\n'
        f'Smoothing window: {layer_model["smooth_window"]} frames'
    )
    refresh_hover(grid)
    refresh_zoom_markers()
    state['updating'] = True
    slider.set_val(fi)
    state['updating'] = False
    fig.canvas.draw_idle()


def refresh_hover(grid):
    col = state['hover_col']
    row = state['hover_row']
    temp = grid[row, col]
    ax_x = dc_axis[col] if col < len(dc_axis) else col
    ax_y = dr_axis[row] if row < len(dr_axis) else row
    hcross_h.set_ydata([ax_y, ax_y])
    hcross_h.set_visible(True)
    hcross_v.set_xdata([ax_x, ax_x])
    hcross_v.set_visible(True)
    hover_text.set_text(f'dc={dc_vals[col]:+d}  dr={dr_vals[row]:+d}\n{temp:.1f} degC')


def start_pan(event):
    if event.button != 1 or event.inaxes not in (ax_heat, ax_layer):
        return False
    if event.xdata is None or event.ydata is None:
        return False
    state['pan'] = {
        'ax': event.inaxes,
        'press_x': event.xdata,
        'press_y': event.ydata,
        'xlim': event.inaxes.get_xlim(),
        'ylim': event.inaxes.get_ylim(),
        'moved': False,
    }
    return True


def drag_pan(event):
    pan = state['pan']
    if pan is None or event.inaxes != pan['ax']:
        return
    if event.xdata is None or event.ydata is None:
        return

    dx = event.xdata - pan['press_x']
    dy = event.ydata - pan['press_y']
    if abs(dx) > 0 or abs(dy) > 0:
        pan['moved'] = True

    ax = pan['ax']
    x0, x1 = pan['xlim']
    y0, y1 = pan['ylim']
    ax.set_xlim(x0 - dx, x1 - dx)
    ax.set_ylim(y0 - dy, y1 - dy)
    fig.canvas.draw_idle()


def stop_pan():
    pan = state['pan']
    state['pan'] = None
    return pan


def reset_axis_view(ax):
    xlim, ylim = initial_limits[ax]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if ax == ax_layer:
        clear_zoom_selection()
    fig.canvas.draw_idle()


def refresh_zoom_markers():
    t0 = state['zoom_t0']
    t1 = state['zoom_t1']
    zoom_start_line.set_visible(t0 is not None)
    zoom_end_line.set_visible(t1 is not None)
    if t0 is not None:
        zoom_start_line.set_xdata([t0, t0])
    if t1 is not None:
        zoom_end_line.set_xdata([t1, t1])


def clear_zoom_selection():
    state['zoom_selecting'] = False
    state['zoom_t0'] = None
    state['zoom_t1'] = None
    refresh_zoom_markers()


def zoom_layer_to_selection():
    t0 = state['zoom_t0']
    t1 = state['zoom_t1']
    if t0 is None or t1 is None:
        return

    t_start = min(t0, t1)
    t_end = max(t0, t1)
    if np.isclose(t_start, t_end):
        return

    mask = (timestamps >= t_start) & (timestamps <= t_end)
    if not np.any(mask):
        return

    y_candidates = np.concatenate(
        [
            layer_model['raw_signal'][mask],
            layer_model['smooth_signal'][mask],
            np.array([layer_model['threshold']], dtype=np.float64),
        ]
    )
    finite = y_candidates[np.isfinite(y_candidates)]
    if finite.size == 0:
        return

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_span = max(y_max - y_min, 1.0)
    bottom_pad = max(y_span * 0.28, 1.5)
    top_pad = max(y_span * 0.08, 0.5)
    ax_layer.set_xlim(t_start, t_end)
    ax_layer.set_ylim(y_min - bottom_pad, y_max + top_pad)
    state['zoom_selecting'] = False
    fig.canvas.draw_idle()


def handle_layer_zoom_click(event):
    if event.inaxes != ax_layer or event.xdata is None:
        return False

    click_time = float(np.clip(event.xdata, timestamps[0], timestamps[-1]))
    if state['zoom_t0'] is None or not state['zoom_selecting']:
        state['zoom_t0'] = click_time
        state['zoom_t1'] = None
        state['zoom_selecting'] = True
    else:
        state['zoom_t1'] = click_time
        zoom_layer_to_selection()

    refresh_zoom_markers()
    fig.canvas.draw_idle()
    return True


def on_motion(event):
    if state['pan'] is not None:
        drag_pan(event)
        return

    if event.inaxes != ax_heat:
        return
    if event.xdata is None or event.ydata is None:
        return

    col, row = axes_to_grid(event.xdata, event.ydata)
    state['hover_col'] = col
    state['hover_row'] = row
    refresh_hover(frames[state['fi']])
    fig.canvas.draw_idle()


def on_press(event):
    if event.dblclick and event.inaxes in (ax_heat, ax_layer):
        reset_axis_view(event.inaxes)
        return
    if event.inaxes == ax_layer and event.button == 3:
        reset_axis_view(ax_layer)
        return
    if event.inaxes == ax_layer and event.button == 1 and event.key == 'shift':
        handle_layer_zoom_click(event)
        return
    start_pan(event)


def on_release(event):
    pan = stop_pan()
    if pan is None:
        return
    if pan['ax'] == ax_layer and not pan['moved'] and event is not None and event.xdata is not None:
        stop_play()
        fi = int(np.clip(np.searchsorted(timestamps, event.xdata), 0, N - 1))
        update_frame(fi)


def on_key(event):
    if event.key in ('q', 'escape'):
        plt.close(fig)
    elif event.key == ' ':
        toggle_play()
    elif event.key == 'right':
        stop_play()
        update_frame(state['fi'] + 1)
    elif event.key == 'left':
        stop_play()
        update_frame(state['fi'] - 1)
    elif event.key == 'a':
        save_pass_minima_csv()
    elif event.key == 'b':
        save_peaks_and_minima_csv()


def on_scroll(event):
    if event.inaxes == ax_heat:
        delta = -1 if event.button == 'up' else 1
        stop_play()
        update_frame(state['fi'] + delta)


def on_slider(val):
    if not state['playing'] and not state['updating']:
        update_frame(int(val))


def toggle_play():
    if state['playing']:
        stop_play()
    else:
        state['playing'] = True
        schedule_next_frame()


def stop_play():
    state['playing'] = False
    if state['timer'] is not None:
        state['timer'].stop()
        state['timer'] = None


def schedule_next_frame():
    if not state['playing']:
        return
    fi = state['fi'] + 1
    if fi >= N:
        stop_play()
        return
    update_frame(fi)
    interval_ms = int(1000.0 / PLAYBACK_FPS)
    state['timer'] = fig.canvas.new_timer(interval=interval_ms)
    state['timer'].add_callback(_tick)
    state['timer'].single_shot = True
    state['timer'].start()


def _tick():
    schedule_next_frame()


def save_pass_minima_csv():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_csv = filedialog.askopenfilename(
        title='Select selected-pass CSV',
        filetypes=[('CSV files', '*.csv')],
        initialdir=FILE_PICKER_INITIALDIR,
    )
    root.destroy()

    if not selected_csv:
        print('No selected-pass CSV chosen.')
        return

    df = pd.read_csv(selected_csv)

    if 'layer_number' in df.columns and 'pass_number' in df.columns:
        layer_col = 'layer_number'
        pass_col = 'pass_number'
    elif len(df.columns) >= 12:
        layer_col = df.columns[10]
        pass_col = df.columns[11]
    else:
        print('Chosen CSV does not contain layer/pass columns.')
        return

    minima_map = {
        (int(layer_no), int(pass_no)): float(min_temp)
        for layer_no, pass_no, min_temp in zip(
            layer_model['min_layer_number'],
            layer_model['min_perimeter_pass'],
            layer_model['min_value'],
        )
    }

    s_temp_values = []
    for _, row in df.iterrows():
        if pd.isna(row[layer_col]) or pd.isna(row[pass_col]):
            s_temp_values.append(np.nan)
            continue
        key = (int(row[layer_col]), int(row[pass_col]))
        s_temp_values.append(minima_map.get(key, np.nan))

    if 's_temp' in df.columns:
        df = df.drop(columns=['s_temp'])

    df.insert(min(3, len(df.columns)), 's_temp', s_temp_values)
    output_csv = os.path.join(os.path.dirname(selected_csv), 'calibration_roffset.csv')
    df.to_csv(output_csv, index=False)
    print(f'Updated selected-pass CSV -> {output_csv}')


def save_peaks_and_minima_csv():
    peak_df = pd.DataFrame(
        {
            'event_type': 'peak',
            'layer_number_pass_number': [
                f'{int(layer_no)}_{int(pass_no)}'
                for layer_no, pass_no in zip(
                    layer_model['peak_layer_number'],
                    layer_model['peak_perimeter_pass'],
                )
            ],
            'event_value_c': np.round(layer_model['peak_value'], 3),
        }
    )

    minima_df = pd.DataFrame(
        {
            'event_type': 'minima',
            'layer_number_pass_number': [
                f'{int(layer_no)}_{int(pass_no)}'
                for layer_no, pass_no in zip(
                    layer_model['min_layer_number'],
                    layer_model['min_perimeter_pass'],
                )
            ],
            'event_value_c': np.round(layer_model['min_value'], 3),
        }
    )

    base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
    output_csv = os.path.join(os.path.dirname(CSV_PATH), f'{base_name}_peaks_minima.csv')
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        peak_df.to_csv(f, index=False, float_format='%.3f')
        f.write('\n')
        minima_df.to_csv(f, index=False, float_format='%.3f')
    print(f'Saved peaks/minima CSV -> {output_csv}')


fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('scroll_event', on_scroll)
slider.on_changed(on_slider)

update_frame(0)
plt.show()
