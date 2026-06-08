import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --- Calibration ---
CALIB_OFFSET = 0
CALIB_SCALE = 1
emissivity = 0.95
device_id = 0

# Rectangle on the rotated thermal/image grid: (x, y, width, height)
ZOOM_RECT = (111, 97, 24, 24)
MAIN_SCALE = 3
ZOOM_SCALE = 16
WINDOW_NAME = "Center Line Temperature Tool"
LINE_LENGTH = 7
CSV_OUTPUT_DIR = Path(".")

GUIDE_DIRECTIONS = [
    ("Right", (1, 0)),
    ("Left", (-1, 0)),
    ("Down", (0, 1)),
    ("45 deg", (1, 1)),
    ("135 deg", (-1, 1)),
]

GUIDE_COLORS = {
    "Right": (0, 140, 255),
    "Left": (255, 220, 0),
    "Down": (0, 200, 0),
    "45 deg": (255, 70, 70),
    "135 deg": (180, 0, 255),
}

ARROW_KEY_DELTAS = {
    2424832: (-1, 0),  # Left arrow
    2490368: (0, -1),  # Up arrow
    2555904: (1, 0),   # Right arrow
    2621440: (0, 1),   # Down arrow
}


def decode_temperatures(thermal: np.ndarray) -> np.ndarray:
    lo = thermal[..., 0].astype(np.int32)
    hi = thermal[..., 1].astype(np.int32)
    raw = lo + (hi << 8)
    temps = (raw / 64.0) - 273.15
    return temps * CALIB_SCALE + CALIB_OFFSET


def clamp_rect(image_shape, rect):
    height, width = image_shape[:2]
    x, y, w, h = rect

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def sample_guide_lines(temps: np.ndarray, center_point, line_length: int):
    cx, cy = center_point
    height, width = temps.shape[:2]
    profiles = []

    for label, (dx, dy) in GUIDE_DIRECTIONS:
        samples = []
        for step in range(1, line_length + 1):
            x = cx + dx * step
            y = cy + dy * step
            if 0 <= x < width and 0 <= y < height:
                samples.append(
                    {
                        "step": step,
                        "x": int(x),
                        "y": int(y),
                        "temp": float(temps[y, x]),
                    }
                )
        profiles.append({"label": label, "color": GUIDE_COLORS[label], "samples": samples})

    return profiles


def draw_guide_points(image: np.ndarray, profiles, scale_x: float, scale_y: float):
    for profile in profiles:
        color = profile["color"]
        for sample in profile["samples"]:
            x = int(round((sample["x"] + 0.5) * scale_x))
            y = int(round((sample["y"] + 0.5) * scale_y))
            cv2.circle(image, (x, y), max(2, int(round(min(scale_x, scale_y) / 3))), color, -1)


def build_zoom_view(
    image: np.ndarray,
    rect,
    zoom_scale: int,
    center_point=None,
    profiles=None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = clamp_rect(image.shape, rect)
    crop = image[y : y + h, x : x + w]
    zoomed = cv2.resize(
        crop,
        (w * zoom_scale, h * zoom_scale),
        interpolation=cv2.INTER_NEAREST,
    )

    if profiles is not None:
        for profile in profiles:
            color = profile["color"]
            for sample in profile["samples"]:
                x_pt = sample["x"]
                y_pt = sample["y"]
                if x <= x_pt < x + w and y <= y_pt < y + h:
                    px = int((x_pt - x + 0.5) * zoom_scale)
                    py = int((y_pt - y + 0.5) * zoom_scale)
                    cv2.circle(zoomed, (px, py), max(3, zoom_scale // 4), color, -1)

    if center_point is not None:
        cx, cy = center_point
        if x <= cx < x + w and y <= cy < y + h:
            px = int((cx - x + 0.5) * zoom_scale)
            py = int((cy - y + 0.5) * zoom_scale)
            cv2.circle(zoomed, (px, py), max(4, zoom_scale // 3), (0, 0, 255), -1)

    for px in range(0, zoomed.shape[1] + 1, zoom_scale):
        cv2.line(zoomed, (px, 0), (px, zoomed.shape[0]), (255, 255, 255), 1)
    for py in range(0, zoomed.shape[0] + 1, zoom_scale):
        cv2.line(zoomed, (0, py), (zoomed.shape[1], py), (255, 255, 255), 1)

    cv2.rectangle(zoomed, (0, 0), (zoomed.shape[1] - 1, zoomed.shape[0] - 1), (0, 255, 255), 2)
    return zoomed, (x, y, w, h)


def draw_temperature_plot(profiles, width: int = 560, height: int = 420) -> np.ndarray:
    plot = np.zeros((height, width, 3), dtype=np.uint8)
    margin_left = 70
    margin_right = 20
    margin_top = 105
    margin_bottom = 70

    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top

    cv2.rectangle(plot, (0, 0), (width - 1, height - 1), (35, 35, 35), 1)
    cv2.line(plot, (x0, y0), (x1, y0), (180, 180, 180), 1)
    cv2.line(plot, (x0, y0), (x0, y1), (180, 180, 180), 1)

    all_temps = [sample["temp"] for profile in profiles for sample in profile["samples"] if not np.isnan(sample["temp"])]
    if all_temps:
        min_temp = min(all_temps)
        max_temp = max(all_temps)
    else:
        min_temp = 0.0
        max_temp = 1.0

    if abs(max_temp - min_temp) < 1e-6:
        min_temp -= 0.5
        max_temp += 0.5
    else:
        padding = max(0.5, (max_temp - min_temp) * 0.1)
        min_temp -= padding
        max_temp += padding

    for step in range(1, LINE_LENGTH + 1):
        x = int(round(x0 + (step - 1) * (x1 - x0) / max(1, LINE_LENGTH - 1)))
        cv2.line(plot, (x, y0), (x, y1), (45, 45, 45), 1)
        cv2.putText(plot, str(step), (x - 6, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    for tick in range(5):
        ratio = tick / 4 if 4 else 0
        y = int(round(y0 - ratio * (y0 - y1)))
        temp = min_temp + ratio * (max_temp - min_temp)
        cv2.line(plot, (x0, y), (x1, y), (45, 45, 45), 1)
        cv2.putText(plot, f"{temp:.1f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    for profile in profiles:
        points = []
        for sample in profile["samples"]:
            x = int(round(x0 + (sample["step"] - 1) * (x1 - x0) / max(1, LINE_LENGTH - 1)))
            ratio = (sample["temp"] - min_temp) / (max_temp - min_temp)
            y = int(round(y0 - ratio * (y0 - y1)))
            points.append((x, y))

        if len(points) >= 2:
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(plot, start, end, profile["color"], 2)
        for point in points:
            cv2.circle(plot, point, 4, profile["color"], -1)

    cv2.putText(plot, "Line temperatures", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(plot, "Distance from center [pixels]", (150, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(plot, "Temp [C]", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    legend_x = width - margin_right - 150
    legend_y = 74
    for index, profile in enumerate(profiles):
        y = legend_y + index * 24
        cv2.line(plot, (legend_x, y), (legend_x + 24, y), profile["color"], 2)
        cv2.circle(plot, (legend_x + 12, y), 4, profile["color"], -1)
        cv2.putText(plot, profile["label"], (legend_x + 34, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return plot


def move_center_by_arrow_key(key_code: int, center_point, image_shape):
    dx_dy = ARROW_KEY_DELTAS.get(key_code)
    if dx_dy is None:
        return center_point, False

    cx, cy = center_point
    dx, dy = dx_dy
    height, width = image_shape[:2]
    next_cx = int(np.clip(cx + dx, 0, width - 1))
    next_cy = int(np.clip(cy + dy, 0, height - 1))
    return (next_cx, next_cy), (next_cx, next_cy) != (cx, cy)


def compose_canvas(full_view: np.ndarray, zoom_view: np.ndarray, plot_view: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    padding = 20
    label_space = 36
    info_height = 170

    content_height = max(full_view.shape[0], zoom_view.shape[0], plot_view.shape[0])
    canvas_height = content_height + label_space + padding * 2 + info_height
    canvas_width = full_view.shape[1] + zoom_view.shape[1] + plot_view.shape[1] + padding * 4
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    full_x = padding
    zoom_x = full_x + full_view.shape[1] + padding
    plot_x = zoom_x + zoom_view.shape[1] + padding
    frame_y = padding + label_space

    canvas[frame_y : frame_y + full_view.shape[0], full_x : full_x + full_view.shape[1]] = full_view
    canvas[frame_y : frame_y + zoom_view.shape[0], zoom_x : zoom_x + zoom_view.shape[1]] = zoom_view
    canvas[frame_y : frame_y + plot_view.shape[0], plot_x : plot_x + plot_view.shape[1]] = plot_view

    cv2.putText(canvas, "Full frame", (full_x, padding + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Zoomed rectangle", (zoom_x, padding + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Live plot", (plot_x, padding + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    layout = {
        "full_x": full_x,
        "frame_y": frame_y,
        "full_width": full_view.shape[1],
        "full_height": full_view.shape[0],
    }
    return canvas, layout


def save_profiles_to_csv(center_point, profiles, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cx, cy = center_point
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"center_x{cx}_y{cy}_{timestamp}.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["center_x", "center_y", "direction", "step", "x", "y", "temperature_c"])
        for profile in profiles:
            for sample in profile["samples"]:
                writer.writerow(
                    [
                        cx,
                        cy,
                        profile["label"],
                        sample["step"],
                        sample["x"],
                        sample["y"],
                        f"{sample['temp']:.4f}",
                    ]
                )

    return path


def center_line_temperature_tool(
    device=device_id,
    backend=cv2.CAP_MSMF,
    scale=MAIN_SCALE,
    zoom_scale=ZOOM_SCALE,
    colormap=cv2.COLORMAP_INFERNO,
):
    cap = cv2.VideoCapture(device, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "2"))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cx, cy = 119, 113
    scale_x = 1.0
    scale_y = 1.0
    layout = {"full_x": 0, "frame_y": 0, "full_width": 0, "full_height": 0}
    latest_profiles = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal cx, cy
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        full_x = layout["full_x"]
        frame_y = layout["frame_y"]
        full_width = layout["full_width"]
        full_height = layout["full_height"]

        inside_full = full_x <= x < full_x + full_width and frame_y <= y < frame_y + full_height
        if not inside_full:
            return

        cx = int((x - full_x) / scale_x)
        cy = int((y - frame_y) / scale_y)
        print(f"Center moved to ({cx}, {cy})")

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Click inside the left frame to move center.")
    print("Colored guide points show 7 pixels in the directions right, left, down, 45 deg and 135 deg.")
    print("Use the arrow keys to move the center one pixel at a time.")
    print("Press S to save the current center and the line temperatures to CSV.")
    print("Press ESC to exit.\n")

    last_save_message = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        buf = frame.ravel()
        if buf.size != 384 * 256 * 2:
            continue

        reshaped = buf.reshape(384, 256, 2)
        visual_half, thermal_half = np.array_split(reshaped, 2)

        temps = decode_temperatures(thermal_half)
        temps = temps / (emissivity ** 0.25)
        temps = np.rot90(temps, 2)

        if np.isnan(temps).all():
            continue

        height_t, width_t = temps.shape[:2]
        cx = int(np.clip(cx, 0, width_t - 1))
        cy = int(np.clip(cy, 0, height_t - 1))

        profiles = sample_guide_lines(temps, (cx, cy), LINE_LENGTH)
        latest_profiles = profiles

        visual_bgr = cv2.cvtColor(visual_half, cv2.COLOR_YUV2BGR_YUYV)
        visual_gray = cv2.cvtColor(visual_bgr, cv2.COLOR_BGR2GRAY)
        full_frame = cv2.applyColorMap(visual_gray, colormap)
        full_frame = cv2.rotate(full_frame, cv2.ROTATE_180)

        zoom_view, (rect_x, rect_y, rect_w, rect_h) = build_zoom_view(
            full_frame,
            ZOOM_RECT,
            zoom_scale,
            center_point=(cx, cy),
            profiles=profiles,
        )

        full_view = cv2.resize(
            full_frame,
            (full_frame.shape[1] * scale, full_frame.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        height, width = full_view.shape[:2]
        scale_x = width / temps.shape[1]
        scale_y = height / temps.shape[0]

        cv2.rectangle(
            full_view,
            (rect_x * scale, rect_y * scale),
            ((rect_x + rect_w) * scale - 1, (rect_y + rect_h) * scale - 1),
            (0, 255, 255),
            2,
        )

        draw_guide_points(full_view, profiles, scale_x, scale_y)

        cx_scaled = int(round((cx + 0.5) * scale_x))
        cy_scaled = int(round((cy + 0.5) * scale_y))
        overlay = full_view.copy()
        cv2.circle(overlay, (cx_scaled, cy_scaled), 6, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.45, full_view, 0.55, 0, full_view)

        roi_temps = temps[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w]
        min_temp = float(np.nanmin(roi_temps))
        max_temp = float(np.nanmax(roi_temps))

        plot_view = draw_temperature_plot(profiles)
        canvas, layout = compose_canvas(full_view, zoom_view, plot_view)

        info_top = canvas.shape[0] - 160
        cv2.rectangle(canvas, (10, info_top), (canvas.shape[1] - 10, canvas.shape[0] - 10), (0, 0, 0), -1)

        cv2.putText(canvas, f"Center: ({cx}, {cy})", (20, canvas.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"ROI: ({rect_x}, {rect_y}, {rect_w}, {rect_h})", (20, canvas.shape[0] - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"ROI temp range: {min_temp:.2f} C to {max_temp:.2f} C", (20, canvas.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, "Keys: S = save current center to CSV, ESC = exit", (20, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if last_save_message:
            cv2.putText(canvas, last_save_message, (650, canvas.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKeyEx(1)
        if key == 27:
            break
        (cx, cy), moved = move_center_by_arrow_key(key, (cx, cy), temps.shape)
        if moved:
            continue
        if key in (ord("s"), ord("S")):
            save_path = save_profiles_to_csv((cx, cy), latest_profiles, CSV_OUTPUT_DIR)
            last_save_message = f"Saved: {save_path.name}"
            print(f"Saved center ({cx}, {cy}) temperatures to {save_path.resolve()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    center_line_temperature_tool(device=device_id, backend=cv2.CAP_MSMF)
