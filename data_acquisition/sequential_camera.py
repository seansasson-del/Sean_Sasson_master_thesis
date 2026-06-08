import csv
import datetime
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

# --- Calibration ---
CALIB_OFFSET = 0
CALIB_SCALE = 1
STANDARD_EMISSIVITY = 0.92
CENTER_POINT_EMISSIVITY = 0.5
frequency = 25
INITIAL_CENTER_POINT = (122, 109)
PIXEL_POINTS = [
    (127, 109),
    (127, 110),
    (126, 111),
    (125, 112),
    (124, 112),
    (123, 112),
    (122, 112),
    (121, 112),
    (120, 112),
    (119, 112),
    (118, 111),
    (117, 110),
    (117, 109),
]

center_history = deque(maxlen=20000)
TOTAL_SUBPRINTS = 5


class IRCameraLogger:
    def __init__(self, session_folder, run_index=1, device=0, backend=cv2.CAP_MSMF, scale=3, colormap=cv2.COLORMAP_INFERNO):
        self.session_folder = session_folder
        if not 1 <= run_index <= TOTAL_SUBPRINTS:
            raise ValueError(f"run_index must be between 1 and {TOTAL_SUBPRINTS}, got {run_index}")
        self.run_index = run_index
        self.csv_filename = os.path.join(session_folder, "camera_data.csv")

        self.device = device
        self.backend = backend
        self.scale = scale
        self.colormap = colormap

        self.csv_rows = []
        self.recording = False
        self.record_video = False
        self.video_writer = None
        self.video_filename = None
        self.video_fps = 25.0
        self.video_start_time = None
        self.csv_interval_ms = int(1000 / frequency)
        self.emissivity = STANDARD_EMISSIVITY
        self.center_point_emissivity = CENTER_POINT_EMISSIVITY
        self.center_position = INITIAL_CENTER_POINT
        self.pixel_points = list(PIXEL_POINTS)
        self.measured_point_count = len(self.pixel_points)
        self.last_csv_time = 0
        self.stop_flag = False
        self.abort_requested = False
        self.manual_encoder_init_requested = False
        self.encoder_initialized = False
        self.manual_finalize_requested = False
        self.force_abort = False
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_center_temp = None
        self.window_name = "TC001"
        self.window_created = False

        self.thread = threading.Thread(target=self._run)

    def _subprint_label(self):
        return f"subprint_{self.run_index:02d}"

    # -----------------------
    # Public control methods
    # -----------------------

    def start(self):
        self.thread.start()

    def start_recording(self):
        if not self.recording:
            self.csv_rows.clear()
            center_history.clear()
            self.last_csv_time = 0
            self.recording = True
            print("IR CSV recording started.")

    def stop_recording(self):
        if self.recording:
            self._save_csv()
            self.recording = False
            print("IR CSV recording stopped.")

    def start_video(self):
        if not self.record_video:
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = os.path.join(
                self.session_folder,
                f"tc001_{self._subprint_label()}_{ts_str}.avi",
            )
            self.video_writer = None
            self.record_video = True
            self.video_start_time = time.time()
            print(f"Video recording started: {self.video_filename}")

    def stop_video(self):
        if self.record_video:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_video = False
            self.video_start_time = None
            print("Video saved successfully.")

    def shutdown(self):
        print("Shutting down IR camera...")
        self.request_shutdown()

        if self.record_video:
            self.stop_video()

        if self.thread.is_alive():
            self.thread.join()

        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False

    def request_shutdown(self):
        """Ask the camera thread to exit without marking the run as aborted."""
        self.stop_flag = True

    def request_encoder_init(self):
        if self.encoder_initialized:
            print("Encoder is already initialized.")
            return

        if not self.manual_encoder_init_requested:
            self.manual_encoder_init_requested = True
            print("Manual encoder initialization requested (I).")

    def mark_encoder_initialized(self):
        self.encoder_initialized = True
        self.manual_encoder_init_requested = False
        print("Encoder initialization confirmed.")

    def request_manual_finalize(self):
        """
        Signal the controller to finalize the last run and stop the camera UI
        immediately so the hardware state matches a normal final shutdown.
        """
        self.manual_finalize_requested = True
        self.request_shutdown()

    def process_ui_events(self):
        frame_to_show = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_to_show = self.latest_frame.copy()

        if frame_to_show is None:
            return

        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True

        cv2.resizeWindow(self.window_name, frame_to_show.shape[1], frame_to_show.shape[0])
        cv2.imshow(self.window_name, frame_to_show)
        self._handle_keypress(cv2.waitKey(1) & 0xFF)

    def _handle_keypress(self, key):
        if key == 255:
            return

        if key == ord("c"):
            print("Enter reference temperature:")
            ref_temp = float(input())
            global CALIB_OFFSET
            raw_center_temp = getattr(self, "latest_center_temp", None)
            if raw_center_temp is None:
                print("Calibration skipped: no temperature frame available yet.")
                return
            raw_center_temp = (raw_center_temp - CALIB_OFFSET) / CALIB_SCALE
            CALIB_OFFSET = ref_temp - (raw_center_temp * CALIB_SCALE)
            print(f"Calibration applied: offset = {CALIB_OFFSET:.2f}")
        elif key == ord("i"):
            self.request_encoder_init()
        elif key == ord("v") and not self.record_video:
            self.start_video()
        elif key == ord("s") and self.record_video:
            self.stop_video()
        elif key == ord("f"):
            print("Manual finalization requested (F).")
            self.request_manual_finalize()
        elif key == ord("q"):
            print("Emergency stop requested (Q).")
            self.abort_requested = True
            self.stop_flag = True
        elif key == 27:
            self.stop_flag = True

    # -----------------------
    # Internal functions
    # -----------------------

    def decode_temperatures(self, thermal: np.ndarray) -> np.ndarray:
        lo = thermal[..., 0].astype(np.int32)
        hi = thermal[..., 1].astype(np.int32)
        raw = lo + (hi << 8)
        temps = (raw / 64.0) - 273.15
        return temps * CALIB_SCALE + CALIB_OFFSET

    def _save_csv(self):
        if self.force_abort:
            print("Abort active, camera CSV not saved.")
            return

        if not self.csv_rows:
            return

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(self.session_folder, f"tc001_{self._subprint_label()}_{ts_str}.csv")

        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            point_headers = [f"p{i}" for i in range(len(self.csv_rows[0]) - 2)]
            writer.writerow(["timestamp", "center_temp"] + point_headers)
            writer.writerows(self.csv_rows)

        self.csv_filename = fname
        print(f"CSV saved: {fname}")
        self.csv_rows.clear()

    def _ensure_video_writer(self, frame_size):
        if self.force_abort or not self.record_video:
            return False

        if self.video_filename is None:
            self.start_video()

        if self.video_writer is None:
            os.makedirs(self.session_folder, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                fourcc,
                self.video_fps,
                frame_size,
            )

            if not self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                self.record_video = False
                raise RuntimeError(f"Unable to open video writer for {self.video_filename}")

        return True

    def draw_plot_center(self, width, height, font_scale=1):
        plot = np.zeros((height, width, 3), dtype=np.uint8)
        plot[:] = (18, 18, 18)

        cv2.rectangle(plot, (0, 0), (width - 1, height - 1), (55, 55, 55), 1)
        cv2.line(plot, (0, height - 28), (width, height - 28), (45, 45, 45), 1)
        cv2.putText(
            plot,
            "Center Temperature Trend",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        if len(center_history) > 1:
            vals = np.array(center_history)
            min_temp = np.nanmin(vals)
            max_temp = np.nanmax(vals)
            temp_span = max(max_temp - min_temp, 1.0)
            usable_height = max(height - 44, 20)
            pts = (((vals - min_temp) / temp_span) * usable_height).astype(int)

            for i in range(1, len(pts)):
                cv2.line(
                    plot,
                    (int((i - 1) / len(vals) * width), height - 28 - pts[i - 1]),
                    (int(i / len(vals) * width), height - 28 - pts[i]),
                    (0, 200, 255),
                    2,
                )

            cv2.putText(
                plot,
                f"{vals[-1]:.1f} C",
                (width - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
            )

        return plot

    def _draw_overlay_panel(self, image, top_left, bottom_right, alpha=0.6, color=(12, 12, 12)):
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, top_left, bottom_right, (80, 80, 80), 1)

    def _format_recording_time(self):
        if not self.record_video or self.video_start_time is None:
            return "00:00"

        elapsed = max(0, int(time.time() - self.video_start_time))
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _run(self):
        global CALIB_OFFSET

        cap = cv2.VideoCapture(self.device, self.backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "2"))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        if not cap.isOpened():
            raise RuntimeError("Unable to open camera TC001")

        cx, cy = self.center_position

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            buf = frame.ravel()
            if buf.size != 384 * 256 * 2:
                continue

            reshaped = buf.reshape(384, 256, 2)
            visual_half, thermal_half = np.array_split(reshaped, 2)

            raw_temps = self.decode_temperatures(thermal_half)
            raw_temps = np.rot90(raw_temps, 2)
            temps = raw_temps / (self.emissivity ** 0.25)
            center_temp = float(raw_temps[cy, cx] / (self.center_point_emissivity ** 0.25))
            self.latest_center_temp = center_temp

            if self.recording:
                now = round(time.time() * 1000)
                if now - self.last_csv_time >= self.csv_interval_ms:
                    ts = round(time.time() * 1000)
                    point_temps = []

                    for px, py in self.pixel_points:
                        if 0 <= py < temps.shape[0] and 0 <= px < temps.shape[1]:
                            point_temps.append(float(temps[int(py), int(px)]))
                        else:
                            point_temps.append(np.nan)

                    if any(np.isfinite(point_temps)):
                        formatted_row = [ts, f"{center_temp:.3f}"] + [
                            f"{t:.3f}" if np.isfinite(t) else "" for t in point_temps
                        ]
                        self.csv_rows.append(formatted_row)
                        self.last_csv_time = now

            visual_bgr = cv2.cvtColor(visual_half, cv2.COLOR_YUV2BGR_YUYV)
            disp = cv2.resize(
                visual_bgr,
                (visual_bgr.shape[1] * self.scale, visual_bgr.shape[0] * self.scale),
            )

            disp_color = cv2.applyColorMap(disp, self.colormap)
            disp_color = cv2.rotate(disp_color, cv2.ROTATE_180)

            height, width = disp_color.shape[:2]
            scale_x = width / temps.shape[1]
            scale_y = height / temps.shape[0]

            for idx, (x, y) in enumerate(self.pixel_points):
                xs, ys = int(x * scale_x), int(y * scale_y)
                color = (0, 255, 0) if idx == 0 else (100, 255, 255)
                cv2.circle(disp_color, (xs, ys), 4, color, -1)

            cx_scaled = int(cx * scale_x)
            cy_scaled = int(cy * scale_y)
            overlay = disp_color.copy()
            cv2.circle(overlay, (cx_scaled, cy_scaled), 10, (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.18, disp_color, 0.82, 0, disp_color)
            cv2.circle(disp_color, (cx_scaled, cy_scaled), 6, (0, 0, 255), -1)

            self._draw_overlay_panel(disp_color, (12, 12), (360, 164), alpha=0.58)
            self._draw_overlay_panel(disp_color, (12, height - 40), (500, height - 10), alpha=0.52)
            self._draw_overlay_panel(disp_color, (width - 150, 12), (width - 12, 54), alpha=0.5)

            cv2.putText(disp_color, f"Center Temp  {center_temp:.1f} C", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 215, 255), 1)
            cv2.putText(disp_color, f"Center Pos   ({cx}, {cy})", (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
            cv2.putText(disp_color, f"Measured pixels  {self.measured_point_count}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
            cv2.putText(disp_color, "Pixel list    absolute coords", (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
            encoder_status = "READY" if self.encoder_initialized else "WAITING FOR I"
            cv2.putText(disp_color, f"Encoder      {encoder_status}", (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
            cv2.putText(disp_color, f"Subprint     {self.run_index}/{TOTAL_SUBPRINTS}", (24, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
            cv2.putText(
                disp_color,
                "Controls: I init encoder   C calibrate   V start video   S stop video   F finalize   Q quit",
                (24, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (220, 220, 220),
                1,
            )

            indicator_color = (0, 0, 255) if self.record_video else (70, 70, 120)
            cv2.circle(disp_color, (width - 116, 33), 7, indicator_color, -1)
            status_label = "REC" if self.record_video else "IDLE"
            status_color = (245, 245, 245) if self.record_video else (210, 210, 210)
            cv2.putText(disp_color, status_label, (width - 102, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)
            cv2.putText(
                disp_color,
                self._format_recording_time(),
                (width - 102, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (220, 220, 220),
                1,
            )

            center_history.append(center_temp)
            plot_center = self.draw_plot_center(width, height // 4)
            combined = np.vstack([disp_color, plot_center])
            with self.frame_lock:
                self.latest_frame = combined.copy()

            if self.record_video:
                frame_size = (combined.shape[1], combined.shape[0])
                if self._ensure_video_writer(frame_size):
                    self.video_writer.write(combined)

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
