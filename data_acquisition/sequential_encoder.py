import csv
import os
import threading
import time

import pandas as pd
import serial

# --- Settings ---
PORT = "COM6"
BAUD = 115200

# --- Build volume limits ---
Xmin, Xmax = 75, 85
Zmin, Zmax = 0.0, 50.8

TOTAL_SUBPRINTS = 5
SUBPRINT_Y_START = 260.0
SUBPRINT_Y_STEP = 60.0
Y_FILTER_HALF_WIDTH = (265.5 - 254.5) / 2.0


class EncoderLogger:
    def __init__(self, session_folder, run_index=1):
        self.session_folder = session_folder
        self.run_index = run_index

        self.output_file = os.path.join(session_folder, "encoder_log_unix.csv")
        self.filtered_file = os.path.join(session_folder, "encoder_log_filtered.csv")
        self.current_position = None
        self.position_lock = threading.Lock()
        self.last_position_update_time = None
        self.logging_active = False
        self.stop_program = False
        self.serial_done = False
        self.force_abort = False
        self.last_terminal_log_time = 0.0
        self.started = False

        self.serial_thread = threading.Thread(target=self.read_serial)

    def get_volume_bounds(self):
        if not 1 <= self.run_index <= TOTAL_SUBPRINTS:
            raise ValueError(
                f"run_index must be between 1 and {TOTAL_SUBPRINTS}, got {self.run_index}"
            )

        target_y = SUBPRINT_Y_START - (self.run_index - 1) * SUBPRINT_Y_STEP
        ymin = target_y - Y_FILTER_HALF_WIDTH
        ymax = target_y + Y_FILTER_HALF_WIDTH
        return Xmin, Xmax, ymin, ymax, Zmin, Zmax

    # ---------------------------
    # Public control functions
    # ---------------------------

    def start(self):
        """Start serial thread (non-blocking)."""
        if self.started:
            return
        self.started = True
        self.serial_thread.start()

    def start_logging(self):
        if not self.logging_active:
            self.logging_active = True
            print("Encoder logging started.")

    def stop_logging(self):
        if self.logging_active:
            self.logging_active = False
            print("Encoder logging stopped.")

    def request_shutdown(self):
        """Stop logging and ask the serial thread to exit."""
        self.logging_active = False
        self.stop_program = True

    def shutdown(self):
        print("Shutting down encoder logger...")
        self.request_shutdown()

        if self.serial_thread.is_alive():
            self.serial_thread.join()

        # Do not run filter if we are aborting.
        if self.force_abort:
            print("Abort active, skipping file processing.")
            return

        if self.serial_done:
            self.run_filter()
        elif not self.started:
            print("Encoder was never initialized; skipping filtering.")
        else:
            print("Serial thread did not finish properly; skipping filtering.")

    # ---------------------------
    # Internal functions
    # ---------------------------

    def read_serial(self):
        with serial.Serial(PORT, BAUD, timeout=1) as ser, open(
            self.output_file, "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["unix_time", "millis", "X", "Y", "Z", "dX", "dY", "dZ"])
            time.sleep(2)
            print("Connected to Arduino. Ready.\n")

            while True:
                if self.stop_program:
                    break

                line = ser.readline().decode(errors="ignore").strip()

                if self.stop_program:
                    break

                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    continue

                try:
                    floats = [float(p) for p in parts]
                except ValueError:
                    print("Ignoring malformed line:", line)
                    continue

                x_pos, y_pos, z_pos = floats[1], floats[2], floats[3]
                with self.position_lock:
                    self.current_position = (x_pos, y_pos, z_pos)
                    self.last_position_update_time = time.time()

                if self.logging_active and not self.stop_program:
                    unix_time = round(time.time() * 1000)

                    millis_val = f"{floats[0]:.0f}"
                    x_str, y_str, z_str, dx_str, dy_str, dz_str = (
                        f"{value:.3f}" for value in floats[1:]
                    )

                    writer.writerow(
                        [f"{unix_time}", millis_val, x_str, y_str, z_str, dx_str, dy_str, dz_str]
                    )
                    f.flush()

                    now = time.time()
                    if now - self.last_terminal_log_time >= 1.0:
                        print(unix_time, millis_val, x_str, y_str, z_str, dx_str, dy_str, dz_str)
                        self.last_terminal_log_time = now

            print("Encoder serial loop stopped.")

        self.serial_done = True

    def run_filter(self):
        print("\nFiltering logged data...\n")

        try:
            df_original = pd.read_csv(self.output_file, low_memory=False)
            filtered_df = self.auto_filter_csv(self.output_file, self.filtered_file)

            if filtered_df is None:
                print("Filtering skipped due to empty or invalid CSV.")
                return

            rows_removed = len(df_original) - len(filtered_df)
            _, _, ymin, ymax, _, _ = self.get_volume_bounds()
            print(f"Subprint filter: {self.run_index}/{TOTAL_SUBPRINTS}, Y={ymin:.1f} to {ymax:.1f}")

            print("Summary:")
            print(f"   Original rows : {len(df_original)}")
            print(f"   Filtered rows : {len(filtered_df)}")
            print(f"   Rows removed  : {rows_removed}")
            print(f"\nFiltered CSV saved to: {self.filtered_file}")

        except Exception as exc:
            print(f"Error during final filtering summary: {exc}")

    def get_current_position(self):
        with self.position_lock:
            return self.current_position

    def get_last_position_update_time(self):
        with self.position_lock:
            return self.last_position_update_time

    def auto_filter_csv(self, input_file, output_file):
        try:
            if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
                print("Encoder CSV is empty. Skipping filtering.")
                return None

            df = pd.read_csv(input_file, low_memory=False)

            required_cols = ["unix_time", "millis", "X", "Y", "Z", "dX", "dY", "dZ"]
            if not all(col in df.columns for col in required_cols):
                print("CSV missing required columns. Skipping filtering.")
                return None

            if df.empty:
                print("CSV contains no data rows. Skipping filtering.")
                return None

            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["X", "Y", "Z"])
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_volume_bounds()

            filtered_df = df[
                (df["X"] >= xmin)
                & (df["X"] <= xmax)
                & (df["Y"] >= ymin)
                & (df["Y"] <= ymax)
                & (df["Z"] >= zmin)
                & (df["Z"] <= zmax)
            ].copy()

            filtered_df.loc[:, "unix_time"] = filtered_df["unix_time"].round(4)
            filtered_df.loc[:, "millis"] = filtered_df["millis"].round(0)
            filtered_df.loc[:, ["X", "Y", "Z", "dX", "dY", "dZ"]] = filtered_df[
                ["X", "Y", "Z", "dX", "dY", "dZ"]
            ].round(3)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            filtered_df.to_csv(output_file, index=False, float_format="%.3f")

            removed = len(df) - len(filtered_df)
            print(f"   Subprint {self.run_index}/{TOTAL_SUBPRINTS} Y range: {ymin:.1f} to {ymax:.1f}")
            print(f"\nCSV filtered and saved to: {output_file}")
            print(f"   Rows kept: {len(filtered_df)} | Rows removed: {removed}\n")

            return filtered_df

        except Exception as exc:
            print(f"Error filtering CSV: {exc}")
            return None
