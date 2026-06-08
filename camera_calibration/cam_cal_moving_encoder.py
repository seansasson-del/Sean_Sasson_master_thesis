import serial
import time
import csv
import threading
import pandas as pd
import os

# --- Settings ---
PORT = 'COM6'
BAUD = 115200

# --- Build volume limits ---
Xmin, Xmax = 120, 242
Ymin, Ymax = 120, 242 
Zmin, Zmax = 0.0, 151

class EncoderLogger:

    def __init__(self, session_folder):
        self.session_folder = session_folder

        self.output_file = os.path.join(
            session_folder,
            "encoder_log_unix.csv"
        )

        self.filtered_file = os.path.join(
            session_folder,
            "encoder_log_filtered.csv"
        )
        self.current_position = None
        self.logging_active = False
        self.stop_program = False
        self.serial_done = False
        self.force_abort = False
        self.last_terminal_log_time = 0.0
        self.started = False

        self.serial_thread = threading.Thread(target=self.read_serial)

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
            print("âœ… Encoder logging started.")

    def stop_logging(self):
        if self.logging_active:
            self.logging_active = False
            print("â¸ï¸ Encoder logging stopped.")

    def shutdown(self):
        print("ðŸ‘‹ Shutting down encoder logger...")
        self.logging_active = False
        self.stop_program = True

        if self.serial_thread.is_alive():
            self.serial_thread.join()

        # CRITICAL: Do not run filter if we are aborting!
        if self.force_abort:
            print("Abort active â€” skipping file processing.")
            return

        if self.serial_done:
            self.run_filter()
        elif not self.started:
            print("Encoder was never initialized; skipping filtering.")
        else:
            print("âš ï¸ Serial thread did not finish properly; skipping filtering.")

    # ---------------------------
    # Internal functions
    # ---------------------------

    def read_serial(self):
        with serial.Serial(PORT, BAUD, timeout=1) as ser, \
                open(self.output_file, 'w', newline='') as f:

            writer = csv.writer(f)
            writer.writerow(['unix_time', 'millis', 'X', 'Y', 'Z', 'dX', 'dY', 'dZ'])
            time.sleep(2)
            print("Connected to Arduino. Ready.\n")

            while True:

                # Exit immediately if controller requested shutdown
                if self.stop_program:
                    break

                line = ser.readline().decode(errors='ignore').strip()

                # Check again after blocking read
                if self.stop_program:
                    break

                if not line:
                    continue

                parts = line.split(',')
                if len(parts) != 7:
                    continue

                try:
                    floats = [float(p) for p in parts]
                except ValueError:
                    print("âš ï¸ Ignoring malformed line:", line)
                    continue

                # ALWAYS update current position
                Xf, Yf, Zf = floats[1], floats[2], floats[3]
                self.current_position = (Xf, Yf, Zf)

                # ONLY log if active
                if self.logging_active and not self.stop_program:

                    unix_time = round(time.time() * 1000)

                    millis_val = f"{floats[0]:.0f}"
                    X, Y, Z, dX, dY, dZ = (f"{v:.3f}" for v in floats[1:])

                    writer.writerow(
                        [f"{unix_time}", millis_val, X, Y, Z, dX, dY, dZ]
                    )

                    # Flush occasionally to avoid losing data
                    f.flush()

                    # Throttle terminal output without affecting CSV logging.
                    now = time.time()
                    if not self.stop_program and now - self.last_terminal_log_time >= 1.0:
                        print(f"{unix_time}", millis_val, X, Y, Z, dX, dY, dZ)
                        self.last_terminal_log_time = now

            print("Encoder serial loop stopped.")

        self.serial_done = True
            
    def run_filter(self):
        print("\nFiltering logged data...\n")

        try:
            df_original = pd.read_csv(self.output_file, low_memory=False)
            filtered_df = self.auto_filter_csv(self.output_file, self.filtered_file)

            if filtered_df is None:
                print("âš ï¸ Filtering skipped due to empty or invalid CSV.")
                return

            rows_removed = len(df_original) - len(filtered_df)

            print(f"ðŸ“Š Summary:")
            print(f"   Original rows : {len(df_original)}")
            print(f"   Filtered rows : {len(filtered_df)}")
            print(f"   Rows removed  : {rows_removed}")
            print(f"\nâœ… Filtered CSV saved to: {self.filtered_file}")

        except Exception as e:
            print(f"âš ï¸ Error during final filtering summary: {e}")

    def get_current_position(self):
        return self.current_position

    def auto_filter_csv(self, input_file, output_file):
        try:
            # If file doesn't exist or is empty â†’ skip filtering
            if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
                print("âš ï¸ Encoder CSV is empty. Skipping filtering.")
                return None

            df = pd.read_csv(input_file, low_memory=False)

            # If expected columns are missing â†’ skip
            required_cols = ['unix_time', 'millis', 'X', 'Y', 'Z', 'dX', 'dY', 'dZ']
            if not all(col in df.columns for col in required_cols):
                print("âš ï¸ CSV missing required columns. Skipping filtering.")
                return None

            if df.empty:
                print("âš ï¸ CSV contains no data rows. Skipping filtering.")
                return None

            # Convert numeric columns
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['X', 'Y', 'Z'])

            filtered_df = df[
                (df['X'] >= Xmin) & (df['X'] <= Xmax) &
                (df['Y'] >= Ymin) & (df['Y'] <= Ymax) &
                (df['Z'] >= Zmin) & (df['Z'] <= Zmax)
            ].copy()

            filtered_df.loc[:, 'unix_time'] = filtered_df['unix_time'].round(4)
            filtered_df.loc[:, 'millis'] = filtered_df['millis'].round(0)
            filtered_df.loc[:, ['X', 'Y', 'Z', 'dX', 'dY', 'dZ']] = \
                filtered_df[['X', 'Y', 'Z', 'dX', 'dY', 'dZ']].round(3)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            filtered_df.to_csv(output_file, index=False, float_format='%.3f')

            removed = len(df) - len(filtered_df)
            print(f"\nâœ… CSV filtered and saved to: {output_file}")
            print(f"   Rows kept: {len(filtered_df)} | Rows removed: {removed}\n")

            return filtered_df

        except Exception as e:
            print(f"âš ï¸ Error filtering CSV: {e}")
            return None
