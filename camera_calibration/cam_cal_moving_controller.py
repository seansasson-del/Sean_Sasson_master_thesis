import time
from final_codes.cam_cal_moving_encoder import EncoderLogger
from final_codes.cam_cal_moving_camera import DEFAULT_INNER_RADIUS, DEFAULT_OUTER_RADIUS, IRCameraLogger
import os
import datetime
import shutil
import stat
import subprocess
import json


def create_session_folder():
    base_folder = "automation_test/data/pa6cf/cal_21_high_gain"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(
        base_folder,
        f"cal_{timestamp}_r{DEFAULT_INNER_RADIUS}_to_{DEFAULT_OUTER_RADIUS}",
    )
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

# ---------------------------------
# Configuration
# ---------------------------------
POS_A = (30.0, 30.0, 2.0)
POS_B = (30.0, 30.0, 100.0)

TOLERANCE = 4 # mm
STABLE_TIME = 5  # seconds

# ---------------------------------
# Utility Functions
# ---------------------------------

def save_metadata(session_folder, ir_camera, encoder, status="started"):
    """Creates/Updates a JSON metadata file for the current session."""
    
    metadata_path = os.path.join(session_folder, "metadata.json")

    data = {
        "session_id": os.path.basename(session_folder),
        "status": status,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "encoder_port": "COM6",
            "camera_device": ir_camera.device,
            "emissivity": ir_camera.emissivity,
            "center_point_emissivity": ir_camera.center_point_emissivity,
            "Fan": "Off",
            "Gain Mode": "High",
            "Material": "PA6-CF",

        },
        "parameters": {
            "inner_radius": ir_camera.inner_radius,
            "outer_radius": ir_camera.outer_radius,
            "center_position": ir_camera.center_position,
            "pixels_in_area": len(ir_camera.selected_pixels),
            "tolerance_mm": TOLERANCE,
            "Infill speed": "10 mm/s",
            "Extrusion multiplier": "1"
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Metadata saved to {metadata_path}")

def delete_session_folder(session_folder):
    """
    The ultimate Windows/OneDrive folder deleter. 
    Uses permission clearing and system-level calls.
    """
    if not os.path.exists(session_folder):
        return

    # Helper function to clear 'Read Only' attributes which OneDrive often sets
    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    print(f"Attempting to force-delete: {session_folder}")

    # 1. Try standard rmtree with a permission-fix handler
    for i in range(3):
        try:
            shutil.rmtree(session_folder, onerror=remove_readonly)
            print("Folder deleted.")
            return
        except Exception:
            time.sleep(1.0) # Wait for OneDrive to blink

    # 2. If it's STILL there, use the Windows Shell 'rd' command (Nuclear Option)
    # This often bypasses locks that Python's high-level libraries can't break.
    try:
        # /S = include subdirectories, /Q = quiet mode
        subprocess.run(['cmd', '/c', 'rd', '/s', '/q', os.path.abspath(session_folder)], check=True)
        print("Folder force-deleted via System Shell.")
    except Exception as e:
        print(f"Even the system shell couldn't kill it. OneDrive is holding it: {e}")


def within_tolerance(current, target, tol):
    return all(abs(c - t) <= tol for c, t in zip(current, target))

def wait_until_stable(get_position_func, target, abort_check):
    """
    Wait until position is within tolerance AND
    remains stable for STABLE_TIME seconds.
    Now constantly checks for an abort signal!
    """
    print(f"Waiting for stable position at {target} ...")
    stable_start = None

    while True:
        # Check if 'q' was pressed in the camera window
        if abort_check():
            return False

        pos = get_position_func()

        if pos is None:
            time.sleep(0.01)
            continue

        if within_tolerance(pos, target, TOLERANCE):
            if stable_start is None:
                stable_start = time.time()
            elif time.time() - stable_start >= STABLE_TIME:
                print(f"Stable at {target} for {STABLE_TIME} seconds.")
                return True
        else:
            stable_start = None

        time.sleep(0.01)


def wait_for_manual_encoder_init(ir_camera, abort_check):
    print("Camera recording is active.")
    print("Press 'I' in the camera window when you want to initialize the encoder.")

    while not ir_camera.manual_encoder_init_requested:
        if abort_check():
            return False
        time.sleep(0.01)

    return True


def execute_abort_sequence(encoder, ir, session_folder):
    """Helper function to kill everything safely."""
    print("Abort detected! Signalling threads to stop...")
    
    encoder.force_abort = True
    ir.force_abort = True

    encoder.shutdown() 
    ir.shutdown()

    delete_session_folder(session_folder)


# ---------------------------------
# Main Controller
# ---------------------------------
def main():
    session_folder = create_session_folder()
    print(f"\nSession folder: {session_folder}\n")

    print("\n=== Automatic Logging Controller ===")
    print("Camera recording starts immediately.")
    print("Press 'I' in the camera window after startup/probing is finished to initialize the encoder.\n")

    encoder = EncoderLogger(session_folder)
    ir = IRCameraLogger(session_folder)

    ir.start()
    ir.start_recording()

    save_metadata(session_folder, ir, encoder, status="recording")
    # Create a quick lambda function we can pass into our loops to check the flag
    check_for_abort = lambda: ir.abort_requested

    if not wait_for_manual_encoder_init(ir, check_for_abort):
        execute_abort_sequence(encoder, ir, session_folder)
        return

    print("Starting encoder initialization...")
    encoder.start()

    # ---------------------------------
    # Wait for encoder to initialize
    # ---------------------------------
    print("Waiting for first valid encoder reading...")
    while encoder.get_current_position() is None:
        if check_for_abort():
            execute_abort_sequence(encoder, ir, session_folder)
            return
        time.sleep(0.01)

    ir.mark_encoder_initialized()
    print("Encoder initialized.")

    # ---------------------------------
    # START CONDITION (Position A)
    # ---------------------------------
    # If wait_until_stable returns False, an abort was triggered.
    if not wait_until_stable(encoder.get_current_position, POS_A, check_for_abort):
        execute_abort_sequence(encoder, ir, session_folder)
        return

    print("\nSTART TRIGGERED\n")
    encoder.start_logging()

    # ---------------------------------
    # STOP CONDITION (Position B)
    # ---------------------------------
    if not wait_until_stable(encoder.get_current_position, POS_B, check_for_abort):
        execute_abort_sequence(encoder, ir, session_folder)
        return

    print("\nSTOP TRIGGERED\n")
    encoder.stop_logging()
    ir.stop_recording()
    ir.stop_video()

    # ---------------------------------
    # Clean shutdown
    # ---------------------------------
    encoder.shutdown()
    ir.shutdown()

    print("Recording completed successfully.")
    print(f"Encoder data saved to: {encoder.filtered_file}")
    print(f"Camera annulus data saved to: {ir.csv_filename}")
    print("Use t_cal_seperator.py on the camera CSV when you want per-radius extraction.")

if __name__ == "__main__":
    main()
