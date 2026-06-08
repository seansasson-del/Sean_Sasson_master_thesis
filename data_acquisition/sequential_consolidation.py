import os
import re

import numpy as np
import pandas as pd

MIN_TEMP_FILTER = 60
MAX_TEMP_FILTER = 280
ANGLE_THRESHOLD_DEG = 5
travel_speed = 0.15


def pixel_sort_key(column_name):
    match = re.fullmatch(r"p(\d+)", column_name)
    if match:
        return int(match.group(1))
    return float("inf")


def find_false_temperature_rows(camera_df, pixel_columns):
    report_columns = [
        "matched_camera_row_number",
        "timestamp",
        "center_temp",
        "pixels_above_center",
        "pixel_values_above_center",
    ]

    if "center_temp" not in camera_df.columns:
        return pd.DataFrame(columns=report_columns)

    camera_df = camera_df.copy()
    camera_df["center_temp"] = pd.to_numeric(camera_df["center_temp"], errors="coerce")

    for column in pixel_columns:
        camera_df[column] = pd.to_numeric(camera_df[column], errors="coerce")

    exceeds_center = camera_df[pixel_columns].gt(camera_df["center_temp"], axis=0)
    false_rows_mask = exceeds_center.any(axis=1)
    false_rows = camera_df.loc[false_rows_mask].copy()

    if false_rows.empty:
        return pd.DataFrame(columns=report_columns)

    false_rows["pixels_above_center"] = exceeds_center.loc[false_rows_mask].apply(
        lambda row: [column for column, is_above in row.items() if is_above],
        axis=1,
    )
    false_rows["pixel_values_above_center"] = false_rows.apply(
        lambda row: {column: row[column] for column in row["pixels_above_center"]},
        axis=1,
    )

    report_df = false_rows[
        ["camera_row_number", "timestamp", "center_temp", "pixels_above_center", "pixel_values_above_center"]
    ].rename(columns={"camera_row_number": "matched_camera_row_number"})

    report_df["timestamp"] = report_df["timestamp"].round().astype("int64")
    return report_df


def filter_stable_extrusion_layers(df, z_column="Z"):
    z_tol = 0.05 # Used to validate if points are within the same layer
    min_nodes = 10
    expected_layer_height = 0.3
    layer_height_tol = 0.1 # Used to say if the next layer is within the tolerance of the expected



    def is_stable_window(values, start_idx, window_size, tol):
        if start_idx + window_size > len(values):
            return False, np.nan
        window = values[start_idx:start_idx + window_size]
        center = float(np.median(window))
        return bool(np.all(np.abs(window - center) <= tol)), center

    filtered_df = df.copy()
    z_values = filtered_df[z_column].to_numpy(dtype=float)
    layer_ids = np.full(len(filtered_df), -1, dtype=int)
    layer_z_values = np.full(len(filtered_df), np.nan, dtype=float)

    current_idx = 0
    layer_id = 0
    previous_layer_z = None

    while current_idx < len(z_values):
        stable_found = False
        while current_idx < len(z_values):
            stable_found, current_layer_z = is_stable_window(z_values, current_idx, min_nodes, z_tol)
            if not stable_found:
                current_idx += 1
                continue

            if previous_layer_z is None:
                break

            layer_step = current_layer_z - previous_layer_z
            same_layer = abs(layer_step) <= z_tol
            expected_step = abs(layer_step - expected_layer_height) <= layer_height_tol
            if same_layer or expected_step:
                break

            current_idx += 1
            stable_found = False

        if not stable_found:
            break

        in_layer_indices = []
        scan_idx = current_idx

        while scan_idx < len(z_values):
            if abs(z_values[scan_idx] - current_layer_z) <= z_tol:
                in_layer_indices.append(scan_idx)
                current_layer_z = float(np.median(z_values[in_layer_indices]))
                scan_idx += 1
                continue

            same_layer_ahead, _ = is_stable_window(z_values, scan_idx + 1, min_nodes, z_tol)
            if same_layer_ahead:
                next_window = z_values[scan_idx + 1:scan_idx + 1 + min_nodes]
                if np.all(np.abs(next_window - current_layer_z) <= z_tol):
                    scan_idx += 1
                    continue

            next_layer_idx = scan_idx + 1
            found_next_layer = False
            while next_layer_idx < len(z_values):
                next_stable, next_layer_z = is_stable_window(z_values, next_layer_idx, min_nodes, z_tol)
                if not next_stable:
                    next_layer_idx += 1
                    continue

                layer_step = next_layer_z - current_layer_z
                same_layer = abs(layer_step) <= z_tol
                expected_step = abs(layer_step - expected_layer_height) <= layer_height_tol
                if same_layer or expected_step:
                    found_next_layer = True
                    break

                next_layer_idx += 1

            if found_next_layer:
                break

            scan_idx += 1

        if len(in_layer_indices) >= min_nodes:
            layer_ids[in_layer_indices] = layer_id
            layer_z_values[in_layer_indices] = current_layer_z
            previous_layer_z = current_layer_z
            layer_id += 1

        current_idx = scan_idx + 1

    filtered_df["z_bin"] = layer_z_values
    filtered_df["layer"] = layer_ids

    total_rows = len(filtered_df)
    dropped_rows = int((filtered_df["layer"] < 0).sum())
    dropped_pct = (dropped_rows / total_rows * 100) if total_rows else 0.0
    print(
        f"Dropped rows not in stable extrusion layers: "
        f"{dropped_rows} of {total_rows} ({dropped_pct:.2f}%)"
    )

    filtered_df = filtered_df[filtered_df["layer"] >= 0].copy()
    filtered_df["layer"] = filtered_df["layer"].astype(int)
    filtered_df["layer_median_z"] = filtered_df.groupby("layer")[z_column].transform("median")

    print(f"Detected layers: {filtered_df['layer'].nunique()}")
    return filtered_df, dropped_rows, dropped_pct


def merge_encoder_camera(encoder_file, camera_file, output_file, summary_file):
    # --- LOAD DATA ---
    enc = pd.read_csv(encoder_file, header=0)
    cam = pd.read_csv(camera_file, header=0)

    cam_len_initial = len(cam)
    enc_len_initial = len(enc)

    # --- Detect pixel columns dynamically ---
    pixel_cols = sorted(
        [column for column in cam.columns if re.fullmatch(r"p\d+", column)],
        key=pixel_sort_key,
    )
    num_pixels = len(pixel_cols)

    if num_pixels == 0:
        raise ValueError("No pixel columns found in camera data.")

    print(f"Detected {num_pixels} pixel points: {pixel_cols[0]} ... {pixel_cols[-1]}")

    # --- SORT & TRIM TO TIME OVERLAP ---
    enc["unix_time"] = enc["unix_time"].astype(float)
    cam["timestamp"] = cam["timestamp"].astype(float)

    enc = enc.sort_values("unix_time").reset_index(drop=True)
    cam = cam.sort_values("timestamp").reset_index(drop=True)

    enc_min, enc_max = enc["unix_time"].min(), enc["unix_time"].max()
    cam_min, cam_max = cam["timestamp"].min(), cam["timestamp"].max()
    start_time, end_time = max(enc_min, cam_min), min(enc_max, cam_max)
    tolerance = 20 * 1000

    if abs(enc_min - cam_min) > tolerance:
        print(f"Start times differ by {abs(enc_min - cam_min):.3f}s, trimming to overlap")
    else:
        print("Start times within tolerance")

    enc = enc[(enc["unix_time"] >= start_time) & (enc["unix_time"] <= end_time)]
    cam = cam[(cam["timestamp"] >= start_time) & (cam["timestamp"] <= end_time)]
    enc = enc.reset_index(drop=True)
    cam = cam.reset_index(drop=True)
    cam["camera_row_number"] = cam.index + 1

    cam_len_trimmed = len(cam)
    enc_len_trimmed = len(enc)

    false_readings_report = find_false_temperature_rows(cam, pixel_cols)
    false_readings_path = os.path.join(os.path.dirname(output_file), "false_readings_filtered.csv")

    # --- MERGE ASOF ---
    merged = pd.merge_asof(
        cam.sort_values("timestamp"),
        enc.sort_values("unix_time"),
        left_on="timestamp",
        right_on="unix_time",
        direction="nearest",
        tolerance=100,
    )

    # --- COMPUTE ANGLE OF MOTION ---
    merged["angle_rad"] = np.arctan2(merged["dY"], merged["dX"])
    merged["angle_deg"] = np.degrees(merged["angle_rad"]).replace([np.inf, -np.inf], np.nan)
    positive_angle_fix_mask = merged["angle_deg"].gt(90) & merged["angle_deg"].lt(90 + ANGLE_THRESHOLD_DEG)
    negative_angle_fix_mask = merged["angle_deg"].lt(-90) & merged["angle_deg"].gt(-90 - ANGLE_THRESHOLD_DEG)
    small_dx_mask = positive_angle_fix_mask | negative_angle_fix_mask
    merged.loc[positive_angle_fix_mask, "angle_deg"] = 90.0
    merged.loc[negative_angle_fix_mask, "angle_deg"] = -90.0
    valid_direction_mask = (merged["dX"] >= 0) | small_dx_mask
    merged["angle_deg"] = merged["angle_deg"].fillna(0)
    merged = merged[
        valid_direction_mask & merged["angle_deg"].between(-90, 90, inclusive="both")
    ].reset_index(drop=True)

    count_small_dx_corrected = int(small_dx_mask.sum())
    print(
        f"Remapped {count_small_dx_corrected} near-vertical rows with angle within "
        f"{ANGLE_THRESHOLD_DEG} degrees of +/-90 to +/-90 degrees."
    )

    # --- MAP ANGLE TO PIXEL INDEX ---
    merged["p_index"] = ((90 - merged["angle_deg"]) / 180 * (num_pixels - 1)).round().astype(int)
    merged["p_index"] = merged["p_index"].clip(0, num_pixels - 1)

    # --- GET PIXEL TEMPERATURE ---
    def get_pixel_temp(row):
        if pd.isna(row["p_index"]):
            return np.nan
        column = f"p{int(row['p_index'])}"
        return row[column] if column in merged.columns else np.nan

    merged["p_temp"] = merged.apply(get_pixel_temp, axis=1)

    # --- CLEANUP ---
    merged = merged[
        ["camera_row_number", "timestamp", "p_index", "p_temp", "X", "Y", "Z", "dX", "dY", "dZ", "angle_deg"]
    ]
    merged = merged.dropna(subset=["X", "Y", "Z", "p_temp"]).reset_index(drop=True)
    total_rows = len(merged)

    false_camera_rows = set(false_readings_report["matched_camera_row_number"].tolist())
    false_readings_report = false_readings_report[
        false_readings_report["matched_camera_row_number"].isin(set(merged["camera_row_number"].tolist()))
    ].reset_index(drop=True)
    count_false_readings = int(merged["camera_row_number"].isin(false_camera_rows).sum())

    initial_negative_dz_mask = merged.index.to_series().lt(5) & merged["dZ"].lt(0)
    count_initial_negative_dz = int(initial_negative_dz_mask.sum())
    merged = merged[~initial_negative_dz_mask].reset_index(drop=True)

    merged = merged[["timestamp", "p_index", "p_temp", "X", "Y", "Z", "dX", "dY", "dZ", "angle_deg"]]
    merged = merged.round(4)
    merged["timestamp"] = merged["timestamp"].astype("int64")

    # --- STATS BEFORE FILTER ---
    avg_temp_init = merged["p_temp"].mean()
    min_temp_init = merged["p_temp"].min()
    max_temp_init = merged["p_temp"].max()

    print("\n--- INITIAL DATA STATS ---")
    print(f"Rows total: {total_rows}")
    print(f"Average p_temp: {avg_temp_init:.3f}")
    print(f"Max p_temp: {max_temp_init:.3f}")
    print(f"Min p_temp: {min_temp_init:.3f}")

    pct_false_readings = (count_false_readings / total_rows) * 100 if total_rows else 0
    pct_initial_negative_dz = (count_initial_negative_dz / total_rows) * 100 if total_rows else 0

    print(
        f"Detected {count_false_readings} rows ({pct_false_readings:.2f}% of total) "
        "with false temperature readings."
    )
    print(
        f"Filtered out {count_initial_negative_dz} rows ({pct_initial_negative_dz:.2f}% of total) "
        "from the first 5 rows where dZ < 0."
    )

    # --- FILTER 1: Remove rows where p_temp > MAX_TEMP_FILTER ---
    max_filter = merged[merged["p_temp"] > MAX_TEMP_FILTER]
    merged = merged[merged["p_temp"] <= MAX_TEMP_FILTER].reset_index(drop=True)
    count_max_filter = len(max_filter)
    pct_max_filter = (count_max_filter / total_rows) * 100 if total_rows else 0
    print(
        f"\nFiltered out {count_max_filter} rows ({pct_max_filter:.2f}% of total) "
        f"with p_temp > {MAX_TEMP_FILTER}C."
    )

    # --- FILTER 2: Remove rows where p_temp < MIN_TEMP_FILTER ---
    min_filter = merged[merged["p_temp"] < MIN_TEMP_FILTER]
    merged = merged[merged["p_temp"] >= MIN_TEMP_FILTER].reset_index(drop=True)
    count_min_filter = len(min_filter)
    pct_min_filter = (count_min_filter / total_rows) * 100 if total_rows else 0
    print(
        f"Filtered out {count_min_filter} rows ({pct_min_filter:.2f}% of total) "
        f"with p_temp < {MIN_TEMP_FILTER}C."
    )

    # --- FILTER 3: Remove rows where sqrt(dx^2 + dy^2) > travel_speed ---
    merged["move_speed"] = np.sqrt(merged["dX"] ** 2 + merged["dY"] ** 2)
    too_fast = merged[merged["move_speed"] > travel_speed]
    merged = merged[merged["move_speed"] <= travel_speed].reset_index(drop=True)
    count_fast = len(too_fast)
    pct_fast = (count_fast / total_rows) * 100 if total_rows else 0
    print(f"Filtered out {count_fast} rows ({pct_fast:.2f}% of total) where movement speed > {travel_speed} ")

    # --- FILTER 4: Remove rows not belonging to stable extrusion layers ---
    merged, count_unstable_layers, pct_unstable_layers = filter_stable_extrusion_layers(merged)
    merged = merged[["timestamp", "p_index", "p_temp", "X", "Y", "Z", "dX", "dY", "dZ", "angle_deg"]].copy()

    # --- FINAL STATS AFTER FILTERING ---
    remaining = len(merged)
    total_removed = (
        count_initial_negative_dz
        + count_max_filter
        + count_min_filter
        + count_fast
        + count_unstable_layers
    )
    total_pct_removed = (total_removed / total_rows) * 100 if total_rows else 0

    avg_temp_final = merged["p_temp"].mean()
    min_temp_final = merged["p_temp"].min()
    max_temp_final = merged["p_temp"].max()

    print("\n--- FINAL FILTER SUMMARY ---")
    print(f"Total rows removed: {total_removed} ({total_pct_removed:.2f}% of total)")
    print(f"Remaining rows: {remaining} ({100 - total_pct_removed:.2f}% of total)")
    print(f"Final average p_temp: {avg_temp_final:.3f}")
    print(f"Final max p_temp: {max_temp_final:.3f}")
    print(f"Final min p_temp: {min_temp_final:.3f}")

    # --- SAVE CLEANED DATA ---
    merged.to_csv(output_file, index=False, float_format="%.4f")
    print(f"\nFinal cleaned dataset saved to:\n{output_file}")

    false_readings_report.to_csv(false_readings_path, index=False)
    print(f"False temperature readings report saved to:\n{false_readings_path}")

    # --- SAVE SUMMARY TO CSV ---
    summary_data = {
        "Camera rows before trimming": cam_len_initial,
        "Encoder rows before trimming": enc_len_initial,
        "Camera rows after trimming": cam_len_trimmed,
        "Encoder rows after trimming": enc_len_trimmed,
        "Post merging - Pre filtering": total_rows,
        "Post filtering": remaining,
        "Total rows removed in filtering": total_removed,
        "% total removed in filtering": total_pct_removed,
        "Initial avg p_temp": avg_temp_init,
        "Initial min p_temp": min_temp_init,
        "Initial max p_temp": max_temp_init,
        "Reported false temperature readings": count_false_readings,
        "% false temperature readings": pct_false_readings,
        "Filtered initial dZ < 0 in first 5 rows": count_initial_negative_dz,
        "% initial dZ < 0 in first 5 rows": pct_initial_negative_dz,
        f"Filtered over {MAX_TEMP_FILTER}C": count_max_filter,
        f"% over {MAX_TEMP_FILTER}C": pct_max_filter,
        f"Filtered under {MIN_TEMP_FILTER}C": count_min_filter,
        f"% under {MIN_TEMP_FILTER}C": pct_min_filter,
        "Filtered move_speed > 1.9": count_fast,
        "% move_speed > 1.9": pct_fast,
        "Filtered unstable extrusion layers": count_unstable_layers,
        "% unstable extrusion layers": pct_unstable_layers,
        "Remapped near-vertical rows": count_small_dx_corrected,
        "Angle threshold around +/-90 deg": ANGLE_THRESHOLD_DEG,
        "Final avg p_temp": avg_temp_final,
        "Final min p_temp": min_temp_final,
        "Final max p_temp": max_temp_final,
    }

    summary_rows = [
        ("Camera rows before trimming", cam_len_initial),
        ("Encoder rows before trimming", enc_len_initial),
        ("Camera rows after trimming", cam_len_trimmed),
        ("Encoder rows after trimming", enc_len_trimmed),
        ("", ""),
        ("Post merging - Pre filtering", summary_data["Post merging - Pre filtering"]),
        ("Post filtering", summary_data["Post filtering"]),
        ("Total rows removed in filtering", summary_data["Total rows removed in filtering"]),
        ("% total removed in filtering", summary_data["% total removed in filtering"]),
        ("", ""),
        ("Initial avg p_temp", summary_data["Initial avg p_temp"]),
        ("Initial min p_temp", summary_data["Initial min p_temp"]),
        ("Initial max p_temp", summary_data["Initial max p_temp"]),
        ("", ""),
        ("Final avg p_temp", summary_data["Final avg p_temp"]),
        ("Final min p_temp", summary_data["Final min p_temp"]),
        ("Final max p_temp", summary_data["Final max p_temp"]),
        ("", ""),
        ("Reported false temperature readings", summary_data["Reported false temperature readings"]),
        ("% false temperature readings", summary_data["% false temperature readings"]),
        ("", ""),
        ("Filtered initial dZ < 0 in first 5 rows", summary_data["Filtered initial dZ < 0 in first 5 rows"]),
        ("% initial dZ < 0 in first 5 rows", summary_data["% initial dZ < 0 in first 5 rows"]),
        ("", ""),
        (f"Filtered over {MAX_TEMP_FILTER}C", summary_data[f"Filtered over {MAX_TEMP_FILTER}C"]),
        (f"% over {MAX_TEMP_FILTER}C", summary_data[f"% over {MAX_TEMP_FILTER}C"]),
        ("", ""),
        (f"Filtered under {MIN_TEMP_FILTER}C", summary_data[f"Filtered under {MIN_TEMP_FILTER}C"]),
        (f"% under {MIN_TEMP_FILTER}C", summary_data[f"% under {MIN_TEMP_FILTER}C"]),
        ("", ""),
        ("Filtered move_speed > 1.9", summary_data["Filtered move_speed > 1.9"]),
        ("% move_speed > 1.9", summary_data["% move_speed > 1.9"]),
        ("", ""),
        ("Filtered unstable extrusion layers", summary_data["Filtered unstable extrusion layers"]),
        ("% unstable extrusion layers", summary_data["% unstable extrusion layers"]),
        ("", ""),
        ("Remapped near-vertical rows", summary_data["Remapped near-vertical rows"]),
        ("Angle threshold around +/-90 deg", summary_data["Angle threshold around +/-90 deg"]),
    ]

    summary_df = pd.DataFrame(summary_rows, columns=["Feauture", "Value"])
    summary_df.to_csv(summary_file, index=False, float_format="%.3f")

    print(f"\nFilter summary saved to:\n{summary_file}")
