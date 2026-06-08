import argparse
import csv
import math
import os
import re
import tkinter as tk
from tkinter import TclError
from tkinter import filedialog

from final_codes.sequential_consolidation import generate_pydeck
from final_codes.sequential_generate_stt import merge_encoder_camera


DEFAULT_CENTER_X = 122
DEFAULT_CENTER_Y = 108
DEFAULT_ENCODER_FILENAME = "encoder_log_filtered.csv"
PIXEL_HEADER_PATTERN = re.compile(r"^p\d+_x(-?\d+)_y(-?\d+)$")
RADIUS_TO_PIXEL_COUNT = {
    2: 7,
    3: 9,
    4: 14,
    5: 15,
    6: 19,
    7: 21,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a full lower half-annulus IR CSV into one CSV per target radius, "
            "store each radius in its own folder, and run merge_data / generate_pydeck "
            "for each split file."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a CSV file created by t_cal_ir.py",
    )
    parser.add_argument(
        "--encoder-path",
        help=(
            "Path to the filtered encoder CSV. "
            f"Defaults to '{DEFAULT_ENCODER_FILENAME}' in the same folder as the input CSV."
        ),
    )
    parser.add_argument("--center-x", type=int, default=DEFAULT_CENTER_X)
    parser.add_argument("--center-y", type=int, default=DEFAULT_CENTER_Y)
    return parser.parse_args()


def select_input_file():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.askopenfilename(
            title="Select tc001 CSV file",
            filetypes=[
                ("TC001 CSV files", "tc001*.csv"),
                ("CSV files", "*.csv"),
                
            ],
            initialdir='automation_test/data/pa6cf/cal_21_high_gain'
        )
        root.destroy()
        return selected_path.strip()
    except TclError:
        return input("Select tc001 CSV file by path: ").strip()


def resolve_source_file(input_path):
    if os.path.isfile(input_path):
        return input_path

    raise FileNotFoundError(f"Input file does not exist: {input_path}")


def resolve_encoder_file(source_csv, encoder_path):
    if encoder_path:
        if os.path.isfile(encoder_path):
            return encoder_path
        raise FileNotFoundError(f"Encoder file does not exist: {encoder_path}")

    inferred_path = os.path.join(os.path.dirname(source_csv), DEFAULT_ENCODER_FILENAME)
    if os.path.isfile(inferred_path):
        return inferred_path

    raise FileNotFoundError(
        "Could not find the filtered encoder CSV automatically. "
        f"Expected: {inferred_path}. Use --encoder-path to specify it."
    )


def is_in_lower_half_measurement(y, center_y):
    return y >= center_y


def generate_half_circle_points(cx, cy, radius, n_points, top_half=False):
    if n_points < 2:
        raise ValueError("Half-circle extraction requires at least two points.")

    sign = -1 if top_half else 1
    return [
        (
            int(round(cx + radius * math.cos(math.pi * i / (n_points - 1)))),
            int(round(cy + sign * radius * math.sin(math.pi * i / (n_points - 1)))),
        )
        for i in range(n_points)
    ]


def build_source_lookup(header, center_y):
    metadata_indexes = []
    coordinate_to_source = {}

    for idx, column_name in enumerate(header):
        match = PIXEL_HEADER_PATTERN.match(column_name)
        if not match:
            metadata_indexes.append(idx)
            continue

        x_coord = int(match.group(1))
        y_coord = int(match.group(2))
        if not is_in_lower_half_measurement(y_coord, center_y):
            continue

        coordinate_to_source[(x_coord, y_coord)] = {
            "index": idx,
            "header": column_name,
        }

    if not coordinate_to_source:
        raise ValueError(
            "No pixel columns in the lower half measurement area were found. "
            "This splitter expects CSV files created by the current t_cal_ir.py format."
        )

    return metadata_indexes, coordinate_to_source


def build_radius_extraction_plan(header, center_x, center_y):
    metadata_indexes, coordinate_to_source = build_source_lookup(header, center_y)
    radius_to_plan = {}

    for radius, pixel_count in sorted(RADIUS_TO_PIXEL_COUNT.items()):
        target_points = generate_half_circle_points(center_x, center_y, radius, pixel_count)
        missing_points = [point for point in target_points if point not in coordinate_to_source]
        if missing_points:
            missing_str = ", ".join(f"({x},{y})" for x, y in missing_points)
            raise ValueError(
                f"Source CSV does not contain all required pixels for radius {radius}. "
                f"Missing coordinates: {missing_str}"
            )

        source_indexes = [coordinate_to_source[point]["index"] for point in target_points]
        source_headers = [coordinate_to_source[point]["header"] for point in target_points]
        radius_to_plan[radius] = {
            "pixel_count": pixel_count,
            "target_points": target_points,
            "source_indexes": source_indexes,
            "source_headers": source_headers,
        }

    return metadata_indexes, radius_to_plan


def normalize_row_length(row, expected_len):
    if len(row) < expected_len:
        return row + [""] * (expected_len - len(row))
    if len(row) > expected_len:
        return row[:expected_len]
    return row


def create_output_writers(source_csv, header, metadata_indexes, radius_to_plan):
    source_dir = os.path.dirname(source_csv)
    source_stem = os.path.splitext(os.path.basename(source_csv))[0]

    writers = {}
    handles = []
    metadata_header = [header[idx] for idx in metadata_indexes]

    for radius, plan in sorted(radius_to_plan.items()):
        radius_dir = os.path.join(source_dir, f"R{radius}")
        os.makedirs(radius_dir, exist_ok=True)

        output_path = os.path.join(radius_dir, f"{source_stem}_radius_{radius}.csv")
        handle = open(output_path, "w", newline="")
        handles.append(handle)
        writer = csv.writer(handle)

        selected_indexes = metadata_indexes + plan["source_indexes"]
        pixel_headers = [f"p{i}" for i in range(plan["pixel_count"])]
        writer.writerow(metadata_header + pixel_headers)

        writers[radius] = {
            "writer": writer,
            "indexes": selected_indexes,
            "path": output_path,
            "radius_dir": radius_dir,
            "pixel_count": plan["pixel_count"],
            "target_points": plan["target_points"],
            "source_headers": plan["source_headers"],
        }

    return source_dir, writers, handles


def run_post_processing(encoder_file, camera_file, output_dir):
    merged_file = os.path.join(output_dir, "merged_data.csv")
    summary_file = os.path.join(output_dir, "summary_filter.csv")
    html_output = os.path.join(output_dir, "visualization.html")

    merge_encoder_camera(encoder_file, camera_file, merged_file, summary_file)
    generate_pydeck(merged_file, html_output)

    return {
        "merged_file": merged_file,
        "summary_file": summary_file,
        "html_output": html_output,
        "false_readings_file": os.path.join(output_dir, "false_readings_filtered.csv"),
    }


def split_csv_by_radius(source_csv, encoder_file, center_x, center_y):
    with open(source_csv, "r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV file is empty: {source_csv}")

        metadata_indexes, radius_to_plan = build_radius_extraction_plan(
            header,
            center_x,
            center_y,
        )
        output_dir, writers, handles = create_output_writers(
            source_csv,
            header,
            metadata_indexes,
            radius_to_plan,
        )

        try:
            for row in reader:
                row = normalize_row_length(row, len(header))
                for config in writers.values():
                    selected_row = [row[idx] for idx in config["indexes"]]
                    config["writer"].writerow(selected_row)
        finally:
            for handle in handles:
                handle.close()

    print(f"Processed: {source_csv}")
    print(f"Base output folder: {output_dir}")

    for radius in sorted(writers):
        config = writers[radius]
        point_summary = ", ".join(
            f"p{i}->{point}"
            for i, point in enumerate(config["target_points"])
        )
        print(
            f"  Radius {radius}: {config['pixel_count']} extracted pixels -> {config['path']}"
        )
        print(f"    {point_summary}")

        post_outputs = run_post_processing(
            encoder_file=encoder_file,
            camera_file=config["path"],
            output_dir=config["radius_dir"],
        )
        print(f"    merged -> {post_outputs['merged_file']}")
        print(f"    summary -> {post_outputs['summary_file']}")
        print(f"    false readings -> {post_outputs['false_readings_file']}")
        print(f"    visualization -> {post_outputs['html_output']}")


def main():
    args = parse_args()
    input_path = args.input_path or select_input_file()
    if not input_path:
        raise ValueError("No input file selected.")

    source_csv = resolve_source_file(input_path)
    encoder_file = resolve_encoder_file(source_csv, args.encoder_path)

    split_csv_by_radius(
        source_csv=source_csv,
        encoder_file=encoder_file,
        center_x=args.center_x,
        center_y=args.center_y,
    )


if __name__ == "__main__":
    main()
