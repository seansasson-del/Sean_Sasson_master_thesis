import os
import pandas as pd
import numpy as np
import pydeck as pdk
import tkinter as tk
from tkinter import filedialog


def generate_pydeck(merged_file, output_html):
    # Create output filenames automatically
    
    color_by = "p_temp"                         # column to color points by ('p_temp')
    base_point_size = 4                       # larger so points are visible
    alpha = 200
    # ---------- FIXED INITIAL VIEW SETTINGS ----------
    INITIAL_ROTATION_ORBIT = 90     # horizontal rotation around object
    INITIAL_ROTATION_X = 20        # tilt angle (0 = top-down, 90 = side view)
    INITIAL_ZOOM = 4              # smaller zoom = farther away
    INITIAL_TARGET = [-10, 0, 2]      # center of the scene
    # ------------------------------------
    def get_unique_filename(filepath):
        """Return a unique filepath by appending _1, _2, etc. if file exists."""
        base, ext = os.path.splitext(filepath)
        counter = 1
        newpath = filepath
        while os.path.exists(newpath):
            newpath = f"{base}_{counter}{ext}"
            counter += 1
        return newpath

    output_html = get_unique_filename(output_html)

    # --- Load and clean data ---
    df = pd.read_csv(merged_file)
    for col in ["X", "Y", "Z", color_by]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {merged_file}")

    df = df.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)

    # --- Rescale coordinates for visualization ---
    x, y, z = df["X"].values, df["Y"].values, df["Z"].values
    xmid, ymid, zmid = np.mean(x), np.mean(y), np.mean(z)
    xspan, yspan, zspan = np.ptp(x), np.ptp(y), np.ptp(z)

    scale = max(xspan, yspan, zspan)
    if scale == 0 or np.isnan(scale):
        scale = 1.0

    df["x_scaled"] = (x - xmid) / scale * 100.0
    df["y_scaled"] = (y - ymid) / scale * 100.0
    df["z_scaled"] = (z - zmid) / scale * 100.0

    # --- Temperature-based color map (simple blue→red) ---
    t = df[color_by].astype(float)
    tmin = 60
    tmax = 250

    df["temp_norm"] = (t - tmin) / (tmax - tmin)
    df["r"] = (255 * df["temp_norm"]).astype(int)
    df["g"] = (255 * (1 - abs(df["temp_norm"] - 0.5) * 2)).astype(int)
    df["b"] = (255 * (1 - df["temp_norm"])).astype(int)
    df["a"] = alpha

    # --- Build PointCloudLayer ---
    df["position"] = df[["x_scaled", "y_scaled", "z_scaled"]].values.tolist()

    layer = pdk.Layer(
        "PointCloudLayer",
        data=df.to_dict(orient="records"),  # <-- FIX: Convert to native Python dicts
        get_position="position",
        get_color=["r", "g", "b", "a"],
        point_size=base_point_size,
        pickable=True,
        auto_highlight=True,
    )

    # --- Define OrbitView (Fixed Camera Setup) ---
    view = pdk.View(type="OrbitView", controller=True)
    view_state = pdk.ViewState(
        target=INITIAL_TARGET,              # center position
        rotation_orbit=INITIAL_ROTATION_ORBIT,  # horizontal rotation
        rotation_x=INITIAL_ROTATION_X,      # tilt
        zoom=INITIAL_ZOOM                   # zoom level
    )

    # --- Combine everything ---
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        views=[view],
        map_style=None,
        tooltip={"text": "X: {X}\nY: {Y}\nZ: {Z}\nTemp: {" + color_by + "}"}
    )

    deck.to_html(output_html, notebook_display=False, iframe_height=800)
    print(f"✅ 3D point cloud saved to {output_html}")
    print(f"🧭 Fixed camera view → rotation_orbit={INITIAL_ROTATION_ORBIT}, rotation_x={INITIAL_ROTATION_X}, zoom={INITIAL_ZOOM}")
    print(f"Points plotted: {len(df)}")
