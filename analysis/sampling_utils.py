import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from universal_transcoder.plots_and_logs import *
from universal_transcoder.calculations.energy_intensity import *
from usat_designer.processing.constants import *
from usat_designer.processing.plots_usat_designer import *
import usat_designer.utils.parameter_utils as pu
import warnings

def get_width_and_angular_error(cloud_points, S, output_layout):
    radial_i        = radial_I_calculation(cloud_points, S, output_layout)
    transverse_i    = transverse_I_calculation(cloud_points, S, output_layout)
    
    angular_error_calc  = angular_error(radial_i, transverse_i)
    width_calc          = width_angle(radial_i)

    return angular_error_calc, width_calc

from typing import Optional

def create_df_from_files(base_dir: str, max_folders: Optional[int] = None):
    entries = []

    outputs_dir = os.path.join(base_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        raise ValueError(f"No outputs directory found at {outputs_dir}")
    
    subfolders = [f for f in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, f))]
    if not subfolders:
        raise ValueError(f"No subfolders found inside outputs directory at {outputs_dir}")
    
    second_level_dir = os.path.join(outputs_dir, subfolders[0])
    if not os.path.isdir(second_level_dir):
        raise ValueError(f"No directory found at second level: {second_level_dir}")

    folders = os.listdir(second_level_dir)
    if max_folders is not None:
        folders = folders[:max_folders]

    for folder_name in tqdm(folders, desc="Loading folders"):
        folder_path = os.path.join(second_level_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            seed = int(folder_name.split("_")[-1])
        except ValueError:
            print(f"Skipping folder with invalid seed: {folder_name}")
            continue

        npz_path  = os.path.join(folder_path, f"matrix_data_{seed}.npz")
        json_path = os.path.join(folder_path, f"metadata_{seed}.json")
        xml_path  = os.path.join(folder_path, f"y_parameters_{seed}.xml")

        try:
            with np.load(npz_path) as data:
                entry = {
                    "seed": seed,
                    "folder": folder_path,
                    DSN_OUT_SPEAKER_MATRIX: data[DSN_OUT_SPEAKER_MATRIX],
                    DSN_OUT_ENCODING_MATRIX: data[DSN_OUT_ENCODING_MATRIX],
                    DSN_OUT_TRANSCODING_MATRIX: data[DSN_OUT_TRANSCODING_MATRIX],
                    DSN_OUT_DECODING_MATRIX: data[DSN_OUT_DECODING_MATRIX],
                }
        except Exception as e:
            print(f"Failed to load {npz_path}: {e}")
            continue

        if os.path.exists(xml_path):
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_string  = f.read()
                parsed_xml  = pu.usat_xml_to_dict(xml_string)
                entry["y"]  = parsed_xml

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                metadata    = json.load(f)
                coordinates = pu.restore_coordinates(metadata)
            entry.update(coordinates)
        '''
        energy                  = energy_calculation(entry[DSN_OUT_SPEAKER_MATRIX])
        ang_error, source_width = get_width_and_angular_error(entry[DSN_OUT_CLOUD], 
                                                              entry[DSN_OUT_SPEAKER_MATRIX], 
                                                              entry[DSN_OUT_OUTPUT_LAYOUT])
        q_s, x_p = compute_qs_and_xp(ang_error, source_width, energy)
        entry[DSN_SMPL_QUALITY_SCORE] = round(q_s, 2)
        entry[DSN_SMPL_X_p] = round(x_p, 2)
        '''
        entries.append(entry)

    return pd.DataFrame(entries)

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from matplotlib.image import imread
from io import BytesIO
import base64

def plot_focus_grid(focus_plot_data, metric_settings, colormap=None, dpi=150):
    """
    Display scalar map comparisons across focus groups and metrics using plot_scalar_map().

    Args:
        focus_plot_data (dict): Dict with keys "low", "mid", "high", each containing metric arrays and cloud.
        metric_settings (list): List of dicts with keys:
            - 'key': key in focus_plot_data
            - 'label': label for colorbar
            - 'clim': tuple (min, max)
        colormap (Colormap): Matplotlib colormap to use (LinearSegmentedColormap or str).
        dpi (int): DPI used for plot_scalar_map.
    """

    # Focus labels and metric list
    focus_labels    = list(focus_plot_data.keys())
    num_rows        = len(metric_settings)
    num_cols        = len(focus_labels)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))

    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]

    for row_idx, metric in enumerate(metric_settings):
        metric_key = metric["key"]
        label = metric["label"]
        clim = metric["clim"]

        for col_idx, focus in enumerate(focus_labels):
            data = focus_plot_data[focus]
            cloud = data["cloud"]

            title_text = f"{metric_key.capitalize()} ({focus})"

            img_b64 = plot_scalar_map(
                values=data[metric_key],
                cloud_points=cloud,
                title=label,
                colorbar_label=label,
                clim_range=clim,
                cmap=colormap,
                dpi=dpi,
                return_base64=True
            )

            img_bytes = base64.b64decode(img_b64)
            img_array = imread(BytesIO(img_bytes), format='png')

            ax = axs[row_idx][col_idx]
            ax.imshow(img_array)
            ax.axis('off')

            # Add row labels on the left
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=14)

            # Add column titles at the top
            if row_idx == 0:
                ax.set_title(focus.capitalize(), fontsize=14)

    plt.tight_layout()
    plt.show()

