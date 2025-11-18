import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import rcParams


def setup_plot_style(
    figure_bg="#1E1E1E",
    axes_bg="#1E1E1E",
    text_color="white"
):
    rcParams['axes.titlesize'] = 18
    rcParams['axes.grid'] = False
    rcParams['grid.alpha'] = 0.3
    rcParams['axes.edgecolor'] = 'gray'
    rcParams['axes.linewidth'] = 1
    rcParams['figure.facecolor'] = figure_bg
    rcParams['axes.facecolor'] = axes_bg
    rcParams['text.color'] = text_color
    rcParams['axes.labelcolor'] = text_color
    rcParams['xtick.color'] = text_color
    rcParams['ytick.color'] = text_color
    rcParams['axes.titlecolor'] = text_color
    rcParams['legend.fontsize'] = 12
    rcParams['legend.frameon'] = False


setup_plot_style()


def plot_speaker_layout_cartopy(
    points,
    *,
    projection="Mollweide",
    hemisphere=True,
    symmetry_axis=None,
    title="Speaker Layout",
    show_grid=True,
    grid_step_az=30,
    grid_step_el=30,
    marker_size=80,
    marker_face="crimson",
    marker_edge="black"
):
    """
    Robust spherical speaker layout plot using Cartopy.
    Supports multiple projections and hemisphere masking.
    """

    pts = np.asarray(points)
    if pts.ndim == 3 and pts.shape[0] == 1:
        pts = pts[0]
    if pts.shape[1] < 2:
        raise ValueError("points must have azimuth and elevation columns")

    az = pts[:, 0].astype(float)
    el = pts[:, 1].astype(float)
    az = ((az + 180) % 360) - 180  # Normalize to [-180, 180]

    if hemisphere:
        mask = el >= 0.0
        az, el = az[mask], el[mask]

    # Projection setup
    proj_map = {
        "Mollweide": ccrs.Mollweide(),
        "Robinson": ccrs.Robinson(),
        "PlateCarree": ccrs.PlateCarree(),
        "Orthographic": ccrs.Orthographic(central_longitude=0, central_latitude=20),
    }
    proj = proj_map.get(projection, ccrs.Mollweide())

    # Create figure
    fig = plt.figure(figsize=(8.0, 4.2) if projection != "Orthographic" else (6.0, 6.0))
    ax = plt.axes(projection=proj)
    ax.set_title(title)

    # Extent configuration
    if hemisphere:
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    else:
        # Not all projections support set_global()
        try:
            ax.set_global()
        except Exception:
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # --- Gridlines (aligned with reference degrees) ---
    if show_grid:
        az_lines = [-150, -90, -30, 0, 30, 90, 150]
        el_lines = [0, 20, 45, 70]
        ax.gridlines(
            draw_labels=False,
            xlocs=az_lines,
            ylocs=el_lines,
            linewidth=0.6,
            color="gray",
            alpha=0.5
        )

    # --- Symmetry planes ---
    if symmetry_axis:
        sym = symmetry_axis.lower()
        if sym == "x":   # azimuth 0°
            ax.plot([0, 0], [-90, 90], transform=ccrs.PlateCarree(),
                    linestyle="--", color="dodgerblue", lw=1.2, label="Az=0°")
        elif sym == "y": # azimuth 90°
            ax.plot([90, 90], [-90, 90], transform=ccrs.PlateCarree(),
                    linestyle="--", color="dodgerblue", lw=1.2, label="Az=90°")
        elif sym == "z": # elevation 0°
            ax.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(),
                    linestyle="--", color="dodgerblue", lw=1.2, label="El=0°")

    # --- Speaker points ---
    ax.scatter(
        az, el,
        s=marker_size,
        facecolors=marker_face,
        edgecolors=marker_edge,
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # --- Reference labels ---
    az_ref = [0, 30, 90, 150]
    el_ref = [0, 20, 45, 70]

    # Azimuth labels (bottom, along equator)
    for a in az_ref:
        ax.text(a, -5, f"{a}°", color="white", fontsize=10,
                ha="center", va="top", transform=ccrs.PlateCarree(), zorder=6)
        ax.text(-a, -5, f"{-a}°", color="white", fontsize=10,
                ha="center", va="top", transform=ccrs.PlateCarree(), zorder=6)

    # --- Elevation labels (projected just outside the globe edges, with fine offsets) ---
    for e in el_ref:
        # Compute projected coordinates at ±180° for this elevation
        x_right, y_right = proj.transform_point(180, e, src_crs=ccrs.PlateCarree())
        x_left,  y_left  = proj.transform_point(-180, e, src_crs=ccrs.PlateCarree())

        # Horizontal and vertical offsets (4% and 1% of projected height)
        if e == 70:
            x_offset = abs(x_right) * 0.1
        else:
            x_offset = abs(x_right) * 0.02

        y_offset = abs(y_right) * 0.01

        # Place text slightly outside the globe edge
        ax.text(x_right + x_offset, y_right + y_offset, f"{e}°",
                color="white", fontsize=9,
                ha="left", va="center",
                transform=proj,
                zorder=6, clip_on=False)
        ax.text(x_left - x_offset, y_left - y_offset, f"{e}°",
                color="white", fontsize=9,
                ha="right", va="center",
                transform=proj,
                zorder=6, clip_on=False)

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
