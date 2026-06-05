import shutil
import matplotlib.pyplot as plt 
from snowpacktools.snowpro import pro_helper, pro_plotter
import sys
sys.path.append("/home/ratha/Prof_comps")
import regobs_evo_comp.helper_scripts as hs
import regobs_evo_comp.comp_helper as ch
import copy
from types import SimpleNamespace
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, AutoDateLocator
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from datetime import datetime, timedelta
import io
import pandas as pd
from typing import List, Any, Optional, Tuple, Union, Dict
import importlib
import string

## Function that flips snow layers and temperatures to match orientation of sim data

def flip_all_layers(obs):
    """
    Return a deep-copied observation with snow_profile.layers reversed
    and snow_profile.temperatures flipped. Works for dicts and attribute objects.
    Works for one observation at a time. To process multiple, use in a loop or comprehension.
    """
    if obs is None:
        return None

    obs_copy = copy.deepcopy(obs)

    snow_profile = ch._get_attr_or_key(obs_copy, "snow_profile", None)
    if snow_profile is None:
        return obs_copy

    # Flip layers
    layers = ch._get_attr_or_key(snow_profile, "layers", None)
    if isinstance(layers, list) and layers:
        flipped_layers = list(reversed(layers))
        ch._set_attr_or_key(snow_profile, "layers", flipped_layers)

    # Flip temperatures
    temps = ch._get_attr_or_key(snow_profile, "temperatures", None)
    if isinstance(temps, list) and temps:
        # helpers to read depth/temp from dict or attribute object
        def read_depth(t):
            return t["depth_cm"] if ch._is_mapping(t) else getattr(t, "depth_cm", None)

        def read_temp(t):
            return t["temp_c"] if ch._is_mapping(t) else getattr(t, "temp_c", None)

        # collect numeric depths if all entries have numeric depth
        depth_vals = []
        for t in temps:
            d = read_depth(t)
            if isinstance(d, (int, float)):
                depth_vals.append(d)
            else:
                depth_vals = []
                break

        if depth_vals:
            max_depth = max(depth_vals)
            flipped = []
            for t in reversed(temps):
                d = read_depth(t)
                temp_val = read_temp(t)
                new_depth = max_depth - d
                if ch._is_mapping(t):
                    flipped.append({"depth_cm": new_depth, "temp_c": temp_val})
                else:
                    flipped.append(SimpleNamespace(depth_cm=new_depth, temp_c=temp_val))
            # sort ascending so surface (0) is first
            flipped.sort(key=lambda x: x["depth_cm"] if ch._is_mapping(x) else getattr(x, "depth_cm", None))
        else:
            # no numeric depths: just reverse and preserve types where possible
            flipped = list(reversed(temps))
            # if original entries were attribute objects, convert dicts to SimpleNamespace
            if any(not ch._is_mapping(t) for t in temps) and any(ch._is_mapping(t) for t in flipped):
                converted = []
                for t in flipped:
                    if ch._is_mapping(t):
                        converted.append(SimpleNamespace(depth_cm=t.get("depth_cm"), temp_c=t.get("temp_c")))
                    else:
                        converted.append(t)
                flipped = converted

        ch._set_attr_or_key(snow_profile, "temperatures", flipped)

    return obs_copy

# Cleaning up false values

def make_cleaned_sims_dir(sims: Tuple[Dict[str, str], Dict[str, str]],
                            out_dir: str,
                            subfolder_name: str = "pro",
                            overwrite: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Create cleaned copies of files referenced by sims inside out_dir/subfolder_name.
    Replace tokens on lines starting with '0534,' that end with .25 -> .0 or .75 -> .5.
    Return sims_cleaned where versions map to the cleaned directory (same path for all keys)
    and stn is returned unchanged.
    """
    versions, stn = sims
    out_base = Path(out_dir)
    pro_dir = out_base / subfolder_name
    pro_dir.mkdir(parents=True, exist_ok=True)

    # Keep track of files already used in this run to avoid duplicates when choosing from a dir
    used_files = set()

    # Write cleaned files into pro_dir
    for key, path_str in versions.items():
        src = Path(path_str)
        chosen = None

        # If src is a directory, choose a file inside (prefer station match, else next unused)
        if src.exists() and src.is_dir():
            files = ch._list_files_in_dir(src)
            if not files:
                print(f"Warning: directory {src} for {key} contains no files; skipping.")
                continue
            station_name = stn.get(key, "")
            station_lower = station_name.lower() if station_name else ""
            # prefer file containing station name and not used yet
            match = None
            for p in files:
                if station_lower and station_lower in p.name.lower() and str(p) not in used_files:
                    match = p
                    break
            if match is None:
                # pick first unused file, else pick first file
                match = next((p for p in files if str(p) not in used_files), files[0])
            chosen = match
        else:
            chosen = src

        if not chosen.exists():
            print(f"Warning: source for {key} not found: {chosen}; skipping.")
            continue

        used_files.add(str(chosen))

        dst = pro_dir / chosen.name
        if dst.exists() and not overwrite:
            # reuse existing cleaned file
            continue

        # read, fix 0534 lines, write
        try:
            with chosen.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {chosen} for {key}: {e}")
            continue

        new_lines = []
        for line in lines:
            stripped = line.rstrip("\n")
            if stripped.startswith("0534,"):
                parts = stripped.split(",")
                if len(parts) >= 3:
                    header = parts[:2]
                    values = parts[2:]
                    fixed_values = [ch._fix_token_ending_25_or_75(v) for v in values]
                    new_line = ",".join(header + fixed_values)
                else:
                    new_line = stripped
                new_lines.append(new_line + "\n")
            else:
                new_lines.append(line)

        try:
            with dst.open("w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"Error writing {dst} for {key}: {e}")
            continue

    # Build versions_cleaned mapping where every value is the directory path (string)
    versions_cleaned = {k: str(out_dir) for k in versions.keys()}

    return (versions_cleaned, dict(stn))

# NA removing function (replaces with nan)
def overwrite_0503_NA(sims_cleaned: Tuple[Dict[str, str], Dict[str, str]],
                                            subfolder: str = "pro",
                                            ext: str = ".pro",
                                            backup: bool = True,
                                            encoding: str = "utf-8") -> Tuple[Tuple[Dict[str, str], Dict[str, str]], List[str]]:
    """
    For sims_cleaned = (versions_dict, stn_dict):
        - expects files at: Path(versions[key]) / subfolder / (stn[key] + ext)
        - replaces tokens exactly 'NA' (any case) or empty with 'nan' on lines starting with '0503,'
        - creates .bak backups for modified files when backup=True
        - returns (new_sims_cleaned, modified_files_list)
    """
    versions, stn = sims_cleaned
    modified_files: List[str] = []

    for key, base_dir in versions.items():
        base = Path(base_dir)
        pro_dir = base / subfolder
        station_name = stn.get(key)
        if station_name is None:
            print(f"Warning: no station name for key {key}; skipping.")
            continue

        target_file = pro_dir / (station_name + ext)
        if not target_file.exists():
            # try case-insensitive extension or alternative names if needed
            alt = None
            if pro_dir.exists() and pro_dir.is_dir():
                # try to find a file whose name contains the station_name (case-insensitive)
                for p in pro_dir.iterdir():
                    if p.is_file() and station_name.lower() in p.name.lower():
                        alt = p
                        break
            if alt:
                target_file = alt
            else:
                print(f"Warning: expected file not found for {key}: {target_file}; skipping.")
                continue

        try:
            text = target_file.read_text(encoding=encoding)
        except Exception as e:
            print(f"Warning: could not read {target_file}: {e}")
            continue

        lines = text.splitlines()
        changed = False
        out_lines: List[str] = []

        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith("0503,"):
                leading = ln[:len(ln) - len(stripped)]
                parts = stripped.split(",")
                if len(parts) >= 3:
                    header = parts[:2]
                    values = parts[2:]
                    fixed_values = []
                    for v in values:
                        vs = v.strip()
                        if vs == "" or vs.upper() == "NA":
                            fixed_values.append("nan")
                            if vs != "nan":
                                changed = True
                        else:
                            fixed_values.append(vs)
                    new_body = ",".join(header + fixed_values)
                    new_line = leading + new_body
                else:
                    new_line = ln
                out_lines.append(new_line)
            else:
                out_lines.append(ln)

        if changed:
            if backup:
                bak = target_file.with_suffix(target_file.suffix + ".bak")
                if not bak.exists():
                    try:
                        shutil.copy2(target_file, bak)
                    except Exception as e:
                        print(f"Warning: could not create backup for {target_file}: {e}")
            try:
                target_file.write_text("\n".join(out_lines) + "\n", encoding=encoding)
                modified_files.append(str(target_file))
                print(f"Sanitized 0503 NAs -> nan in {target_file}")
            except Exception as e:
                print(f"Error writing {target_file}: {e}")
                # attempt restore from backup if available
                if backup:
                    bak = target_file.with_suffix(target_file.suffix + ".bak")
                    if bak.exists():
                        shutil.copy2(bak, target_file)
                        print(f"Restored original from backup for {target_file}")

    # return sims_cleaned unchanged in shape (versions still point to base dir strings)
    new_versions = {k: str(Path(v)) for k, v in versions.items()}
    new_sims = (new_versions, dict(stn))
    return new_sims, modified_files

def plot_regobs_evo(results, color_scheme='SARP',
                    figsize=(12, 6), width_days=None, add_cbar="bottom", ax=None,
                    min_width_days=0.05, max_width_days=30.0, overlap_margin=0.99,
                    x_date="obs", start_date=None, end_date=None, snow_lim=None, title=None,
                    show_compression_tests=False):
    """
    Plot multiple regobslib snow profiles in a single plot on a linear time axis.
    Width auto-adjusts to avoid overlapping unless width_days is explicitly provided.

    Parameters
    ----------
    results : list
        List of regobslib results, each containing a snow_profile and obs_time (datetime).
    color_scheme : str
        Color scheme for grain types.
    figsize : tuple
        Figure size.
    width_days : float or None
        If provided, use this width (in days). If None, compute automatically to avoid overlap.
    add_cbar : str
        Where to add the grain type legend (colorbar). Options: "top", or None.
    min_width_days : float
        Minimum allowed width in days when auto-computing.
    max_width_days : float
        Maximum allowed width in days when auto-computing.
    overlap_margin : float
        Fraction of the available gap to use (0 < overlap_margin <= 1). Use <1 to leave small gaps.
    x_date : str
        "obs": shows x-lables only for the acuatlly plotted dates. 
        "both_bottom" shows both obs and dates at the bottom x-axis.
        "both" shows date on bottom x-axis and obs on top x-axis.
        "compact" simmilar to both but top axis annotation are rotated (use add_cbar="top", except or multi_5)
        "multi_5" use for combined_regobs_and_multi_snp if 5 .pro files are passed along
    start_date : 
        if None the date will be adjusted to the first date in the data
        if date is set, the start date will be set to it if the set date is earlier than in the data
    end_date : 
        if None the date will be adjusted to the last date in the data
        if date is set, the end date will be set to it if the set date is later than in the data 
    snow_lim : set the upper y-limit to a fixed value
    title : str
        set a title name
    show_compression_tests : bool
        whether to show compression tests
    """

    if not results:
        raise ValueError("No results provided.")

    # Prepare grain type colors
    LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, _, _, _ = \
        pro_helper.get_grain_type_colors(color_scheme)
    grain_color_map = dict(zip(LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE))
    grain_hatch_map = dict(zip(LABELS_GRAIN_TYPE, HATCHES_GRAIN_TYPE))

    # Prepare ECT colors
    ECT_COLORS = {
    'ECTPV': '#d73027',  # red   – propagates very easily
    'ECTP':  '#fc8d59',  # orange
    'ECTX':  '#4575b4',  # blue  – no propagation
    'CTM':   '#fee090',
    'CTV':   '#91bfdb',
    'CTN':   '#4575b4',
    'LBT':   '#808080',
    }

    # Create figure / axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Collect valid entries and compute depths
    entries = []
    max_height = 0
    for r in results:
        prof = getattr(r, 'snow_profile', None)
        ts = getattr(r, 'obs_time', None)
        if prof is None or not getattr(prof, 'layers', None) or ts is None:
            continue
        # ensure datetime
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except Exception:
                # skip unparsable strings
                continue
        elif isinstance(ts, datetime):
            ts_dt = ts
        else:
            try:
                ts_dt = ts.to_pydatetime()
            except Exception:
                continue

        xnum = mdates.date2num(ts_dt)
        depth = sum([getattr(l, 'thickness_cm', 0) or 0 for l in prof.layers])
        max_height = max(max_height, depth)
        entries.append((r, prof, ts_dt, xnum))

    if not entries:
        raise ValueError("No valid profiles with timestamps found.")

    # Sort by date (xnum)
    entries.sort(key=lambda e: e[3])
    xnums = np.array([e[3] for e in entries])

    # Auto-compute width_days if not provided
    if width_days is None:
        if len(xnums) == 1:
            # single profile: choose a reasonable default (1 day or min_width_days)
            auto_width = max(min_width_days, min(1.0, max_width_days))
        else:
            # compute gaps between adjacent xnums
            diffs = np.diff(xnums)
            # If any diffs are zero (same timestamp), we need to account for multiple profiles at same x
            if np.any(diffs == 0):
                # count max number of profiles sharing the same xnum (within tiny tolerance)
                unique, counts = np.unique(np.round(xnums, 6), return_counts=True)
                max_same = counts.max()
                # find the smallest non-zero gap to neighboring distinct timestamps (fallback)
                nonzero_diffs = diffs[diffs > 0]
                if nonzero_diffs.size > 0:
                    min_nonzero = nonzero_diffs.min()
                else:
                    # all timestamps identical: spread within one day
                    min_nonzero = 1.0
                # divide available space by number of same-day profiles + small margin
                auto_width = (min_nonzero * overlap_margin) / max_same
            else:
                min_gap = diffs.min()
                auto_width = min_gap * overlap_margin

        # clamp to min/max
        width_days = float(np.clip(auto_width, min_width_days, max_width_days))

    # final half width
    half_width = width_days / 2.0

    # If computed half_width would still overlap because of extremely close points,
    # ensure half_width <= half of the minimum non-zero gap
    if len(xnums) > 1:
        nonzero_gaps = np.diff(xnums)
        nonzero_gaps = nonzero_gaps[nonzero_gaps > 0]
        if nonzero_gaps.size > 0:
            min_nonzero_gap = nonzero_gaps.min()
            max_half_allowed = min_nonzero_gap / 2.0 * overlap_margin
            if half_width > max_half_allowed:
                half_width = max_half_allowed
                width_days = half_width * 2.0
                # enforce min clamp
                if width_days < min_width_days:
                    width_days = min_width_days
                    half_width = width_days / 2.0

    # Plot each profile at its date position
    for (result, prof, ts_dt, xnum) in entries:
        depth = 0
        for layer in prof.layers:
            h = getattr(layer, 'thickness_cm', None)
            if not h or h <= 0:
                continue

            bottom = depth
            top = depth + h

            grain_code = hs._regobs_grain_type_code(layer.grain_form_primary)
            color = grain_color_map.get(grain_code, "white")
            hatch = grain_hatch_map.get(grain_code, "")

            ax.add_patch(
                plt.Rectangle(
                    (xnum - half_width, bottom),  # x, y
                    width_days,                    # width in days
                    h,                             # height in cm
                    facecolor=color,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=0.5,
                    transform=ax.transData
                )
            )

            depth = top

        # label under each profile
        if x_date=="obs" or x_date=="both_bottom":
            label = ts_dt.strftime("%b-%d")
            ax.text(xnum, -max(2, max_height * 0.02), label, ha="center", va="top", fontsize=9)
            if x_date=="obs":
                ax.set_xticks([])
        elif x_date=="both" or x_date=="compact" or x_date=="multi_5":
            if x_date=="both":
                label = ts_dt.strftime("%b-%d")
            else:
                label = ts_dt.strftime("%d.%m")
            ax.text(xnum, 1.02, label,
                    ha="center", va="bottom",
                    fontsize=9,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False, rotation=(30 if x_date=="compact" else 0))
        
        # Compresion test
        if show_compression_tests:
            tests = getattr(result, 'compression_tests', []) or []
            total_depth = sum(getattr(l, 'thickness_cm', 0) or 0 for l in prof.layers)
            for ct in tests:
                fd = getattr(ct, 'fracture_depth_cm', None)
                tr = getattr(ct, 'test_result', None)
                nt = getattr(ct, 'number_of_taps', None)
                if fd is None or total_depth == 0:
                    continue
                h = total_depth - fd  # convert to height from ground
                if h < 0:
                    continue
                tr_name = getattr(tr, 'name', str(tr)) if tr is not None else '?'
                color = ECT_COLORS.get(tr_name.split('(')[0], '#808080')
                # horizontal line across the profile column
                ax.plot([xnum - half_width, xnum + half_width], [h, h],
                        color=color, linewidth=2, zorder=5)
                # label: e.g. "ECTP21"
                label = f"{tr_name}{nt if nt is not None else ''}"
                ax.annotate(label, xy=(xnum, 0), ha='center', va='bottom',
                        fontsize=7, color="black", zorder=6, textcoords="offset points", xytext=(0, -20),
                        rotation=30)


    # x-axis: date formatting and locator
    if not x_date=="obs" or x_date==False:
        ax.xaxis_date()
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b-%d"))
        if x_date=="both_bottom":
            ax.tick_params(axis="x", which="major", pad=19)
        if show_compression_tests:
            ax.tick_params(axis="x", which="major", pad=20)
        
        

    # adding a color legend for grain types
        
    if add_cbar == "top":
        pro_helper.add_custom_legend(
        ax,
        LABELS_GRAIN_TYPE[1:],
        COLORS_GRAIN_TYPE[1:],
        HATCHES_GRAIN_TYPE[1:],
        alpha=1,
        x=0.015, y=1.2, width=0.028, height=0.025, spacing=0.092
        )
    
    elif add_cbar == "bottom":
        pro_helper.add_custom_legend(
        ax,
        LABELS_GRAIN_TYPE[1:],
        COLORS_GRAIN_TYPE[1:],
        HATCHES_GRAIN_TYPE[1:],
        alpha=1,
        x=0.015, y=-0.15 if x_date == "both_bottom" else -0.1, width=0.028, height=0.025, spacing=0.092
        )

    # Axes 
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_tick_params(labelright=True)
    ax.set_ylabel("height / cm")
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Limits
    if snow_lim: y_up = snow_lim
    else: y_up = max_height * 1.05
    ax.set_ylim(0, y_up)
    
    left_num = ch._to_date_num(start_date)
    right_num = ch._to_date_num(end_date)

    if left_num is None:
        left = min(xnums) - half_width
    else:
        left = min(min(xnums) - half_width, left_num)

    if right_num is None:
        right = max(xnums) + half_width
    else:
        right = max(max(xnums) + half_width, right_num)
    ax.set_xlim(left, right)
    # adjusting the pad and set the title
    pad = 20 if x_date == "both" else (30 if x_date == "compact" else 0)
    ax.set_title(title, pad=pad)

    return fig, ax


def plot_regobs_evo_flipped(results, color_scheme='SARP',
                            figsize=(12, 6), add_cbar="bottom", ax=None, x_date="obs",
                            start_date=None, end_date=None, snow_lim=None, title=None,
                            show_compression_tests=False):
    """
    Flips all snow profiles (layers + temperatures) before plotting
    a time series evolution plot. (Note: temperature is not shown in the plot)

    Parameters
    ----------
    results : list
        List of regobslib results.
    color_scheme : str
        Grain color scheme.
    figsize : tuple
        Figure size.
    x_spacing : float
        Horizontal spacing between profiles.
    add_cbar : str
        Where to add the grain type legend (colorbar). Options: "ax" (inside axes), "right", "bottom", or None for no legend.
    ax : 
    title : str
        Sets a title name
    """

    if not results:
        raise ValueError("No results provided.")

    # Flip all observations
    flipped_results = [flip_all_layers(r) for r in results]

    # If an axes was provided, let the underlying function draw into it.
    if ax is not None:
        fig = ax.figure
        # assume plot_regobs_evo can accept an ax argument; if not, refactor it similarly
        plot_regobs_evo(flipped_results, color_scheme=color_scheme,
                        figsize=figsize, add_cbar=add_cbar, ax=ax, x_date=x_date,
                        start_date=start_date, end_date=end_date, snow_lim=snow_lim, title=title,
                        show_compression_tests=show_compression_tests)
        return fig, ax

    # fallback: original behavior (create its own figure)
    fig, ax = plot_regobs_evo(flipped_results, color_scheme=color_scheme,
                                figsize=figsize, add_cbar=add_cbar, x_date=x_date,
                                start_date=start_date, end_date=end_date, snow_lim=snow_lim, title=title,
                                show_compression_tests=show_compression_tests)
    return fig, ax


def fix_pro_metadata(input_path: str, output_dir: str, metadata=None):
    """
    Reads a PRO file, fixes NA metadata fields based on filename rules,
    and writes a corrected PRO file to output_dir.
    If metadata is passed along the function will use this metadata.
    Example: metadata = {
    "StationName": "snower-station",
    "Latitude": None,
    "Longitude": None,
    "Altitude": 1290,
    "SlopeAngle": 15,
    "SlopeAzi": 85
    }

    Filename format example:
        1_1200-1500_F.pro

    Rules:
    - Region index = first number before '_'
    - Height range = two numbers between '_'
    - Aspect = final letter before .pro (F, N, E, S, W)
    - StationName:
        region 1 → aggregated_Stong
        region 2 -> aggregated_Kalvavatni
        else    → aggregated_<region>
    - Lat/Lon:
        region 1 → 60.9 / 8.1
        else     → leave NA
    - Altitude:
        mean of height range
    - SlopeAngle:
        F → 0
        else → 38
    - SlopeAzi:
        F or N → 0
        E → 90
        S → 180
        W → 270
    """
    
    input_path = Path(input_path)
    filename = input_path.name
    base = filename.replace(".pro", "")

    if not metadata:
        metadata = {
            "StationName": None,
            "Latitude": None,
            "Longitude": None,
            "Altitude": None,
            "SlopeAngle": None,
            "SlopeAzi": None,
        }
        # Parse filename only when metadata wasn't supplied
        region_str, height_str, aspect = base.split("_")
        region = int(region_str)
        h1, h2 = map(int, height_str.split("-"))
        mean_alt = int((h1 + h2) / 2)
        

    # Read file
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Prepare output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Process lines
    new_lines = []
    for line in lines:
        if line.startswith("StationName="):
            if metadata["StationName"]:
                line = f"StationName= {metadata["StationName"]}\n"
            elif "NA" in line:
                if region == 1:
                    line = "StationName= aggregated_Stong\n"
                else:
                    line = f"StationName= aggregated_{region}\n"

        elif line.startswith("Latitude="):
            if metadata["Latitude"]:
                line = f"Latitude= {metadata["Latitude"]}\n"
            elif "NA" in line:
                if region == 1:
                    line = "Latitude= 60.9\n"

        elif line.startswith("Longitude="):
            if metadata["Longitude"]:
                line = f"Longitude= {metadata["Longitude"]}\n"
            elif "NA" in line:
                if region == 1:
                    line = "Longitude= 8.1\n"

        elif line.startswith("Altitude="):
            if metadata["Altitude"]:
                line = f"Altitude= {metadata["Altitude"]}\n"
            elif "NA" in line:
                line = f"Altitude= {mean_alt}\n"

        elif line.startswith("SlopeAngle="):
            if metadata["SlopeAngle"]:
                line = f"SlopeAngle= {metadata["SlopeAngle"]}\n"
            elif "NA" in line:
                if aspect == "F":
                    line = "SlopeAngle= 0\n"
                elif aspect in ["E", "S", "W", "N"]:
                    line = "SlopeAngle= 38\n"

        elif line.startswith("SlopeAzi="):
            if metadata["SlopeAzi"]:
                line = f"SlopeAzi= {metadata["SlopeAzi"]}\n"
            if "NA" in line:
                if aspect in ["F", "N"]:
                    line = "SlopeAzi= 0\n"
                elif aspect == "E":
                    line = "SlopeAzi= 90\n"
                elif aspect == "S":
                    line = "SlopeAzi= 180\n"
                elif aspect == "W":
                    line = "SlopeAzi= 270\n"


        new_lines.append(line)

    # Write new file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return output_path

def combined_regobs_and_multi_snps(results=None, pro_paths=None, out_path=None,
                                figsize=(10, 12),
                                regobs_kwargs=None, snp_kwargs=None,
                                height_ratios=None, hspace=0.1,
                                regobs_rgb=None, snp_rgbs=None, titles=None, header=None):
    """
    Stack a single plot_regobs_evo_flipped (top) and one or more SNP evolutions
    (rendered from .pro files) underneath, each as its own horizontal panel.
    Uses the updated version of pro_plotter (added optional out_path=False and legend=False)
    
    Example Use
    -----------
    fig, (ax_top, ax_bottoms) = ch.combined_regobs_and_multi_snps(
    results=regobs,
    pro_paths=pro_paths,
    out_path=None,
    figsize=(10, 8),
    regobs_kwargs={"color_scheme":"SARP", "add_cbar":"top", "x_date":"compact"},
    snp_kwargs={"var":"grain_type", "res":"1D", "color_scheme":"SARP", "legend":"False",},
    titles=titles)

    Parameters
    ----------
    results : list or None
        Data for plot_regobs_evo_flipped (ignored if regobs_rgb provided).
    pro_paths : str or list[str] or None
        Path or list of paths forwarded to _render_snp_evo_to_rgb (and used to
        extract dates). If a single string is provided it will be treated as a
        single-item list.
    out_path : str or None
        If provided, save the combined figure to this path.
    regobs_kwargs : dict or None
        Extra kwargs forwarded to plot_regobs_evo_flipped. 
        !! If start_date or end_date is set it will be used for all plots!!
    snp_kwargs : dict or None
        Extra kwargs forwarded to _render_snp_evo_to_rgb. If a dict is provided
        it will be used for all .pro files; to pass different kwargs per .pro,
        provide a list of dicts matching pro_paths length.
    height_ratios : tuple or list or None
        Height ratios for the stacked panels. If None, defaults to (1, 1, 1, ...)
        with one top panel and one panel per .pro file.
    hspace : float
        Vertical spacing between subplots.
    regobs_rgb : ndarray or None
        Pre-rendered image for the top panel (optional).
    snp_rgbs : ndarray or list[ndarray] or None
        Pre-rendered images for bottom panels (optional). If a single ndarray
        is provided and multiple pro_paths exist, it will be used for the first
        bottom panel only.
    titles : list or None
        Giving titles for the snp plots (regobs gets a title passed along from its parent function)
    header : str or None
        sets the main title for all the plots
        
    Returns
    -------
    fig, axes
        Matplotlib figure and tuple (ax_top, [ax_bottom1, ax_bottom2, ...]).
    """
    regobs_kwargs = dict(regobs_kwargs or {})
    # Normalize pro_paths to list
    if pro_paths is None:
        pro_paths_list = []
    elif isinstance(pro_paths, (list, tuple)):
        pro_paths_list = list(pro_paths)
    else:
        pro_paths_list = [pro_paths]

    n_snps = len(pro_paths_list)
    # Normalize snp_kwargs: allow single dict or list of dicts
    if snp_kwargs is None:
        snp_kwargs_list = [{} for _ in range(n_snps)]
    elif isinstance(snp_kwargs, (list, tuple)):
        if len(snp_kwargs) != n_snps:
            raise ValueError("If snp_kwargs is a list it must match the number of pro_paths.")
        snp_kwargs_list = list(snp_kwargs)
    else:
        snp_kwargs_list = [dict(snp_kwargs) for _ in range(n_snps)]

    # Normalize snp_rgbs to list (may be None entries)
    if snp_rgbs is None:
        snp_rgbs_list = [None] * n_snps
    elif isinstance(snp_rgbs, (list, tuple)):
        if len(snp_rgbs) != n_snps:
            raise ValueError("If snp_rgbs is a list it must match the number of pro_paths.")
        snp_rgbs_list = list(snp_rgbs)
    else:
        # single ndarray provided -> use for first bottom panel
        snp_rgbs_list = [snp_rgbs] + [None] * (n_snps - 1)

    importlib.reload(ch)
    # Calculate maximum snow height across regobs results and all pro files
    max_regobs_h = ch.max_height_from_results(results) or 0.0
    max_snp_h = 0.0
    for p in pro_paths_list:
        try:
            h = ch.max_height_from_pro(p) or 0.0
        except Exception:
            h = 0.0
        if h > max_snp_h:
            max_snp_h = h
    max_snow = max(max_regobs_h, max_snp_h) * 1.1

    # Add the max to regobs_kwargs and each snp_kwargs
    regobs_kwargs = {**(regobs_kwargs or {}), 'snow_lim': max_snow}
    for i in range(n_snps):
        snp_kwargs_list[i] = {**(snp_kwargs_list[i] or {}), 'height_max': max_snow}

    # If user didn't provide start/end in regobs_kwargs, try to extract from first pro file
    
    if pro_paths_list and ('start_date' not in regobs_kwargs or 'end_date' not in regobs_kwargs):
        use_st = True
        use_end = True
        start_dt, end_dt = ch._extract_start_end_from_pro_0500(pro_paths_list[0])
        if start_dt is not None and 'start_date' not in regobs_kwargs:
            use_st = False
            regobs_kwargs['start_date'] = start_dt
        if end_dt is not None and 'end_date' not in regobs_kwargs:
            use_end = False
            regobs_kwargs['end_date'] = end_dt

    # Validate inputs
    if regobs_rgb is None and results is None:
        raise ValueError("Either 'regobs_rgb' or 'results' must be provided for the top panel.")
    if n_snps == 0 and all(x is None for x in snp_rgbs_list):
        raise ValueError("At least one 'pro_path' or one 'snp_rgb' must be provided for bottom panels.")

    # Build height_ratios: one for top + one per snp
    if height_ratios is None:
        height_ratios_final = [1] * (1 + n_snps)
    else:
        # Accept tuple/list or single value
        if isinstance(height_ratios, (list, tuple)):
            if len(height_ratios) != 1 + n_snps:
                raise ValueError("height_ratios must have length 1 + number of pro files.")
            height_ratios_final = list(height_ratios)
        else:
            # single numeric -> top gets that, others 1
            height_ratios_final = [height_ratios] + [1] * n_snps

    # Adjust figure height proportional to number of panels if user passed default
    if figsize is None:
        figsize = (10, 3 * (1 + n_snps))
    else:
        # If user provided a single height, scale it by number of panels
        if len(figsize) == 2:
            base_w, base_h = figsize
            figsize = (base_w, base_h * (1 + n_snps) / 2)

    # Create figure and axes
    nrows = 1 + n_snps
    fig, axes = plt.subplots(nrows, 1, figsize=figsize,
                                gridspec_kw={'height_ratios': height_ratios_final})
    fig.subplots_adjust(hspace=hspace)

    # Normalize axes to always be a list: axes[0] is top
    if nrows == 1:
        axes = [axes]
    ax_top = axes[0]
    ax_bottoms = axes[1:] if n_snps > 0 else []

    # Top panel: regobs
    if regobs_rgb is not None:
        ax_top.imshow(regobs_rgb, aspect='auto')
        ax_top.axis('off')
    else:
        plot_regobs_evo_flipped(results, ax=ax_top, **regobs_kwargs)

    # Bottom panels: one per .pro file or pre-rendered rgb
    for i in range(n_snps):
        ax = ax_bottoms[i]
        if snp_rgbs_list[i] is not None:
            ax.axis('off')
            ax.imshow(snp_rgbs_list[i], aspect='auto')
        else:
            pro_plotter.snp_evo(pro_paths_list[i], **snp_kwargs_list[i], ax_sub=ax_bottoms[i], out_path=False)
            
    if titles:
        # set bottom titles
        for i, ax in enumerate(ax_bottoms):
            title = titles[i]
            ax.text(
                0.5, 0.96, title,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    # enumerate subplots
    letters = list(string.ascii_lowercase)
    for i, ax in enumerate(axes):
        ax.annotate(
            f"{letters[i]})",
            xy=(0.95, 0.96),
            xycoords='axes fraction',
            ha='right', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    # Synchronise x-axis limits across all panels
    all_axes = [ax_top] + list(ax_bottoms)

    # get per-axis limits as (left, right) with left <= right
    xlims = []
    for ax in all_axes:
        a, b = ax.get_xlim()
        xlims.append((min(a, b), max(a, b)))

    # choose xmin and xmax according to flags
    if use_st:
        # take left edge from ax_top only
        a, b = ax_top.get_xlim()
        xmin = min(a, b)
    else:
        xmin = min(x[0] for x in xlims)

    if use_end:
        # take right edge from ax_top only
        a, b = ax_top.get_xlim()
        xmax = max(a, b)
    else:
        xmax = max(x[1] for x in xlims)

    # apply to all axes
    for ax in all_axes:
        ax.set_xlim(xmin, xmax)
        
    # header
    if header:
        fig.suptitle(header, fontsize=13, y=1.01)
    

    if out_path:
        fig.savefig(out_path, dpi=300, facecolor='w', edgecolor='w')

    plt.show()
    return fig, (ax_top, ax_bottoms)

def plot_regobs_min_max(regobs_list: List[Any],
                        ax: Optional[plt.Axes] = None,
                        date_parser: Optional[callable] = None,
                        marker="x", linestyle="-") -> Tuple[Optional[plt.Figure], plt.Axes]:
    """
    Plot min and max temperature time series from a list of regobs results.

    Each item in regobs_list is passed to `extract_regobs_tmp(item)` which must
    be available in the same namespace and return a dict with keys:
        - 'date' (datetime or string)
        - 'min_temp_c'
        - 'max_temp_c'
    
    marker : 
        sets the style of the marker in the plot
    linestyle :
        sets the linestyle for the plot 

    Returns (fig, ax). If an external ax is provided, fig is None.
    """
    entries = []
    for item in regobs_list:
        try:
            info = ch.extract_regobs_tmp(item)
        except Exception:
            info = None
        if not info:
            continue
        entries.append({
            'date': info.get('date'),
            'min_temp_c': info.get('min_temp_c'),
            'max_temp_c': info.get('max_temp_c')
        })

    if not entries:
        raise ValueError("No valid regobs entries to plot.")

    df = pd.DataFrame(entries)

    # Robust date parsing to UTC
    def _parse_date(val):
        if val is None:
            return pd.NaT
        if isinstance(val, datetime):
            return pd.to_datetime(val, utc=True)
        s = str(val)
        if date_parser is not None:
            try:
                parsed = date_parser(s)
                return pd.to_datetime(parsed, utc=True)
            except Exception:
                pass
        try:
            from dateutil import parser as _dparser  # type: ignore
            parsed = _dparser.parse(s)
            return pd.to_datetime(parsed, utc=True)
        except Exception:
            return pd.to_datetime(s, errors='coerce', utc=True)

    df['date'] = df['date'].apply(_parse_date)
    df = df.dropna(subset=['date'])
    if df.empty:
        raise ValueError("No valid dates parsed from regobs entries.")

    # Creating the min and max data frame
    df['min_temp_c'] = pd.to_numeric(df['min_temp_c'], errors='coerce')
    df['max_temp_c'] = pd.to_numeric(df['max_temp_c'], errors='coerce')

    if df['min_temp_c'].dropna().empty and df['max_temp_c'].dropna().empty:
        raise ValueError("No valid temperature values in regobs entries.")

    created_fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        created_fig = fig

    # Sort by date
    df = df.sort_values('date')

    # Plot min and max
    ax.plot(df['date'], df['max_temp_c'], marker=marker, linestyle=linestyle, color='#d62728', label='max')
    ax.plot(df['date'], df['min_temp_c'], marker=marker, linestyle=linestyle, color='#1f77b4', label='min')

    # Formatting
    #ax.set_xlabel('Date')
    ax.set_ylabel('Snow Temperature (°C)')
    ax.set_title("Regobs")
    ax.grid(True, linestyle=':', alpha=0.6)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()

    return (created_fig, ax)

def plot_pro_min_max(pro_source: Union[str, None],
                        from_file: bool = True,
                        ax: Optional[plt.Axes] = None,
                        date_parser: Optional[callable] = None,
                        marker="x", linestyle="-",
                        custom_name="") -> Tuple[Optional[plt.Figure], plt.Axes]:
    """
    Plot min and max temperature time series from a single .pro source.

    pro_source may be a file path (from_file=True) or a string with the .pro contents
    (from_file=False). This function relies on `extract_pro_tmp(source, from_file)`
    being available in the same namespace and returning a list of dicts with keys:
        - 'profile_index', 'timestamp', 'min_temp_c', 'max_temp_c', ...
    """
    if pro_source is None:
        raise ValueError("pro_source must be provided (file path or raw .pro text).")

    try:
        profiles = ch.extract_pro_tmp(pro_source, from_file=from_file)
    except Exception as e:
        raise RuntimeError(f"Failed to parse .pro source with extract_pro_tmp: {e}")

    entries = []
    for p in profiles:
        ts = p.get('timestamp')
        # skip header-like timestamps such as literal 'Date' or empty
        if ts is None:
            continue
        if isinstance(ts, str) and ts.strip().lower() == 'date':
            continue
        entries.append({
            'date': ts,
            'min_temp_c': p.get('min_temp_c'),
            'max_temp_c': p.get('max_temp_c')
        })

    if not entries:
        raise ValueError("No valid profiles found in .pro source to plot.")

    df = pd.DataFrame(entries)

    # Robust date parsing
    def _parse_date(val):
        if val is None:
            return pd.NaT
        if isinstance(val, datetime):
            return pd.to_datetime(val, utc=True)
        s = str(val).strip()
        try:
            # explicit dayfirst format for "04.10.2025 12:00:00"
            return pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S", utc=True)
        except Exception:
            # fallback to dateutil
            from dateutil import parser as _dparser
            try:
                return pd.to_datetime(_dparser.parse(s, dayfirst=True), utc=True)
            except Exception:
                return pd.to_datetime(s, errors='coerce', utc=True)

    # Using the date
    df['date'] = df['date'].apply(_parse_date)

    df = df.dropna(subset=['date'])
    if df.empty:
        raise ValueError("No valid dates parsed from .pro profiles.")

    # Calculating the min and max temps
    df['min_temp_c'] = pd.to_numeric(df['min_temp_c'], errors='coerce')
    df['max_temp_c'] = pd.to_numeric(df['max_temp_c'], errors='coerce')

    if df['min_temp_c'].dropna().empty and df['max_temp_c'].dropna().empty:
        raise ValueError("No valid temperature values in .pro profiles.")

    created_fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        created_fig = fig

    # Sorting along the time
    df = df.sort_values('date')

    # Actual plot
    ax.plot(df['date'], df['max_temp_c'], marker=marker, linestyle=linestyle, color='#d62728', label='max')
    ax.plot(df['date'], df['min_temp_c'], marker=marker, linestyle=linestyle, color='#1f77b4', label='min')

    # Setting lables
    #ax.set_xlabel('Date')
    ax.set_ylabel('SnowTemperature (°C)')
    
    # Plot title
    # If custom name is parsed choose the custom name
    if custom_name: 
        ax.set_title(custom_name)
    # else use the file name as title
    else:
        ax.set_title(ch._pro_title(pro_source, from_file=from_file))
    ax.grid(True, linestyle=':', alpha=0.6)

    # Ticklables for the x axis
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()

    return (created_fig, ax)


def plot_regobs_and_pros_tmp(regobs_list: List[Any],
                                pro_sources: List[Union[str, None]],
                                from_file: bool = True,
                                date_parser: Optional[callable] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                sharex: bool = True,
                                suptitle: Optional[str] = None,
                                marker_pro = "x",
                                custom_name = [],
                                ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a figure with the regobs min/max plot on top and one .pro subplot per item in pro_sources.

    Parameters
    - regobs_list: list passed to plot_regobs_min_max
    - pro_sources: list of file paths or raw .pro text (each gets its own subplot)
    - from_file: forwarded to plot_pro_min_max
    - date_parser: forwarded to both plotting functions
    - figsize: optional figure size; if None it's computed from number of subplots
    - sharex: whether subplots share the x-axis (default True)
    - suptitle: optional figure title
    - marker: string be passed on to the pro plotting function, for the regobs marker is always set to "x"
    - custom_name: list with custom names for the .pro plot titles

    Returns (fig, axes_list) where axes_list[0] is the regobs Axes and subsequent entries are .pro Axes in the same order as pro_sources.
    """
    n_pro = len(pro_sources) if pro_sources is not None else 0
    n_rows = 1 + n_pro  # one for regobs + one per pro file

    # sensible default figsize if not provided
    if figsize is None:
        figsize = (10, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=sharex)
    # ensure axes is always a list-like
    if n_rows == 1:
        axes = [axes]
    else:
        axes = list(axes)

    ax_regobs = axes[0]
    # Plot regobs on the top axis
    plot_regobs_min_max(regobs_list, ax=ax_regobs, date_parser=date_parser, marker="x")

    pro_axes = []
    for i, src in enumerate(pro_sources):
        ax = axes[i + 1]
        plot_pro_min_max(src, from_file=from_file, ax=ax, date_parser=date_parser, marker=marker_pro, custom_name=(custom_name[i] if custom_name else None))
        pro_axes.append(ax)

    # Collect all axes for convenience
    all_axes = [ax_regobs] + pro_axes

    # Compute combined y-limits across all axes and apply to each
    y_mins = []
    y_maxs = []
    for ax in all_axes:
        ymin, ymax = ax.get_ylim()
        y_mins.append(ymin)
        y_maxs.append(ymax)
    combined_ymin = min(y_mins)
    combined_ymax = max(y_maxs)
    for ax in all_axes:
        ax.set_ylim(combined_ymin, combined_ymax)

    # Compute combined x-limits and apply to each (keeps time range identical)
    x_mins = []
    x_maxs = []
    for ax in all_axes:
        xmin, xmax = ax.get_xlim()
        x_mins.append(xmin)
        x_maxs.append(xmax)
    combined_xmin = min(x_mins)
    combined_xmax = max(x_maxs)
    for ax in all_axes:
        ax.set_xlim(combined_xmin, combined_xmax)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    
    # enumerate subplots
    letters = list(string.ascii_lowercase)
    for i, ax in enumerate(axes):
        ax.annotate(
            f"{letters[i]})",
            xy=(0.95, 0.96),
            xycoords='axes fraction',
            ha='right', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    for ax in all_axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelrotation=30, labelbottom=True)  # ensure labels are shown
    plt.tight_layout()

    # Optional shared title and layout
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.003)
    else:
        plt.tight_layout()

    return fig, all_axes


def find_timestamps_below_threshold(
    source: Union[str, io.TextIOBase],
    threshold: float,
    temp_code: str = '0503'
) -> List[Dict]:
    """
    Scan a .pro file (path or string or file-like) and return timestamps
    where any temperature value (line code temp_code) is < threshold.

    Parameters
    - source: file path (str) OR file content (str) OR file-like object
    - threshold: numeric threshold (temperatures strictly less than this)
    - temp_code: the data code for temperature lines (default '0503')

    Returns: list of dicts, each dict:
        {
        'timestamp': '01.09.2025 03:00:00',
        'temp_values': [ ... all parsed temps ... ],
        'below': [(index1, value1), (index2, value2), ...]  # 1-based element indices
        }
    """
    # Read content
    if hasattr(source, 'read'):
        content = source.read()
    else:
        try:
            # treat as file path
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            # treat as raw content string
            content = str(source)

    lines = content.splitlines()

    results = []
    current_timestamp = None
    # We'll collect lines belonging to the current record until next 0500
    record_lines = []

    def process_record(ts: str, rec_lines: List[str]):
        # find the first line that starts with temp_code + ','
        temp_line = None
        for ln in rec_lines:
            if ln.strip().startswith(temp_code + ','):
                temp_line = ln
                break
        if temp_line is None:
            return None
        temps = ch._parse_values_from_line(temp_line)
        below = []
        for i, val in enumerate(temps, start=1):
            if val < threshold:
                below.append((i, val))
        if below:
            return {'timestamp': ts, 'temp_values': temps, 'below': below}
        return None

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith('0500,'):
            # process previous record
            if current_timestamp is not None:
                rec_result = process_record(current_timestamp, record_lines)
                if rec_result:
                    results.append(rec_result)
            # start new record
            # timestamp is the remainder after '0500,'
            parts = stripped.split(',', 1)
            current_timestamp = parts[1].strip() if len(parts) > 1 else ''
            record_lines = [stripped]
        else:
            # accumulate lines for current record (only if we've seen a timestamp)
            if current_timestamp is not None:
                record_lines.append(stripped)

    # process last record
    if current_timestamp is not None:
        rec_result = process_record(current_timestamp, record_lines)
        if rec_result:
            results.append(rec_result)

    return results

def round_pro_timestamps(input_path, output_path):
    """
    Reads a .PRO file and rounds all 0500 timestamps to the nearest full hour.
    
    Arguments:
        input_path (str):   Path to the input .PRO file
        output_path (str):  Path to write the modified .PRO file, if no path is specified the file will be overwritten
    """
    if not output_path: 
        output_path = input_path
    
    def round_to_nearest_hour(dt):
        if dt.minute >= 30:
            dt = dt.replace(second=0, microsecond=0, minute=0) + timedelta(hours=1)
        else:
            dt = dt.replace(second=0, microsecond=0, minute=0)
        return dt

    with open(input_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    for line in lines:
        if line.startswith('0500') and ',' in line:
            parts = line.strip().split(',', 1)
            try:
                dt = datetime.strptime(parts[1].strip(), '%d.%m.%Y %H:%M:%S')
                dt_rounded = round_to_nearest_hour(dt)
                line = f"0500,{dt_rounded.strftime('%d.%m.%Y %H:%M:%S')}\n"
            except ValueError:
                pass  # leave header line (0500,Date) untouched
        output_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output_lines)