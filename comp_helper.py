import re
import sys
sys.path.append("/home/ratha/Prof_comps")
from pathlib import Path
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timezone
import io
from typing import List, Any, Optional, Union, Dict
import os

## helper functions for the flipping function
def _is_mapping(obj: Any) -> bool:
    return isinstance(obj, dict)

def _get_attr_or_key(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if _is_mapping(obj):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _set_attr_or_key(obj: Any, name: str, value):
    if _is_mapping(obj):
        obj[name] = value
    else:
        setattr(obj, name, value)
        
## helper functions for the make_clean_sims_dir
# match integer part and either .25 or .75 (optionally followed by zeros)
_RE_ENDS_25_75 = re.compile(r'^([+-]?\d+)\.(25|75)(?:0*)$')

def _fix_token_ending_25_or_75(token: str) -> str:
    """
    Convert tokens ending with .25 -> .0 and .75 -> .5, preserving sign.
    Examples:
        '-1.25' -> '-1.0'
        '3.750' -> '3.5'
        '+12.25' -> '+12.0' (keeps sign if present in token string)
    If token doesn't match, return original stripped token.
    """
    t = str(token).strip()
    m = _RE_ENDS_25_75.match(t)
    if not m:
        return t
    intpart, frac = m.group(1), m.group(2)
    if frac == "25":
        return intpart + ".0"
    else:  # frac == "75"
        return intpart + ".5"

def _list_files_in_dir(dirpath: Path):
    return sorted([p for p in dirpath.iterdir() if p.is_file()])

## helper function for the regobs evo
def _to_date_num(val):
    """Convert val to a matplotlib date number or return None for False/None."""
    if val in (False, None):
        return None
    # already a matplotlib date number
    if isinstance(val, (int, float)):
        return float(val)
    # parse ISO-like string
    if isinstance(val, str):
        dt = datetime.fromisoformat(val)
    elif isinstance(val, datetime):
        dt = val
    else:
        raise TypeError("start_date/end_date must be False, datetime, ISO string, or date number")

    # normalize tz-aware datetimes to UTC naive (recommended)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return mdates.date2num(dt)

## helper functions for combined_regobs_and_multi_snp
def _extract_start_end_from_pro_0500(pro_path):
    """
    Extract datetimes from lines starting with '0500,' in a .pro file.
    Expected date format in DATA section: 'DD.MM.YYYY HH:MM:SS' or 'DD.MM.YYYY'.
    Returns (start_dt, end_dt) as naive datetimes (UTC-normalized if tz present),
    or (None, None) if no dates found.
    """
    date_re = re.compile(r'^\s*0500\s*,\s*(.+)$', re.MULTILINE)
    try:
        with open(pro_path, 'r', encoding='utf-8') as fh:
            text = fh.read()
    except Exception:
        return None, None

    matches = date_re.findall(text)
    if not matches:
        return None, None

    parsed = []
    for s in matches:
        s = s.strip()
        # common formats in your example: "04.10.2025 12:00:00" or "04.10.2025"
        for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                parsed.append(dt)
                break
            except Exception:
                continue
        else:
            # try ISO fallback
            try:
                dt = datetime.fromisoformat(s)
                parsed.append(dt)
            except Exception:
                # ignore unparsable strings
                continue

    if not parsed:
        return None, None

    start_dt = min(parsed)
    end_dt = max(parsed)

    # normalize tz-aware datetimes to UTC naive for consistency
    if start_dt.tzinfo is not None:
        start_dt = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
    if end_dt.tzinfo is not None:
        end_dt = end_dt.astimezone(timezone.utc).replace(tzinfo=None)

    return start_dt, end_dt

def max_height_from_results(results):
    """
    Inspect results and return maximum height (in same units your plotting uses).
    Adjust the attribute access to match your result structure.
    Example assumes each result has a list of layer top/bottom positions in cm:
        result['layers'] -> list of dicts with 'top' and 'bottom' (cm)
    """
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
    return max_height

def max_height_from_pro(pro_path, snow_pos=True):
    """
    Parse .pro file for 0501 lines (height positions). Returns max absolute height.
    0501 line format: 0501,nElems,height1,height2,...
    Heights in your example are in cm with positive = top.
    
    snow_pos : 
        If True the hieght is given in positive values, if False the height is given in negative values
    """
    date_re = re.compile(r'^\s*0501\s*,\s*\d+\s*,\s*(.+)$', re.MULTILINE)
    try:
        with open(pro_path, 'r', encoding='utf-8') as fh:
            text = fh.read()
    except Exception:
        return None
    matches = date_re.findall(text)
    if not matches:
        return None
    max_h = 0.0
    for m in matches:
        parts = [p.strip() for p in m.split(',') if p.strip()]
        try:
            heights = [float(p) for p in parts]
        except ValueError:
            continue
        # heights may be positive/negative; take absolute or max depending on convention
        if snow_pos:
            # heights are positive values; take the maximum directly
            max_h = max(max_h, max(heights))
        else:
            # heights are negative values; take absolute values
            max_h = max(max_h, max(abs(h) for h in heights))
    return max_h

## helper functions for plot min max (pro and regobs)
def extract_regobs_tmp(result_or_profile: Any) -> Optional[Dict[str, Any]]:
    """
    Extract min and max temperatures and the profile date from a regobslib result or snow_profile.

    Parameters
    - result_or_profile: object that is either a regobslib result with attribute
        `snow_profile` or a snow_profile object itself. The profile is expected to
        have a `temperatures` attribute which is an iterable of objects with
        attributes `temp_c` and `depth_cm`. The function will also try to extract
        a date/time from common attributes on the result/profile.

    Returns
    - dict with keys:
        'date'          : datetime or str or None (parsed datetime if possible, otherwise raw value)
        'min_temp_c'    : float or None
        'min_depth_cm'  : float or None
        'max_temp_c'    : float or None
        'max_depth_cm'  : float or None
        or None if no valid temperature measurements are present.
    """
    # Resolve profile if a result object was passed
    prof = getattr(result_or_profile, 'snow_profile', None) or result_or_profile
    if prof is None:
        return None

    # --- attempt to extract a date from common attributes ---
    date_value = None

    # helper to try attributes in order
    candidate_attrs = [
        'date', 'time', 'timestamp', 'datetime', 'date_time', 'obs_time',
        'observation_time', 'observation_date', 'profile_date', 'registration_time',
        'registration_date'
    ]

    # check top-level object first (result_or_profile), then prof
    for obj in (result_or_profile, prof):
        if obj is None:
            continue
        # if object is a dict-like
        if isinstance(obj, dict):
            for k in candidate_attrs:
                if k in obj and obj[k] not in (None, ''):
                    date_value = obj[k]
                    break
        else:
            for k in candidate_attrs:
                if hasattr(obj, k):
                    v = getattr(obj, k)
                    if v not in (None, ''):
                        date_value = v
                        break
        if date_value is not None:
            break

    # If still not found, try common nested places (e.g., header, meta)
    if date_value is None:
        for obj in (result_or_profile, prof):
            if obj is None:
                continue
            # dict-like containers
            if isinstance(obj, dict):
                for container_key in ('header', 'meta', 'properties'):
                    container = obj.get(container_key)
                    if isinstance(container, dict):
                        for k in candidate_attrs:
                            if k in container and container[k] not in (None, ''):
                                date_value = container[k]
                                break
                        if date_value is not None:
                            break
            else:
                for container_key in ('header', 'meta', 'properties'):
                    container = getattr(obj, container_key, None)
                    if isinstance(container, dict):
                        for k in candidate_attrs:
                            if k in container and container[k] not in (None, ''):
                                date_value = container[k]
                                break
                        if date_value is not None:
                            break
            if date_value is not None:
                break

    # try to parse date_value into a datetime if possible
    parsed_date = None
    if date_value is not None:
        # If it's already a datetime, keep it
        if isinstance(date_value, datetime):
            parsed_date = date_value
        else:
            # try common parsing strategies
            # 1) try dateutil if available
            try:
                from dateutil import parser as _dparser  # type: ignore
                parsed_date = _dparser.parse(str(date_value))
            except Exception:
                # 2) try fromisoformat (works for many ISO strings)
                try:
                    parsed_date = datetime.fromisoformat(str(date_value))
                except Exception:
                    # 3) try some common formats (day.month.year hour:minute:second)
                    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                        try:
                            parsed_date = datetime.strptime(str(date_value), fmt)
                            break
                        except Exception:
                            continue
                    # if none matched, leave parsed_date as None and keep raw value

    temps = []
    for t in getattr(prof, 'temperatures', []) or []:
        if t is None:
            continue
        temp = getattr(t, 'temp_c', None)
        depth = getattr(t, 'depth_cm', None)
        if temp is None:
            continue
        try:
            temp_f = float(temp)
        except Exception:
            continue
        try:
            depth_f = float(depth) if depth is not None else None
        except Exception:
            depth_f = None
        temps.append((temp_f, depth_f))

    if not temps:
        return None

    # Find min and max temperature entries
    min_temp, min_depth = min(temps, key=lambda x: x[0])
    max_temp, max_depth = max(temps, key=lambda x: x[0])

    return {
        'date': parsed_date if parsed_date is not None else date_value,
        'min_temp_c': min_temp,
        'min_depth_cm': min_depth,
        'max_temp_c': max_temp,
        'max_depth_cm': max_depth
    }


def extract_pro_tmp(source: Union[str, io.StringIO], from_file: bool = True) -> List[Dict[str, Optional[float]]]:
    """
    Parse a .pro file and extract min and max temperatures per profile.

    Parameters
    - source: file path (str) when from_file is True, otherwise a string with file contents.
    - from_file: if True treat `source` as a file path; if False treat `source` as raw file text.

    Returns
    - List of dicts, one per profile, with keys:
        'profile_index'    : int (0-based)
        'timestamp'        : str or None (value from 0500 line)
        'n_elems'          : int or None (element count from 0501/0503 lines if present)
        'min_temp_c'       : float or None
        'min_depth_cm'     : float or None
        'max_temp_c'       : float or None
        'max_depth_cm'     : float or None

    Notes
    - The function expects temperature lines to be marked with code 0503 and depth lines with code 0501.
    - If depths are not present or lengths mismatch, the function will pair temperatures with depths by index up to the shortest list.
    - Non-numeric tokens are ignored.
    """
    # Read file content
    if from_file:
        with open(source, 'r', encoding='utf-8') as fh:
            text = fh.read()
    else:
        text = source

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != '']

    profiles = []
    current = {
        'timestamp': None,
        'depths': None,   # list of floats (cm) from 0501
        'temps': None,    # list of floats (degC) from 0503
        'n_elems': None
    }

    def flush_current(idx: int):
        """Compute min/max for the current profile and append to profiles list."""
        depths = current.get('depths') or []
        temps = current.get('temps') or []
        n_elems = current.get('n_elems')
        timestamp = current.get('timestamp')

        # sanitize lists: keep only numeric floats
        def to_floats(lst):
            out = []
            for v in lst:
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out

        depths_f = to_floats(depths)
        temps_f = to_floats(temps)

        if not temps_f:
            profiles.append({
                'profile_index': idx,
                'timestamp': timestamp,
                'n_elems': n_elems,
                'min_temp_c': None,
                'min_depth_cm': None,
                'max_temp_c': None,
                'max_depth_cm': None
            })
            return

        # align lengths
        L = min(len(temps_f), len(depths_f)) if depths_f else len(temps_f)

        # if depths missing, set depths list to None entries
        if not depths_f:
            depths_aligned = [None] * len(temps_f)
        else:
            depths_aligned = depths_f[:L]
            if len(temps_f) > L:
                # append None for remaining temps
                depths_aligned += [None] * (len(temps_f) - L)

        # find min and max temps and corresponding depths (first occurrence)
        min_idx = min(range(len(temps_f)), key=lambda i: temps_f[i])
        max_idx = max(range(len(temps_f)), key=lambda i: temps_f[i])

        min_temp = temps_f[min_idx]
        max_temp = temps_f[max_idx]
        min_depth = depths_aligned[min_idx] if min_idx < len(depths_aligned) else None
        max_depth = depths_aligned[max_idx] if max_idx < len(depths_aligned) else None

        profiles.append({
            'profile_index': idx,
            'timestamp': timestamp,
            'n_elems': n_elems,
            'min_temp_c': min_temp,
            'min_depth_cm': min_depth,
            'max_temp_c': max_temp,
            'max_depth_cm': max_depth
        })

    profile_count = 0
    for ln in lines:
        # skip comment lines in header or lines starting with '#'
        if ln.startswith('#'):
            continue

        # split by comma but preserve possible commas inside values (not expected here)
        parts = [p.strip() for p in ln.split(',')]

        if not parts:
            continue

        code = parts[0]
        # new profile marker
        if code == '0500':
            # if current has any data, flush it
            if current['temps'] is not None or current['depths'] is not None or current['timestamp'] is not None:
                flush_current(profile_count)
                profile_count += 1
                # reset
                current = {'timestamp': None, 'depths': None, 'temps': None, 'n_elems': None}

            # timestamp is the remainder joined (some files have commas in timestamp)
            timestamp = ','.join(parts[1:]).strip() if len(parts) > 1 else None
            current['timestamp'] = timestamp

        elif code == '0501':
            # depths line: format 0501,nElems,val1,val2,...
            if len(parts) >= 2:
                try:
                    n = int(parts[1])
                except Exception:
                    n = None
                current['n_elems'] = n
            # values after the second token
            values = parts[2:] if len(parts) > 2 else []
            current['depths'] = values

        elif code == '0503':
            # temperatures line: 0503,nElems,val1,val2,...
            if len(parts) >= 2 and current.get('n_elems') is None:
                try:
                    current['n_elems'] = int(parts[1])
                except Exception:
                    pass
            values = parts[2:] if len(parts) > 2 else []
            current['temps'] = values

        else:
            # ignore other codes
            continue

    # flush last profile if present
    if current['temps'] is not None or current['depths'] is not None or current['timestamp'] is not None:
        flush_current(profile_count)

    return profiles

def _pro_title(pro_source, from_file=True, max_len=60):
    if pro_source is None:
        return ".pro Min and Max Snow Temperature"

    elif from_file:
        # prefer pathlib but fall back to os.path if needed
        try:
            name = Path(pro_source).name
        except Exception:
            name = os.path.basename(str(pro_source))
        title_base = name or ".pro"
    else:
        # pro_source is raw text: use first non-empty line or a short preview
        s = str(pro_source).strip()
        first_line = s.splitlines()[0] if s else ""
        if first_line:
            title_base = first_line if len(first_line) <= max_len else first_line[:max_len] + "…"
        else:
            title_base = "raw .pro content"

    return f"{title_base} Min and Max Snow Temperature"

## helper function for temperature under threshold
def _parse_values_from_line(line: str) -> List[float]:
    """
    Parse numeric values from a data line like:
    '0503,25,0.00,0.00,-0.00,0.00,...'
    Returns list of floats for the values after the second comma.
    """
    parts = line.strip().split(',')
    if len(parts) < 3:
        return []
    # values are everything after the first two fields (code and count)
    raw_vals = parts[2:]
    vals = []
    for v in raw_vals:
        v = v.strip()
        if v == '' or v.upper() == 'NA':
            continue
        # handle -0.00 and scientific notation
        try:
            vals.append(float(v))
        except ValueError:
            # try to remove stray characters
            v_clean = re.sub(r'[^\d\+\-eE\.]', '', v)
            try:
                vals.append(float(v_clean))
            except ValueError:
                # skip non-numeric tokens
                continue
    return vals
