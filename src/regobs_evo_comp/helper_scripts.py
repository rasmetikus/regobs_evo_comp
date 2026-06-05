
from itertools import product
# import numpy as np
import regobslib
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import os
from snowpacktools.snowpro import snowpro, pro_plotter, pro_helper 

def get_profiles_from_regobs(lat_range:tuple, lon_range:tuple,  
                             date_start:str, date_end:str, 
                             regions=None, nicknames=None,
                               location_id=None, verbose=True ) -> list:
    """
    retrieve regobslib.SnowProfiles from regobs that meet specified criteria. 
    Returns a list of regobslib.SnowRegistration sorted by time.
    
    :param lat_range (tuple of floats): min, max lat values to filter results by
    :param lon_range: Description
    :param date_start (str): start time
    :param date_end (str): end time
    :param regions (regobslib.SnowRegion): region to filter by
    :param nicknames (list of strings): observer nicknames to filter by
    :param location_id: Description
    :param verbose (bool): print feedback

    returns: a list of regobslib.SnowRegistration objects, sorted by time.
    """
    observer_string = f"by observers {nicknames}" if nicknames is not None else ""
    regions_string =  f"in regions {regions}" if regions is not None else ""
    
    if verbose: 
        print(f"fetching snow profiles from regobs from {date_start} to {date_end}, within lat/lon range: {lat_range}/{lon_range}, {observer_string}")
    
    date_format="%Y-%m-%d"; date_format2="%Y-%m-%dT%H:%M"
    obs_st = datetime.datetime.strptime(date_start, date_format) #(2025, 12, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    obs_ed = datetime.datetime.strptime(date_end, date_format) 

    lists=[
        regions if isinstance(regions, list) else [regions],
        [obs_st],
        [obs_ed],
        nicknames if isinstance(nicknames,list) else [nicknames]
    ]
    ## all combinations of the lists of arguments:
    args = [
        combo for combo in product(*lists)
        if len(set(combo)) == len(combo)   # removes any repetition
    ]

    # loop over arg combis and retrieve from regobs:
    connection = regobslib.Connection(prod=True)
    res_list=[]
    for a in range(0, len(args)):
        results = connection.search(regobslib.SnowRegistration,  observation_types=[regobslib.SnowProfile], regions=args[a][0],
                                from_obs_time=args[a][1], to_obs_time=args[a][2],
                                observer_nickname=args[a][3])
        res_list.append(results)
    
    if verbose: print(f"Observations found. Filtering for lat/lon range....")
    ### filter for position:
    profiles_loc=[]
    for rl in range(0,len(res_list)):
        for ires in range(0, len(res_list[rl])):
            if (res_list[rl][ires].position.lat < lat_range[1] and
                res_list[rl][ires].position.lat > lat_range[0] and
                res_list[rl][ires].position.lon < lon_range[1] and
                res_list[rl][ires].position.lon > lon_range[0] 
            ):
                # if verbose: print(res_list[rl][ires].position)   
                profiles_loc.append(res_list[rl][ires] )  # return regobslib objects, not dicts
                if verbose: print(f" {res_list[rl][ires].obs_time.strftime(date_format2)} by {res_list[rl][ires].observer.nickname} at {res_list[rl][ires].position.lat:.3f}/{res_list[rl][ires].position.lon:.3f} ")
    if verbose: print(f" {len(profiles_loc)} snow profiles found matching your criteria") # , profiles_loc[0].keys()
    
    # now sort the list 'profiles_loc' by obs_time:
    if verbose: print(f"sorting snow profiles by date...")
    times=[]
    for res in profiles_loc:  times.append(res.obs_time)
    profiles_sorted = [x for _, x in sorted(zip(times, profiles_loc), key=lambda t: t[0])]

    return(profiles_sorted)
    
    
    
    
# importlib.reload(pro_helper)
# importlib.reload(pro_plotter)


def _regobs_grain_type_code(grain_form):
    if grain_form is None:
        return '-999'
    name = getattr(grain_form, 'name', str(grain_form))
    if name.startswith('PP'):
        return 'PP'
    if name.startswith('DF'):
        return 'DF'
    if name.startswith('SH'):
        return 'SH'
    if name.startswith('DH'):
        return 'DH'
    if name.startswith('FC'):
        return 'FC'
    if name.startswith('RG'):
        return 'RG'
    if name.startswith('MF'):
        return 'MF'
    if name.startswith('IF'):
        return 'IF'
    return '-999'


def _regobs_hardness_to_N(hardness):
    if hardness is None:
        return 0
    name = getattr(hardness, 'name', str(hardness))
    if 'ICE' in name or 'KNIFE' in name:
        return -1422
    if 'PEN' in name:
        return -918
    if 'ONE_FINGER' in name:
        return -538
    if 'FOUR_FINGERS' in name:
        return -269
    if 'FIST' in name:
        return -102
    return 0


def _create_grain_type_legend(fig, ax, color_scheme='SARP', orientation='horizontal'):
    LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, LABELS_GRAIN_TYPE_BAR, COLORS_GRAIN_TYPE_BAR, HATCHES_GRAIN_TYPE_BAR = pro_helper.get_grain_type_colors(color_scheme)
    import matplotlib.patches as mpatches

    handles = [mpatches.Patch(facecolor=color, edgecolor='black', label=label) for label, color in zip(LABELS_GRAIN_TYPE_BAR, COLORS_GRAIN_TYPE_BAR)]
    ax.axis('off')

    if orientation == 'horizontal':
        ax.legend(handles=handles, ncol=min(len(handles), 6), frameon=False, loc='center', fontsize='small')
    else:
        ax.legend(handles=handles, frameon=False, loc='center left', fontsize='small')


def create_grain_type_colorbar(fig=None, ax=None, color_scheme='SARP', orientation='vertical', location='left', figsize=(1, 6)):
    """Create a grain-type legend/colorbar for PRO snow layers."""
    LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, LABELS_GRAIN_TYPE_BAR, COLORS_GRAIN_TYPE_BAR, HATCHES_GRAIN_TYPE_BAR = pro_helper.get_grain_type_colors(color_scheme)
    import matplotlib.patches as mpatches

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    ax.axis('off')
    handles = [
        mpatches.Patch(facecolor=color, edgecolor='black', label=label, hatch=hatch or '')
        for label, color, hatch in zip(LABELS_GRAIN_TYPE_BAR, COLORS_GRAIN_TYPE_BAR, HATCHES_GRAIN_TYPE_BAR)
    ]

    loc_map = {
        'left': 'center left',
        'right': 'center right',
        'center': 'center',
        'upper left': 'upper left',
        'upper right': 'upper right',
        'lower left': 'lower left',
        'lower right': 'lower right'
    }
    legend_loc = loc_map.get(location, location)

    if orientation == 'horizontal':
        ax.legend(handles=handles, ncol=min(len(handles), 6), frameon=False, loc=legend_loc, fontsize='small')
    else:
        ax.legend(handles=handles, frameon=False, loc=legend_loc, fontsize='small')

    return fig, ax


def plot_regobs_profile(result, ax=None, height_max=None, color_scheme='SARP'):
    """Plot a regobslib SnowRegistration snow profile on a given axis."""
    prof = getattr(result, 'snow_profile', None)
    if prof is None:
        raise ValueError('No snow_profile found on the provided regobslib result.')

    layers = getattr(prof, 'layers', None)
    if not layers:
        raise ValueError('Snow profile contains no layers to plot.')

    hand_hardness_dict, tickz_hh, tick_labels_hh = pro_helper.get_hand_hardness_N_dict()
    LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, _, _, _ = pro_helper.get_grain_type_colors(color_scheme)
    grain_color_map = dict(zip(LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE))
    grain_hatch_map = dict(zip(LABELS_GRAIN_TYPE, HATCHES_GRAIN_TYPE))

    thickness = []
    bottoms = []
    hardness_values = []
    colors = []
    hatches = []
    heights = []

    depth = 0.0
    for layer in layers:
        if layer.thickness_cm is None or layer.thickness_cm <= 0:
            continue
        thickness.append(layer.thickness_cm)
        bottoms.append(depth)
        depth += layer.thickness_cm
        heights.append(depth)

        grain_code = _regobs_grain_type_code(layer.grain_form_primary)
        colors.append(grain_color_map.get(grain_code, 'white'))
        hatches.append(grain_hatch_map.get(grain_code, ''))
        hardness_values.append(_regobs_hardness_to_N(layer.hardness))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 5))
    else:
        fig = None

    ax.barh(bottoms, hardness_values, height=thickness, align='edge', color=colors, hatch=hatches)

    if height_max is None:
        height_max = 1.1 * max(heights)
    ax.set_ylim(0, height_max)

    temperatures = getattr(prof, 'temperatures', None)
    if temperatures:
        temps = [t.temp_c for t in temperatures if t is not None]
        depths = [t.depth_cm for t in temperatures if t is not None]
        if temps and depths:
            ax_t = ax.twiny()
            ax_t.plot(temps, depths, color='#DC143C', lw=1)
            ax_t.grid(False)
            ax_t.xaxis.tick_bottom()
            ax_t.xaxis.set_label_position('bottom')
            ax_t.set_xlabel('snow temperature / °C', color='#DC143C')
            ax_t.tick_params(axis='x', colors='#DC143C')
            ax_t.set_xlim(min(temps) - 2, 0)

    ax.set_xlim(-1100, 50)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('height / cm')

    ax_hh = ax.twiny()
    ax_hh.set_xlim(-1100, 50)
    ax_hh.grid(False)
    ax_hh.xaxis.tick_top()
    ax_hh.xaxis.set_label_position('top')
    ax_hh.set_xticks(tickz_hh)
    ax_hh.set_xticklabels(tick_labels_hh)
    ax_hh.tick_params(axis='x', direction='in', pad=-18)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('hand hardness / N')

    return ax


def plot_prof_comp(result, sims, height_max=None, color_scheme= "SARP",ncols=2, add_cbar ="bottom",show_plot=True, save_plot=False, out_path="./"):
    """
    Plot regobslib.SnowProfiles
    
    :param result (regobslib.SnowRegistration): a regobslib.SnowRegistration containing a regobslib.SnowProfile
    :param sims (tuple): contains two dicts with paths and stations of the simulations
    :param color_scheme: SARP or IACS2 grain type colors
    :param ncols (int): number of columns for axes
    :param add_cbar: if anf where to a color bar with grain types (right, bottom, ax)
    :param show_plot (bool): dispay plot (default True)
    :param save_plot (bool): save  plot (default False)
    :param out_path (str): path to save plot to (if save_plot)
    """
    
    # the tuple sims contains two dicts with paths and stations of the simulations to compare to:
    versions = sims[0]
    stn = sims[1]
    ncols=int(ncols)

    if out_path is None:
        out_path=f"./{result.position.region.name}"

    tmp_path = os.path.join(os.path.expanduser('~'), 'plots', 'tmp')  # park automatically generated single profiles here. f"Pout_path}/indv"?
    os.makedirs(tmp_path, exist_ok=True)

    # fig, axs = plt.subplots(4,2, figsize=(10,25))
    
    fig, axs = plt.subplots( int(np.ceil((len(list(versions))+1)/ncols)),ncols, 
                         figsize=(10,5*np.ceil((len(list(versions))+1)/ncols) ))
    # print(height_max) # OK
    ## plot obs
    plot_regobs_profile(result, ax=axs.flatten()[0], height_max=height_max, color_scheme=color_scheme)
    axs.flatten()[0].set_title(f"Observed Profile", weight='bold', fontsize=12)

    # plot simulated profiles:
    # for i,k in enumerate(reversed(versions.keys())):
    def _find_local_pro(version_key, station_name):
        # Map version keys to expected directory patterns
        version_to_dir = {
            "snower_Stng": "snower_Stng",
            "snower_Stng_soil": "snower_Stng",  # Note: both snower versions use the same dir
            "1km": "Stongeskardet_1km",
            "1km_wSoil": "Stongeskardet_1km_soil",
            "1km_next_cell": "Stongeskardet_1km",  # VIR25A is in 1km dir
            "20km_Hallingdal": "Hallingdal",
            "20km_Hallingdal_W": "Hallingdal"
        }
        
        expected_dir = version_to_dir.get(version_key)
        if expected_dir:
            candidate = os.path.join(os.getcwd(), expected_dir, 'pro', f"{station_name}.pro")
            if os.path.exists(candidate):
                return candidate
        
        # Fallback to walking the tree if mapping fails
        for root, dirs, files in os.walk(os.getcwd()):
            if f"{station_name}.pro" in files:
                return os.path.join(root, f"{station_name}.pro")
        return None

    for i,k in enumerate(versions.keys()):
        try:
            pro_path = os.path.join(versions[k], 'pro', f"{stn[k]}.pro")
            if not os.path.exists(pro_path):
                fallback = _find_local_pro(k, stn[k])
                if fallback is not None:
                    print(f"[i] PRO path not found at {pro_path}, falling back to local {fallback}")
                    pro_path = fallback
                else:
                    raise FileNotFoundError(pro_path)

            pro_plotter.single_profile( pro_path=pro_path,
                            out_path=f"{tmp_path}/tmp_prof.png",
                            DATETIME_STR=result.obs_time.strftime("%Y-%m-%dT%H:%M"),
                            height_max=height_max,
                            color_scheme=color_scheme,
                            ax=axs.flatten()[1+i]
                            )
        except Exception as e:
            print(f"\n [ERROR] Could not plot version: {k}, error: {e} \n")
        axs.flatten()[1+i].set_title(f"{k} - {stn[k]}", weight='bold', fontsize=12)

    fig.subplots_adjust(hspace=0.35)
    # plt.tight_layout() # fucks up cbar

    if add_cbar=="ax":
        _create_grain_type_legend(fig, axs.flatten()[0], color_scheme=color_scheme, orientation='vertical')

    if add_cbar=="right":
        cax = fig.add_axes([0.93, 0.11, 0.05, 0.77])  # reserve for legend
        _create_grain_type_legend(fig, cax, color_scheme=color_scheme, orientation='vertical')

    if add_cbar=="bottom": # 'bottom'
        cax = fig.add_axes([0.1, 0.02, 0.8, 0.08])
        _create_grain_type_legend(fig, cax, color_scheme=color_scheme, orientation='horizontal')


    ## save and/or show plot:
    if save_plot:
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(f'{out_path}/profile_{result.obs_time.strftime(format="%Y-%m-%d")}.png', dpi=200)
    if show_plot:
        plt.show()






# def ranks_by_datetime(dts, tie_method="stable"):
#     """
#     Return a list 'rank[i]' giving the rank of dts[i] when sorting by full datetime.
    
#     tie_method:
#       - "stable": equal datetimes keep their original order (0,1,2,...)
#       - "min": all equal values get the smallest index among them
#       - "average": all equal values get the average of their positions
#       - "dense": equal values share the same rank; next distinct gets +1
#     """
#     # Order of indices that sorts by full datetime
#     order = sorted(range(len(dts)), key=lambda i: dts[i])
    
#     # Map sorted position → original index
#     sorted_indices = order
    
#     # Build base positions
#     pos_of = [0] * len(dts)  # pos_of[i] = position of element i in sorted order
#     for pos, i in enumerate(sorted_indices):
#         pos_of[i] = pos

#     if tie_method == "stable":
#         return pos_of

#     # Group equal values by their key to apply other tie methods
#     from collections import defaultdict
#     groups = defaultdict(list)
#     for pos, i in enumerate(sorted_indices):
#         groups[dts[i]].append(pos)

#     rank = [0] * len(dts)
#     if tie_method == "min":
#         for i in range(len(dts)):
#             positions = groups[dts[i]]
#             rank[i] = min(positions)
#     elif tie_method == "average":
#         for i in range(len(dts)):
#             positions = groups[dts[i]]
#             rank[i] = sum(positions) / len(positions)
#     elif tie_method == "dense":
#         # Assign dense ranks to distinct keys in sorted order
#         distinct_order = sorted(groups.keys())
#         dense_rank_map = {key: r for r, key in enumerate(distinct_order)}
#         for i in range(len(dts)):
#             rank[i] = dense_rank_map[dts[i]]
#     else:
#         raise ValueError("Unsupported tie_method")
#     return rank