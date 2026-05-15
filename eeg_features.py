"""
Extracting EEG, EMG, and hypnogram features - so far:
Hypnogram: SPT (sleep period time), WASO (wake after sleep onset), TST (total sleep time), REM latency, SME (sleep maintenance efficiency), sleep fragmentation index, and amount of each sleep stage and proportion of TST
EEG: band powers for each stage and channel (delta, theta, alpha, sigma, beta, total abs power)
EMG: submental RMS (µV) per 30 s epoch 
"""

import mne
import glob
import yasa
import numpy as np
import pandas as pd
import numpy as np

from emg_exploration_utils import emg_submental_rms

def _hyp_stages_int(hyp):
    if hasattr(hyp, "as_int"):
        st = hyp.as_int()
        return st.to_numpy() if hasattr(st, "to_numpy") else np.asarray(st, dtype=int)
    inv = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Uns": -1}
    inv = {v: k for k, v in STAGE_MAP.items()}
    hypno_vals = getattr(hyp, "hypno", None)
    if hypno_vals is None:
        return None
    return np.array([inv.get(str(x), -1) for x in hypno_vals], dtype=int)


def _emg_rms_feature_row(rms_uv, st_arr):
    """One wide row: mean/std RMS per stage 0–4 and across aligned epochs."""
    emg_stage_labels = {k: v for k, v in STAGE_MAP.items() if k >= 0}
    cols = {}
    for lab in emg_stage_labels.values():
        cols[f"EMG_submental_RMS_mean_{lab}"] = np.nan
        cols[f"EMG_submental_RMS_std_{lab}"] = np.nan
    cols["EMG_submental_RMS_mean_all"] = np.nan
    cols["EMG_submental_RMS_std_all"] = np.nan

    if rms_uv is None or st_arr is None:
        return pd.DataFrame([cols])

    rms_uv = np.asarray(rms_uv, dtype=float)
    st_arr = np.asarray(st_arr, dtype=int)
    n = min(len(rms_uv), len(st_arr))
    if n == 0:
        return pd.DataFrame([cols])

    rms_uv = rms_uv[:n]
    st_arr = st_arr[:n]

    for si, lab in emg_stage_labels.items():
        mask = st_arr == si
        if np.any(mask):
            cols[f"EMG_submental_RMS_mean_{lab}"] = float(np.mean(rms_uv[mask]))
            cols[f"EMG_submental_RMS_std_{lab}"] = float(np.std(rms_uv[mask], ddof=0))

    cols["EMG_submental_RMS_mean_all"] = float(np.mean(rms_uv))
    cols["EMG_submental_RMS_std_all"] = float(np.std(rms_uv, ddof=0))

    return pd.DataFrame([cols])

EPOCH_DURATION = 30.0
EEG_CHANNELS   = ['EEG Fpz-Cz', 'EEG Pz-Oz']
STAGE_MAP      = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R', -1: 'Uns'}

# load file paths
# define path to the data
# ADAPT THIS PATH TO WHERE YOU HAVE THE DATA
data_path = "/home/alexia/Downloads/"

folder_path = "sleep-edf-database-expanded-1.0.0"
sc = "sleep-cassette"
st = "sleep-telemetry"

sc_paths = sorted(glob.glob(f"{data_path}/{folder_path}/{sc}/preprocessed/*.fif"))
st_paths = sorted(glob.glob(f"{data_path}/{folder_path}/{st}/preprocessed/*.fif"))

datasets = {'sc': sc_paths, 'st': st_paths}
include_stages = []

all_dfs = []

for dataset_name, paths in datasets.items():
    for path in paths:

        # load raw objects
        raw = mne.io.read_raw_fif(path, preload=True, verbose='ERROR')
        eeg = raw.copy().pick_channels(EEG_CHANNELS)
        eeg.load_data()

        # build hypnogram
        hypno = eeg.annotations.description
        hyp = yasa.Hypnogram.from_integers(hypno, mapping=STAGE_MAP, n_stages=5, freq='30s')

        # EMG features
        _, rms_uv = emg_submental_rms(raw, window_sec=EPOCH_DURATION)
        st_arr = _hyp_stages_int(hyp)
        emg_feats = _emg_rms_feature_row(rms_uv, st_arr)

        # compute bandpowers
        bp = yasa.bandpower(data=eeg, hypno=hyp, include=(0, 1, 2, 3, 4))
        bp = bp.drop(columns=["FreqRes", "Relative"])
        bp = bp.reset_index()

        # convert format into one row
        melted = bp.melt(id_vars=["Stage", "Chan"], var_name="Band", value_name="Value")
        melted['Feature'] = (
            melted['Chan'].astype(str) + '_' +
            melted['Stage'].astype(str) + '_' +
            melted['Band']
        )
        single_row = melted.pivot_table(index=None, columns='Feature', values='Value', aggfunc='first')
        single_row = single_row.reset_index(drop=True)

        sleep_stats = pd.DataFrame(hyp.sleep_statistics(), index=[0])
        sleep_stats = sleep_stats.drop(columns=["TIB", "SOL", "SOL_5min", "SE" ])

        # compute number of seconds in N2 and N3
        nrem_sec = np.sum((hypno=='2')|(hypno=='3'))*EPOCH_DURATION

        # detect slow waves and spindles
        sw = yasa.sw_detect(eeg, hypno=hyp, include=(2,3))
        if sw is None:
            sw_single_row = pd.DataFrame({f'{col}_SW_density': 0 for col in EEG_CHANNELS}, index=[0])
        else:
            sw_counts = sw.summary(grp_chan=True).reset_index() # slow wave count per EEG channel
            # compute slow wave and spindle density in sw/sp per minute
            sw_counts['SW_density'] = sw_counts['Count']/nrem_sec*60

            # convert format into one row
            sw_single_row = sw_counts.pivot_table(index=None, columns='Channel', values='SW_density', aggfunc='first')
            sw_single_row = sw_single_row.reset_index(drop=True)
            for col in EEG_CHANNELS:
                if col not in sw_single_row.columns:
                    sw_single_row[col] = 0
            sw_single_row.columns = [f'{col}_SW_density' for col in sw_single_row.columns]

        
        sp = yasa.spindles_detect(eeg, hypno=hyp, include=(2,3))
        if sp is None:
            sp_single_row = pd.DataFrame({f'{col}_SP_density': 0 for col in EEG_CHANNELS}, index=[0])
        else:
            sp_counts = sp.summary(grp_chan=True).reset_index() # spindle count per EEG channel
            sp_counts['SP_density'] = sp_counts['Count']/nrem_sec*60
            sp_single_row = sp_counts.pivot_table(index=None, columns='Channel', values='SP_density', aggfunc='first')
            sp_single_row = sp_single_row.reset_index(drop=True)
            for col in EEG_CHANNELS:
                if col not in sp_single_row.columns:
                    sp_single_row[col] = 0
            sp_single_row.columns = [f'{col}_SP_density' for col in sp_single_row.columns]

       
        # One row per recording: EEG band powers, EMG RMS by stage, hypnogram stats
        sample = pd.concat([single_row, sw_single_row, sp_single_row, emg_feats, sleep_stats], axis=1)
        sample["dataset"] = dataset_name
        sample["file"] = path

        all_dfs.append(sample)

df = pd.concat(all_dfs, ignore_index=True)

fname = "features.csv"

df.to_csv(fname)