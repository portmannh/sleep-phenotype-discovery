"""
Extracting EEG and Hypnogram features - so far:
Hypnogram: SPT (sleep period time), WASO (wake after sleep onset), TST (total sleep time), REM latency, SME (sleep maintenance efficiency), sleep fragmentation index, and amount of each sleep stage and proportion of TST
EEG: band powers for each stage and channel (delta, theta, alpha, sigma, beta, total abs power)
"""

import mne
import glob
import yasa
import pandas as pd

EPOCH_DURATION = 30.0
EEG_CHANNELS   = ['EEG Fpz-Cz', 'EEG Pz-Oz']
STAGE_MAP      = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R', -1: 'Uns'}

# load file paths
# define path to the data
# ADAPT THIS PATH TO WHERE YOU HAVE THE DATA
data_path = "../../data"

folder_path = "sleep-edfx/1.0.0"
sc = "sleep-cassette"
st = "sleep-telemetry"

sc_paths = sorted(glob.glob(f"{data_path}/{folder_path}/{sc}/preprocessed/*.fif"))
st_paths = sorted(glob.glob(f"{data_path}/{folder_path}/{st}/preprocessed/*.fif"))

datasets = {'sc': sc_paths, 'st': st_paths}

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

        # Combine into one row
        sample = pd.concat([single_row, sleep_stats], axis=1)
        sample["dataset"] = dataset_name
        sample["file"] = path

        all_dfs.append(sample)

df = pd.concat(all_dfs, ignore_index=True)
print(df)