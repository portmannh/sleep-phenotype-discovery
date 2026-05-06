import mne
import numpy as np

def get_data(path):
    """
    Get data from a specific path
    Input:
    - path: str - path to the raw data file.
    Output:
    - raw: mne.io.BaseRaw - loaded raw data.
    - sfreq: float - sampling frequency.
    - chs: tuple - channel names.
    - n_epochs: int - number of epochs.
    - annot: mne.annotations.Annotations - annotations.
    - duration: float - duration of the data.
    """

    raw = mne.io.read_raw_fif(path, preload=False, verbose='ERROR')
    annot = raw.annotations
    duration = raw.n_times / raw.info['sfreq']
    chs = tuple(raw.info['ch_names'])
    n_epochs = len(annot)

    return raw, raw.info['sfreq'], chs, n_epochs, annot, duration

def get_ch(raw, ch_name):
    """
    Get a channel from an MNE Raw object by name.
    Input:
    - raw : mne.io.BaseRaw - FIF recording.
    - ch_name : str - channel name.
    Output:
    - data : np.ndarray - 1D numpy array of channel samples.
    """

    try:
        data = raw.get_data(picks=[ch_name])[0]
        return data
    except Exception as e:
        print(f"Channel List: {raw.ch_names}")
        raise ValueError(f"Unable to get channel data - {e}")
