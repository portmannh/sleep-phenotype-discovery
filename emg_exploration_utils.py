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


def compute_rms(signal, sfreq, window_sec=30.0):
    """
    Compute RMS magnitude

    Input:
    - signal: 1D array-like - EMG signal
    - sfreq: sampling frequency in Hz
    - window_sec: window duration in seconds

    Output:
    - t_centers: time (s) for the center of each RMS window
    - rms: RMS value for each window
    """
    x = np.asarray(signal, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    samples_per_win = int(round(float(sfreq) * float(window_sec)))
    if samples_per_win <= 0:
        raise ValueError("window_sec must produce at least 1 sample per window.")

    n_windows = len(x) // samples_per_win
    if n_windows == 0:
        raise ValueError("Signal shorter than one RMS window.")

    x = x[: n_windows * samples_per_win].reshape(n_windows, samples_per_win) #keep only complete windows and reshape for RMS per epoch
    rms = np.sqrt(np.mean(x**2, axis=1))

    t_centers = (np.arange(n_windows) + 0.5) * window_sec
    return t_centers, rms
