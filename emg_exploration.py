from emg_exploration_utils import get_data, get_ch, compute_rms
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# load file paths
# define path to the data
# ADAPT THIS PATH TO WHERE YOU HAVE THE DATA
data_path = "../../data"

folder_path = "sleep-edfx/1.0.0"
sc = "sleep-cassette"
st = "sleep-telemetry"

sc_paths = sorted(glob.glob(f"{data_path}/{folder_path}/{sc}/preprocessed/*.fif"))
st_paths = sorted(glob.glob(f"{data_path}/{folder_path}/{st}/preprocessed/*.fif"))

datasets = {"Sleep-Cassette": sc_paths, "Sleep-Telemetry": st_paths}
stage_map = {'0': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'R', '-1': '?'}
stage_order = {'W': 0, 'R': 1, 'N1': 2, 'N2': 3, 'N3': 4, '?': 5}


#===================== LOAD DATA =====================#

file_path = datasets["Sleep-Cassette"][0]
raw, sfreq, chs, n_epochs, annot, duration = get_data(file_path)


ch_data = get_ch(raw, "EMG submental")

print(ch_data.shape)

#===================== PSD (WELCH) =====================#
f, pxx = welch(ch_data, fs=sfreq, nperseg=int(4 * sfreq))
plt.figure(figsize=(10, 4))
plt.semilogy(f, pxx)
plt.xlim(0, sfreq / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V²/Hz)")
plt.title("Welch PSD")
plt.grid(True, alpha=0.3)

#===================== EMG + RMS PLOT =====================#
# Convert to microvolts so units are interpretable (µV)
ch_uv = np.asarray(ch_data, dtype=float) * 1e6
ch_uv = np.nan_to_num(ch_uv, nan=0.0, posinf=0.0, neginf=0.0)
ch_uv = ch_uv - np.mean(ch_uv)
print("EMG (µV) min/mean/max:", float(np.min(ch_uv)), float(np.mean(ch_uv)), float(np.max(ch_uv)))

t = np.arange(len(ch_uv)) / sfreq
t_rms, rms = compute_rms(ch_uv, sfreq=sfreq, window_sec=30.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(t, ch_uv, color="C0", linewidth=0.6)
ax1.set_ylabel("EMG (µV)")
ax1.set_title("EMG Signal (top) and RMS Magnitude (bottom)")
ax1.grid(True, alpha=0.3)

ax2.plot(t_rms, rms, color="C1", linewidth=1.2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("RMS (µV)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

