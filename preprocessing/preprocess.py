from pathlib import Path
import mne
import pandas as pd
import numpy as np
import gc

def preprocess(eeg_file_ext, ann_file, bandpass_filter=(0.1, 100), notch_filter=True, resampling_freq=200, verbosity=False):

    subject = Path(eeg_file_ext).parents[1].name # subject ID

    raw = mne.io.read_raw_eeglab(eeg_file_ext, preload=True) # .set file points to .fdt binary eeg file
    raw.pick_channels(['ELA', 'ELB', 'ELC','ELT','ELE','ELI','ERA','ERB','ERC','ERT','ERE','ERI'], verbose=verbosity) # selecting ear-eeg channels only

    raw_data = raw.get_data()
    nan_frac_per_channel = np.isnan(raw_data).mean(axis=1)

    nan_channel_threshold=0.2
    min_channels_required=4

    # identify bad channels
    bad_channels = [
        ch for ch, frac in zip(raw.ch_names, nan_frac_per_channel)
        if frac > nan_channel_threshold
    ]

    # drop bads
    if bad_channels:
        raw.drop_channels(bad_channels)

    if len(raw.ch_names) < min_channels_required:
        print(f"{subject}: skipped, only {len(raw.ch_names)} usable channels remain after NaN screening.")
        del raw
        gc.collect()
        return None, None, None

    data = raw.get_data()

    # interpolate unclean channels
    for ch_idx in range(data.shape[0]):
        x = data[ch_idx]
        nan_mask = np.isnan(x)

        if nan_mask.any():
            good_idx = np.where(~nan_mask)[0]

            if len(good_idx) < 2:
                continue

            data[ch_idx, nan_mask] = np.interp(
                np.where(nan_mask)[0],
                good_idx,
                x[good_idx]
            )

    raw._data = data

    # final safety check before filtering
    if np.isnan(raw._data).any():
        print(f"{subject}: NaNs remain after interpolation, skipping subject.")
        del raw
        gc.collect()
        return None, None, None
    
    raw.filter(bandpass_filter[0], bandpass_filter[1], verbose=verbosity) # filter eeg

    if notch_filter: 
        raw.notch_filter(50) # reduce electrical interference
    
    if resampling_freq is not None:
        raw.resample(resampling_freq, verbose=verbosity) # downsample for better processing time
    
    raw_scaled = raw.copy().apply_function(lambda x: (x - x.mean()) / x.std(), picks='all') # added per-record normalization


    ann_df = pd.read_csv(ann_file, sep='\t') # annotation file with onset times, duration, and labels
    ann_df['onset'] = ann_df['onset'].round()

    valid_labels = [1, 2, 3, 4, 5]
    fs = raw.info['sfreq']

    # extracting events array made of onset samples where labels were valid
    valid_mask = ann_df['Scoring1'].isin(valid_labels)
    valid_ann = ann_df.loc[valid_mask, ['onset', 'Scoring1']].copy()
    onset_samples = (valid_ann['onset'].to_numpy() * fs).astype(int)
    labels = valid_ann['Scoring1'].to_numpy(dtype=int)

    events = np.column_stack([
        onset_samples,
        np.zeros(len(labels), dtype=int),
        labels
    ])

    event_id = {
        'Awake': 1, 
        'REM': 2,
        'N1': 3, 
        'N2': 4, 
        'N3': 5
    }

    # Epochs object for streamlined processing
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id, 
        tmin=0.0, 
        tmax=30.0 - 1/fs, 
        baseline=None, 
        preload=True, 
        verbose=verbosity
    )   

    epochs_scaled = mne.Epochs(
        raw_scaled, 
        events, 
        event_id=event_id, 
        tmin=0.0, 
        tmax=30.0 - 1/fs, 
        baseline=None, 
        preload=True, 
        verbose=verbosity)

    X_raw = epochs.get_data(copy=False).astype(np.float32, copy=False)
    X_scaled = epochs_scaled.get_data(copy=False).astype(np.float32, copy=False)
    y = epochs.events[:, 2].astype(np.int64, copy=False)
    groups = np.full(len(y), subject, dtype=object)

    del epochs
    del raw
    del ann_df
    del valid_ann
    del events
    gc.collect()

    return X_raw, X_scaled, y, groups