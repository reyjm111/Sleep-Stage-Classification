from pathlib import Path
import mne
import pandas as pd
import numpy as np

def preprocess(eeg_file_ext, ann_file, bandpass_filter=(0.3, 40), notch_filter=True, resampling_freq=200, verbosity=False):

    subject = Path(eeg_file_ext).parents[1].name # subject ID

    raw = mne.io.read_raw_eeglab(eeg_file_ext, preload=True) # .set file points to .fdt binary eeg file
    raw.pick_channels(['ELA', 'ELB', 'ELC','ELT','ELE','ELI','ERA','ERB','ERC','ERT','ERE','ERI'], verbose=verbosity) # selecting ear-eeg channels only

    raw.filter(bandpass_filter[0], bandpass_filter[1]) # filter eeg

    if notch_filter: 
        raw.notch_filter(60) # reduce electrical interference
    
    if resampling_freq is not None:
        raw.resample(resampling_freq) # downsample for better processing time

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

    labels = epochs.events[:, 2]
    group_ids = [subject] * len(labels)

    return epochs, labels, group_ids