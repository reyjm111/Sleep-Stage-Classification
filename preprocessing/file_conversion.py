import numpy as np
from pathlib import Path

from .preprocess import preprocess

def file_conversion(path):

    # root folder containing sub-001, sub-002, ...
    root_dir = Path(path)

    all_X = []
    all_y = []
    all_groups = []

    for sub_dir in sorted(root_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue

            eeg_files = list(ses_dir.glob("*_eeg.set"))
            ann_files = list(ses_dir.glob("*scoring1_events.tsv"))

            if len(eeg_files) == 0:
                print(f"Skipping {ses_dir}: no EEG .set file found")
                continue

            if len(ann_files) == 0:
                print(f"Skipping {ses_dir}: no annotation .tsv file found")
                continue

            eeg_file = eeg_files[0]
            ann_file = ann_files[0]

            print(f"Processing: {eeg_file.name}")

            try:
                X, y, groups = preprocess(str(eeg_file), str(ann_file))

                # convert epochs object to numpy array
                # shape: (n_epochs, n_channels, n_times)

                all_X.append(X_sub)
                all_y.append(np.asarray(labels))
                all_groups.append(np.asarray(group_ids))

            except Exception as e:
                print(f"Failed on {ses_dir}: {e}")

    # combine across all subjects/sessions
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    
    return X, y, groups