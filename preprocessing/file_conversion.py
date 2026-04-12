import numpy as np
from pathlib import Path

from .preprocess import preprocess
from .features import feature_extraction

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
                epochs, y, groups = preprocess(str(eeg_file), str(ann_file)) # preprocess eeg

                if epochs is None or y is None or groups is None:
                    print(f"Skipping {ses_dir}: preprocess returned None")
                    continue
                
                X_features_df = feature_extraction(epochs) # extract features
                valid_rows = ~X_features_df.isna().all(axis=1) 

                if valid_rows.sum() == 0:
                    print(f"Skipping {ses_dir}: all extracted feature rows are NaN")
                    continue

                X_features = X_features_df.loc[valid_rows].to_numpy(dtype=np.float32)
                y_valid = np.asarray(y)[valid_rows.to_numpy()]
                groups_valid = np.asarray(groups)[valid_rows.to_numpy()]

                print(f"  Feature matrix shape: {X_features.shape}")

                all_X.append(X_features)
                all_y.append(y_valid)
                all_groups.append(groups_valid)

            except Exception as e:
                print(f"Failed on {ses_dir}: {e}")

    if len(all_X) == 0:
        raise ValueError("No valid sessions were processed successfully.")

    # combine across all subjects/sessions
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    return X, y, groups