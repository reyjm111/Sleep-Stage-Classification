from scipy.signal import welch
import numpy as np
import pandas as pd

def feature_extraction(epochs):

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100),
    }

    fs = 200
    epoch_len = 30
    nperseg = fs * epoch_len
    eps = 1e-12 # to prevent division by 0

    features = []

    # iterate throug each epoch and extract features
    for epoch in epochs:
        # keep only channels with no NaNs
        valid_epoch = epoch[~np.isnan(epoch).any(axis=1)]

        if valid_epoch.shape[0] == 0:
            features.append({
                'delta_abs_bandpower': np.nan,
                'theta_abs_bandpower': np.nan,
                'alpha_abs_bandpower': np.nan,
                'beta_abs_bandpower': np.nan,
                'gamma_abs_bandpower': np.nan,
                'delta_rel_bandpower': np.nan,
                'theta_rel_bandpower': np.nan,
                'alpha_rel_bandpower': np.nan,
                'beta_rel_bandpower': np.nan,
                'gamma_rel_bandpower': np.nan,
                'delta_theta_ratio': np.nan,
                'delta_alpha_ratio': np.nan,
                'theta_alpha_ratio': np.nan,
                'delta_beta_ratio': np.nan,
                'alpha_beta_ratio': np.nan,
                'hjorth_activity': np.nan,
                'hjorth_mobility': np.nan,
                'hjorth_complexity': np.nan,
                'spectral_entropy': np.nan,
            })
            continue

        # extract PSD values 
        f, Pxx = welch(
            valid_epoch,
            fs=fs,
            nperseg=nperseg
        )

        abs_power_features = {}
        rel_power_features = {}
        ratio_features = {}
        hjorth_features = {}
        spectral_entropy_features = {}

        # absolute bandpower
        for band, (f_low, f_high) in bands.items():
            band_mask = (f >= f_low) & (f <= f_high)
            band_psd = Pxx[:, band_mask]
            bandpower_per_ch = np.trapezoid(band_psd, f[band_mask], axis=1) # area under the curve per ch
            abs_power_features[f'{band}_abs_bandpower'] = np.nanmean(bandpower_per_ch) # average across channels

        # relative bandpower
        total_power = sum(abs_power_features.values()) 
        for band in bands:
            key = f'{band}_abs_bandpower'
            rel_power_features[f'{band}_rel_bandpower'] = abs_power_features[key] / (total_power + eps) 

        # band-to-band ratios
        ratio_features["delta_theta_ratio"] = abs_power_features["delta_abs_bandpower"] / (abs_power_features["theta_abs_bandpower"] + eps)
        ratio_features["delta_alpha_ratio"] = abs_power_features["delta_abs_bandpower"] / (abs_power_features["alpha_abs_bandpower"] + eps)
        ratio_features["theta_alpha_ratio"] = abs_power_features["theta_abs_bandpower"] / (abs_power_features["alpha_abs_bandpower"] + eps)
        ratio_features["delta_beta_ratio"]  = abs_power_features["delta_abs_bandpower"] / (abs_power_features["beta_abs_bandpower"] + eps)
        ratio_features["alpha_beta_ratio"]  = abs_power_features["alpha_abs_bandpower"] / (abs_power_features["beta_abs_bandpower"] + eps)

        # Hjord parameters
        activity_per_ch = np.var(valid_epoch, axis=1) # variance of the epoch signal

        first_deriv = np.diff(valid_epoch, axis=1)
        second_deriv = np.diff(first_deriv, axis=1)

        var_d1 = np.var(first_deriv, axis=1)
        var_d2 = np.var(second_deriv, axis=1)

        mobility_per_ch = np.sqrt(var_d1 / (activity_per_ch + eps)) # mean frequency of the signal
        complexity_per_ch = np.sqrt(var_d2 / (var_d1 + eps)) / (mobility_per_ch + eps) # compares signal's similarity to a pure sine wave

        hjorth_features["hjorth_activity"] = np.mean(activity_per_ch)
        hjorth_features["hjorth_mobility"] = np.mean(mobility_per_ch)
        hjorth_features["hjorth_complexity"] = np.mean(complexity_per_ch)

        # spectral entropy
        psd_sum = np.sum(Pxx, axis=1, keepdims=True)
        psd_norm = Pxx / (psd_sum + eps)
        entropy_per_ch = -np.sum(psd_norm * np.log2(psd_norm + eps), axis=1)
        spectral_entropy_features["spectral_entropy"] = np.mean(entropy_per_ch)

        epoch_features = (
            abs_power_features
            | rel_power_features
            | ratio_features
            | hjorth_features
            | spectral_entropy_features
        )

        features.append(epoch_features)

    features_df = pd.DataFrame(features)

    # choose which columns to log-transform
    log_cols = [
        "delta_abs_bandpower",
        "theta_abs_bandpower",
        "alpha_abs_bandpower",
        "beta_abs_bandpower",
        "gamma_abs_bandpower",
        "delta_theta_ratio",
        "delta_alpha_ratio",
        "theta_alpha_ratio",
        "delta_beta_ratio",
        "alpha_beta_ratio",
    ]

    # log transform for commonly skewed features
    features_df[log_cols] = np.log10(features_df[log_cols] + 1e-12)

    return features_df 