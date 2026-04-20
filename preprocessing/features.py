from scipy.signal import welch, butter, filtfilt
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd
from specparam import SpectralModel
from antropy import perm_entropy, higuchi_fd

ALL_FEATURE_KEYS = [
    # bandpower
    'delta_abs_bandpower','theta_abs_bandpower','alpha_abs_bandpower','beta_abs_bandpower','gamma_abs_bandpower',
    'delta_std_bandpower','theta_std_bandpower','alpha_std_bandpower','beta_std_bandpower','gamma_std_bandpower',
    'delta_lr_diff','theta_lr_diff','alpha_lr_diff','beta_lr_diff','gamma_lr_diff',

    # relative
    'delta_rel_bandpower','theta_rel_bandpower','alpha_rel_bandpower','beta_rel_bandpower','gamma_rel_bandpower',

    # ratios
    'delta_theta_ratio','delta_alpha_ratio','theta_alpha_ratio','delta_beta_ratio', 'high_freq_ratio', 
    'alpha_beta_ratio','theta_beta_ratio','alpha_delta_ratio','theta_delta_ratio',

    # peak
    'peak_frequency',

    # hjorth
    'hjorth_activity','hjorth_mobility','hjorth_complexity',

    # entropy
    'spectral_entropy',

    # specparam
    'aperiodic_exponent_mean','aperiodic_exponent_std',
    'aperiodic_offset_mean','aperiodic_offset_std',
    'aperiodic_exponent_lr_diff',

    # new features
    'sef95','perm_entropy','higuchi_fd','kurtosis','skewness',
    'line_length','temporal_variance',
    'spindle_count','spindle_power',
    'zcr','rms'
]

def bandpass(signal, fs, low, high):

            b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')

            return filtfilt(b, a, signal)

def chunk_var(sig, n=4):

            chunks = np.array_split(sig, n)

            return np.var([np.var(c) for c in chunks])

def spectral_edge_frequency(freqs, psd, percentile=0.95):

            cumulative = np.cumsum(psd)
            cumulative = cumulative / (cumulative[-1] + 1e-12)

            return freqs[np.searchsorted(cumulative, percentile)]

def feature_extraction(epochs):

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }

    fs = 200
    nperseg = 1024
    noverlap=512
    eps = 1e-12 # to prevent division by 0

    max_nan_frac_per_ch=0.5
    min_channels=1
    features = []

    # iterate throug each epoch and extract features
    for epoch in epochs:
        # keep only channels with no NaNs
        nan_frac_per_ch = np.isnan(epoch).mean(axis=1)

        # keep channels with limited missingness
        keep_mask = nan_frac_per_ch <= max_nan_frac_per_ch
        valid_epoch = epoch[keep_mask]
        if valid_epoch.shape[0] < min_channels:
            features.append({k: np.nan for k in ALL_FEATURE_KEYS})
            continue
        
        # fill remaining NaNs in each channel using linear interpolation
        for ch in range(valid_epoch.shape[0]):
            x = valid_epoch[ch]
            nan_mask = np.isnan(x)

            if nan_mask.any():
                good_idx = np.where(~nan_mask)[0]

                if len(good_idx) == 0:
                    continue
                elif len(good_idx) == 1:
                    x[nan_mask] = x[good_idx[0]]
                else:
                    x[nan_mask] = np.interp(
                        np.where(nan_mask)[0],
                        good_idx,
                        x[good_idx]
                    )

                valid_epoch[ch] = x

        # final safeguard
        if np.isnan(valid_epoch).any():
            features.append({k: np.nan for k in ALL_FEATURE_KEYS})
            continue
        
        # extract PSD values 
        f, Pxx = welch(
            valid_epoch,
            fs=fs,
            nperseg=nperseg, 
            noverlap=noverlap, 
            axis=1
        )
        
        abs_power_features = {}
        rel_power_features = {}
        ratio_features = {}
        hjorth_features = {}
        spectral_entropy_features = {}
        peak_features = {}

        # Peak Freq
        peak_freqs = f[np.argmax(Pxx, axis=1)]
        peak_features["peak_frequency"] = np.mean(peak_freqs)

        # absolute bandpower
        for band, (f_low, f_high) in bands.items():
            band_mask = (f >= f_low) & (f <= f_high)
            band_psd = Pxx[:, band_mask]
            bandpower_per_ch = np.trapezoid(band_psd, f[band_mask], axis=1) # area under the curve per ch
            abs_power_features[f'{band}_abs_bandpower'] = np.nanmean(bandpower_per_ch) # average across channels
            abs_power_features[f'{band}_std_bandpower'] = np.nanstd(bandpower_per_ch) # variability across channels

            # lateralized power differences
            if bandpower_per_ch.shape[0] == 2:
                abs_power_features[f'{band}_lr_diff'] = bandpower_per_ch[0] - bandpower_per_ch[1]
            else:
                abs_power_features[f'{band}_lr_diff'] = 0.0

        # relative bandpower
        total_power = sum(abs_power_features[f"{band}_abs_bandpower"] for band in bands)        
        
        for band in bands:
            key = f'{band}_abs_bandpower'
            rel_power_features[f'{band}_rel_bandpower'] = abs_power_features[key] / (total_power + eps) 

        # band-to-band ratios
        ratio_features["delta_theta_ratio"] = abs_power_features["delta_abs_bandpower"] / (abs_power_features["theta_abs_bandpower"] + eps)
        ratio_features["delta_alpha_ratio"] = abs_power_features["delta_abs_bandpower"] / (abs_power_features["alpha_abs_bandpower"] + eps)
        ratio_features["theta_alpha_ratio"] = abs_power_features["theta_abs_bandpower"] / (abs_power_features["alpha_abs_bandpower"] + eps)
        ratio_features["delta_beta_ratio"]  = abs_power_features["delta_abs_bandpower"] / (abs_power_features["beta_abs_bandpower"] + eps)
        ratio_features["alpha_beta_ratio"]  = abs_power_features["alpha_abs_bandpower"] / (abs_power_features["beta_abs_bandpower"] + eps)
        ratio_features["theta_beta_ratio"] = abs_power_features["theta_abs_bandpower"] / (abs_power_features["beta_abs_bandpower"] + eps)
        ratio_features["alpha_delta_ratio"] = abs_power_features["alpha_abs_bandpower"] / (abs_power_features["delta_abs_bandpower"] + eps)
        ratio_features["theta_delta_ratio"] = abs_power_features["theta_abs_bandpower"] / (abs_power_features["delta_abs_bandpower"] + eps)
        ratio_features["high_freq_ratio"] = (abs_power_features["beta_abs_bandpower"] + abs_power_features["gamma_abs_bandpower"]) / (total_power + eps)

        # Hjorth parameters
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
        entropy_per_ch /= np.log2(psd_norm.shape[1])        
        spectral_entropy_features["spectral_entropy"] = np.mean(entropy_per_ch)

        # 1/f features
        aperiodic_exponents = []
        aperiodic_offsets = []

        for ch in range(Pxx.shape[0]):

            fm = SpectralModel(
            peak_width_limits=[1, 12],
            max_n_peaks=6,
            min_peak_height=0.1,
            verbose=False)

            try:
                mask = (f >= 1) & (f <= 40)
                fm.fit(f[mask], Pxx[ch][mask])

                if fm.has_model:
                    offset, exponent = fm.get_params('aperiodic')
                else:
                    offset, exponent = np.nan, np.nan

            except Exception:
                offset, exponent = np.nan, np.nan

            aperiodic_offsets.append(offset)
            aperiodic_exponents.append(exponent)

        if len(aperiodic_exponents) == 2:
            lr_diff = aperiodic_exponents[0] - aperiodic_exponents[1]
        else:
            lr_diff = 0.0

        specparam_features = {
            "aperiodic_exponent_mean": np.nanmean(aperiodic_exponents), 
            "aperiodic_exponent_std": np.nanstd(aperiodic_exponents), 
            "aperiodic_offset_mean": np.nanmean(aperiodic_offsets), 
            "aperiodic_offset_std": np.nanstd(aperiodic_offsets),
            "aperiodic_exponent_lr_diff": lr_diff
        }

        # spectral edge frequency
        sef_per_ch = [spectral_edge_frequency(f, Pxx[ch]) for ch in range(Pxx.shape[0])]

        # Permutation entropy
        pe_per_ch = [perm_entropy(valid_epoch[ch], normalize=True) for ch in range(valid_epoch.shape[0])]

        # Fractal dimension
        fd_per_ch = [higuchi_fd(valid_epoch[ch]) for ch in range(valid_epoch.shape[0])]

        # Kurtosis
        kurt_per_ch = [kurtosis(valid_epoch[ch]) for ch in range(valid_epoch.shape[0])]

        # Skew
        skew_per_ch = [skew(valid_epoch[ch]) for ch in range(valid_epoch.shape[0])]

        # Line length
        line_length = np.mean([np.sum(np.abs(np.diff(valid_epoch[ch]))) for ch in range(valid_epoch.shape[0])])

        # Zero Crossings
        zcr = np.mean([np.mean(np.diff(np.sign(valid_epoch[ch])) != 0)for ch in range(valid_epoch.shape[0])])

        # RMS
        rms = np.mean([np.sqrt(np.mean(valid_epoch[ch]**2))for ch in range(valid_epoch.shape[0])])

        # Spindles
        spindle_counts = []
        spindle_power = []

        for ch in range(valid_epoch.shape[0]):
            sig = bandpass(valid_epoch[ch], fs, 11, 16)

            envelope = np.abs(sig)
            threshold = np.percentile(envelope, 95)
            events = envelope > threshold
            count = np.sum(np.diff(events.astype(int)) == 1)

            spindle_counts.append(count)
            spindle_power.append(np.mean(envelope))

        spindle_features = {
            "spindle_count": np.mean(spindle_counts),
            "spindle_power": np.mean(spindle_power),
        }

        # Temporal variability
        temporal_var = np.mean([chunk_var(valid_epoch[ch])for ch in range(valid_epoch.shape[0])])

        other_features = {
            "sef95": np.nanmean(sef_per_ch),
            "perm_entropy": np.nanmean(pe_per_ch),
            "higuchi_fd": np.nanmean(fd_per_ch),
            "kurtosis": np.nanmean(kurt_per_ch),
            "skewness": np.nanmean(skew_per_ch),
            "line_length": line_length,
            "temporal_variance": temporal_var,
            "spindle_count": spindle_features["spindle_count"],
            "spindle_power": spindle_features["spindle_power"], 
            "zcr": zcr, 
            "rms": rms, 
        }     

        epoch_features = (
            abs_power_features
            | rel_power_features
            | ratio_features
            | peak_features
            | hjorth_features
            | spectral_entropy_features
            | specparam_features 
            | other_features
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
        "theta_beta_ratio", 
        "alpha_delta_ratio", 
        "theta_delta_ratio", 
        "delta_std_bandpower",
        "theta_std_bandpower",
        "alpha_std_bandpower",
        "beta_std_bandpower",
        "gamma_std_bandpower"
        ]

    # log transform for commonly skewed features
    features_df[log_cols] = np.log10(features_df[log_cols] + 1e-12)

    return features_df 