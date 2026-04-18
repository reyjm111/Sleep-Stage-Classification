from scipy.signal import welch
import numpy as np
import pandas as pd
from specparam import SpectralModel

def feature_extraction(epochs):

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }

    fs = 200
    epoch_len = 30
    nperseg = 1024
    noverlap=512
    eps = 1e-12 # to prevent division by 0

    max_nan_frac_per_ch=0.05
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
            features.append({
                # bandpower
                'delta_abs_bandpower': np.nan,
                'theta_abs_bandpower': np.nan,
                'alpha_abs_bandpower': np.nan,
                'beta_abs_bandpower': np.nan,
                'gamma_abs_bandpower': np.nan,

                'delta_std_bandpower': np.nan,
                'theta_std_bandpower': np.nan,
                'alpha_std_bandpower': np.nan,
                'beta_std_bandpower': np.nan,
                'gamma_std_bandpower': np.nan,

                # LR diff
                'delta_lr_diff': np.nan,
                'theta_lr_diff': np.nan,
                'alpha_lr_diff': np.nan,
                'beta_lr_diff': np.nan,
                'gamma_lr_diff': np.nan,

                # relative
                'delta_rel_bandpower': np.nan,
                'theta_rel_bandpower': np.nan,
                'alpha_rel_bandpower': np.nan,
                'beta_rel_bandpower': np.nan,
                'gamma_rel_bandpower': np.nan,

                # ratios
                'delta_theta_ratio': np.nan,
                'delta_alpha_ratio': np.nan,
                'theta_alpha_ratio': np.nan,
                'delta_beta_ratio': np.nan,
                'alpha_beta_ratio': np.nan,
                'theta_beta_ratio': np.nan,
                'alpha_delta_ratio': np.nan,
                'theta_delta_ratio': np.nan,

                # peak
                'peak_frequency': np.nan,

                # Hjorth
                'hjorth_activity': np.nan,
                'hjorth_mobility': np.nan,
                'hjorth_complexity': np.nan,

                # entropy
                'spectral_entropy': np.nan,

                # specparam
                'aperiodic_exponent_mean': np.nan,
                'aperiodic_exponent_std': np.nan,
                'aperiodic_offset_mean': np.nan,
                'aperiodic_offset_std': np.nan,
                'alpha_peak_power_mean': np.nan,
                'alpha_peak_freq_mean': np.nan,
                'aperiodic_exponent_lr_diff': np.nan,
            })
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
            features.append({
                # bandpower
                'delta_abs_bandpower': np.nan,
                'theta_abs_bandpower': np.nan,
                'alpha_abs_bandpower': np.nan,
                'beta_abs_bandpower': np.nan,
                'gamma_abs_bandpower': np.nan,

                'delta_std_bandpower': np.nan,
                'theta_std_bandpower': np.nan,
                'alpha_std_bandpower': np.nan,
                'beta_std_bandpower': np.nan,
                'gamma_std_bandpower': np.nan,

                # LR diff
                'delta_lr_diff': np.nan,
                'theta_lr_diff': np.nan,
                'alpha_lr_diff': np.nan,
                'beta_lr_diff': np.nan,
                'gamma_lr_diff': np.nan,

                # relative
                'delta_rel_bandpower': np.nan,
                'theta_rel_bandpower': np.nan,
                'alpha_rel_bandpower': np.nan,
                'beta_rel_bandpower': np.nan,
                'gamma_rel_bandpower': np.nan,

                # ratios
                'delta_theta_ratio': np.nan,
                'delta_alpha_ratio': np.nan,
                'theta_alpha_ratio': np.nan,
                'delta_beta_ratio': np.nan,
                'alpha_beta_ratio': np.nan,
                'theta_beta_ratio': np.nan,
                'alpha_delta_ratio': np.nan,
                'theta_delta_ratio': np.nan,

                # peak
                'peak_frequency': np.nan,

                # Hjorth
                'hjorth_activity': np.nan,
                'hjorth_mobility': np.nan,
                'hjorth_complexity': np.nan,

                # entropy
                'spectral_entropy': np.nan,

                # specparam
                'aperiodic_exponent_mean': np.nan,
                'aperiodic_exponent_std': np.nan,
                'aperiodic_offset_mean': np.nan,
                'aperiodic_offset_std': np.nan,
                'alpha_peak_power_mean': np.nan,
                'alpha_peak_freq_mean': np.nan,
                'aperiodic_exponent_lr_diff': np.nan,
            })
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
        fm = SpectralModel(
            peak_width_limits=[1, 12],
            max_n_peaks=6,
            min_peak_height=0.1,
            verbose=False
        )
        
        aperiodic_exponents = []
        aperiodic_offsets = []
        alpha_peak_powers = []
        alpha_peak_freqs = []

        for ch in range(Pxx.shape[0]):

            fm.fit(f, Pxx[ch])

            # aperiodic
            offset, exponent = fm.aperiodic_params_
            aperiodic_offsets.append(offset)
            aperiodic_exponents.append(exponent)

            # peaks
            peaks = fm.peak_params_
            
            # extract alpha peak (8–13 Hz)
            alpha_peak = [p for p in peaks if 8 <= p[0] <= 13]

            if len(alpha_peak) > 0:
                best_peak = max(alpha_peak, key=lambda x: x[1])  # highest power
                alpha_peak_powers.append(best_peak[1])
                alpha_peak_freqs.append(best_peak[0])

            else:
                alpha_peak_powers.append(np.nan)
                alpha_peak_freqs.append(np.nan)

        if len(aperiodic_exponents) == 2:
            lr_diff = aperiodic_exponents[0] - aperiodic_exponents[1]
        else:
            lr_diff = 0.0

        specparam_features = {
            "aperiodic_exponent_mean": np.mean(aperiodic_exponents), 
            "aperiodic_exponent_std": np.std(aperiodic_exponents), 
            "aperiodic_offset_mean": np.mean(aperiodic_offsets), 
            "aperiodic_offset_std": np.std(aperiodic_offsets), 
            "alpha_peak_power_mean": np.nanmean(alpha_peak_powers),
            "alpha_peak_freq_mean": np.nanmean(alpha_peak_freqs), 
            "aperiodic_exponent_lr_diff": lr_diff
        }

        epoch_features = (
            abs_power_features
            | rel_power_features
            | ratio_features
            | peak_features
            | hjorth_features
            | spectral_entropy_features
            | specparam_features
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