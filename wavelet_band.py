#!/usr/bin/env python3
# cwt_global_peakband_cropped_chartonly_gui_summary_bandmean.py
#
# Purpose:
#   - Read CSV time series (first 2 columns: time, value)
#   - Optional resampling to uniform dt
#   - Detrend / z-score
#   - Continuous Wavelet Transform (CWT)
#   - Determine ONE representative peak period from time-averaged CWT power,
#     but using the mean power within ±peak_band_percent around each candidate period
#   - Crop the scalogram vertically to only global_peak ± peak_band_percent (%)
#   - Save chart-only PNGs (only colored panel; no title, no axes, no colorbar)
#   - Save per-file CSV
#   - Save batch summary CSV
#
# Outputs for each CSV:
#   - <stem>_CWT_global_peak.csv
#   - <stem>_CWT_global_peakband_cropped_chartonly_logY.png
#   - <stem>_CWT_global_peakband_cropped_chartonly_linearY.png
#
# Batch summary output:
#   - CWT_global_peak_summary.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------
# Core I/O & preprocessing
# -----------------------
def read_two_columns_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    except Exception:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 2:
            raise ValueError("CSV must have >=2 columns")
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 10:
        raise ValueError("Not enough valid rows.")
    if np.any(np.diff(t) < 0):
        idx = np.argsort(t)
        t, y = t[idx], y[idx]
    return t, y


def detrend_signal(t: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "median":
        return y - np.median(y)
    if mode == "linear":
        A = np.vstack([t, np.ones_like(t)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return y - (m * t + b)
    raise ValueError(f"Unknown detrend: {mode}")


def resample_uniform(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid dt computed from time column.")
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    yu = np.interp(tu, t, y)
    return tu, yu, dt


# -----------------------
# Interpolation helpers
# -----------------------
def make_common_period_grid(pmin: float, pmax: float, n: int, scale: str) -> np.ndarray:
    if scale == "log":
        return np.exp(np.linspace(np.log(pmin), np.log(pmax), n))
    return np.linspace(pmin, pmax, n)


def interp_time_period_to_grid(
    t_src: np.ndarray,
    period_src: np.ndarray,
    P_src: np.ndarray,
    t_grid: np.ndarray,
    period_grid: np.ndarray
) -> np.ndarray:
    """
    P_src shape: (n_period_src, n_time_src)
    Returns P_grid shape: (n_period_grid, n_time_grid)
    """
    order = np.argsort(period_src)
    ps = period_src[order]
    Pp = P_src[order, :]

    P1 = np.empty((len(period_grid), Pp.shape[1]), dtype=float)
    for j in range(Pp.shape[1]):
        P1[:, j] = np.interp(period_grid, ps, Pp[:, j], left=np.nan, right=np.nan)

    P2 = np.empty((len(period_grid), len(t_grid)), dtype=float)
    for i in range(len(period_grid)):
        row = P1[i, :]
        m = np.isfinite(row) & np.isfinite(t_src)
        if np.sum(m) < 2:
            P2[i, :] = np.nan
        else:
            P2[i, :] = np.interp(t_grid, t_src[m], row[m], left=np.nan, right=np.nan)
    return P2


# -----------------------
# CWT
# -----------------------
def compute_cwt_power(y: np.ndarray, dt: float, wavelet: str, dj: float, pmin: float, pmax: float):
    w = pywt.ContinuousWavelet(wavelet)
    fc = pywt.central_frequency(w)

    smin = pmin * fc / dt
    smax = pmax * fc / dt
    if not (np.isfinite(smin) and np.isfinite(smax) and smin > 0 and smax > smin):
        raise ValueError("Invalid CWT scale range computed from period range.")

    n_scales = int(np.floor(np.log2(smax / smin) / dj)) + 1
    n_scales = max(4, n_scales)

    scales = smin * (2 ** (np.arange(n_scales) * dj))
    coef, freqs = pywt.cwt(y, scales, wavelet, sampling_period=dt)
    power = np.abs(coef) ** 2
    period = 1.0 / np.maximum(freqs, 1e-30)
    return period, power


# -----------------------
# Global peak extraction by band-mean
# -----------------------
def extract_global_peak_period_bandmean(
    period_grid: np.ndarray,
    power_grid: np.ndarray,
    band_percent: float
) -> pd.DataFrame:
    """
    For each candidate center period in period_grid:
      - define band = center ± band_percent (%)
      - compute mean CWT power within that band over all time
    Choose the center period whose band-mean power is maximal.
    """
    if band_percent < 0:
        raise ValueError("band_percent must be >= 0")

    mean_power_by_period = np.nanmean(power_grid, axis=1)
    if not np.any(np.isfinite(mean_power_by_period)):
        raise ValueError("No finite power values found.")

    frac = band_percent / 100.0
    band_mean_scores = np.full(len(period_grid), np.nan, dtype=float)
    band_n_bins = np.zeros(len(period_grid), dtype=int)

    for k, center in enumerate(period_grid):
        if not np.isfinite(center) or center <= 0:
            continue

        p_lo = center * (1.0 - frac)
        p_hi = center * (1.0 + frac)

        m = (period_grid >= p_lo) & (period_grid <= p_hi)
        n_bins = int(np.sum(m))
        band_n_bins[k] = n_bins
        if n_bins < 1:
            continue

        band_vals = power_grid[m, :]
        if np.any(np.isfinite(band_vals)):
            band_mean_scores[k] = float(np.nanmean(band_vals))

    if not np.any(np.isfinite(band_mean_scores)):
        raise ValueError("No finite band-mean scores found.")

    best_k = int(np.nanargmax(band_mean_scores))
    peak_period = float(period_grid[best_k])
    peak_freq = 1.0 / peak_period if peak_period > 0 else np.nan
    peak_band_mean_power = float(band_mean_scores[best_k])

    p_lo = peak_period * (1.0 - frac)
    p_hi = peak_period * (1.0 + frac)
    final_mask = (period_grid >= p_lo) & (period_grid <= p_hi)
    cropped_n_period_bins = int(np.sum(final_mask))

    return pd.DataFrame({
        "global_peak_period": [peak_period],
        "global_peak_freq": [peak_freq],
        "global_peak_band_mean_power": [peak_band_mean_power],
        "peak_band_percent": [band_percent],
        "band_period_min": [p_lo],
        "band_period_max": [p_hi],
        "cropped_n_period_bins": [cropped_n_period_bins]
    })


# -----------------------
# Crop global peak band
# -----------------------
def crop_global_peak_band(
    period_grid: np.ndarray,
    power_grid: np.ndarray,
    global_peak_period: float,
    band_percent: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop period axis to global_peak_period ± band_percent (%).
    Returns cropped period grid and cropped power grid.
    """
    frac = band_percent / 100.0
    if frac < 0:
        raise ValueError("band_percent must be >= 0")
    if not np.isfinite(global_peak_period) or global_peak_period <= 0:
        raise ValueError("Invalid global_peak_period")

    p_lo = global_peak_period * (1.0 - frac)
    p_hi = global_peak_period * (1.0 + frac)

    m = (period_grid >= p_lo) & (period_grid <= p_hi)
    if not np.any(m):
        raise ValueError("No period bins found inside the requested peak band.")

    return period_grid[m], power_grid[m, :]


# -----------------------
# Plot helpers
# -----------------------
def _time_edges_from_centers(t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.array([t[0] - 0.5, t[0] + 0.5])
    dtg = float(np.median(np.diff(t)))
    return np.concatenate(([t[0] - dtg / 2], (t[:-1] + t[1:]) / 2, [t[-1] + dtg / 2]))


def _edges_from_centers(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.array([x[0] - 0.5, x[0] + 0.5])

    d = np.diff(x)
    x_edges = np.empty(len(x) + 1, dtype=float)
    x_edges[1:-1] = (x[:-1] + x[1:]) / 2
    x_edges[0] = x[0] - d[0] / 2
    x_edges[-1] = x[-1] + d[-1] / 2
    return x_edges


def _figure_from_px(width_px: int, height_px: int, dpi: int):
    fig_w = max(1, width_px) / dpi
    fig_h = max(1, height_px) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    return fig


def _compute_vmin_vmax(data: np.ndarray, robust: bool = True, pct_low: float = 1.0, pct_high: float = 99.0):
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return 0.0, 1.0
    if robust:
        vmin = float(np.nanpercentile(vals, pct_low))
        vmax = float(np.nanpercentile(vals, pct_high))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def save_chart_only_scalogram(
    out_png: str,
    t_grid: np.ndarray,
    period_grid: np.ndarray,
    power_grid: np.ndarray,
    y_mode: str,
    dpi: int,
    width_px: int,
    height_px: int,
    cmap: str = "viridis",
    transparent: bool = False,
    robust_color: bool = True,
):
    fig = _figure_from_px(width_px, height_px, dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    t_edges = _time_edges_from_centers(t_grid)

    if y_mode == "log":
        y = np.log2(period_grid)
        y_edges = _edges_from_centers(y)
    elif y_mode == "linear":
        y_edges = _edges_from_centers(period_grid)
    else:
        raise ValueError("y_mode must be 'log' or 'linear'")

    vmin, vmax = _compute_vmin_vmax(power_grid, robust=robust_color)

    ax.pcolormesh(
        t_edges,
        y_edges,
        power_grid,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlim(t_edges[0], t_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    fig.savefig(
        out_png,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        transparent=transparent
    )
    plt.close(fig)


# -----------------------
# Pipeline per file
# -----------------------
def run_one_file(csv_path: str, cfg: dict, out_dir: str):
    t, y = read_two_columns_csv(csv_path)

    if cfg["resample_uniform"]:
        t, y, dt = resample_uniform(t, y)
    else:
        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid dt (time column).")

    y = detrend_signal(t, y, cfg["detrend"])
    if cfg["zscore"]:
        s = np.std(y)
        if s > 0:
            y = (y - np.mean(y)) / s

    T_total = float(t[-1] - t[0])
    nyq_period = 2.0 * dt
    pmin = cfg["period_min"] if cfg["period_min"] is not None else max(nyq_period, T_total / 200.0)
    pmax = cfg["period_max"] if cfg["period_max"] is not None else (T_total / 2.0)
    if not (np.isfinite(pmin) and np.isfinite(pmax) and pmax > pmin > 0):
        raise ValueError(f"Invalid period range: {pmin}..{pmax}")

    period_grid = make_common_period_grid(pmin, pmax, cfg["n_period"], cfg["period_scale"])
    t_grid = t

    per_cwt, pow_cwt = compute_cwt_power(y, dt, cfg["cwt_wavelet"], cfg["cwt_dj"], pmin, pmax)
    cwt_power_grid = interp_time_period_to_grid(t, per_cwt, pow_cwt, t_grid, period_grid)

    df_global_peak = extract_global_peak_period_bandmean(
        period_grid=period_grid,
        power_grid=cwt_power_grid,
        band_percent=cfg["peak_band_percent"]
    )
    global_peak_period = float(df_global_peak["global_peak_period"].iloc[0])

    cropped_period_grid, cropped_power_grid = crop_global_peak_band(
        period_grid=period_grid,
        power_grid=cwt_power_grid,
        global_peak_period=global_peak_period,
        band_percent=cfg["peak_band_percent"]
    )

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    base_out = out_dir if out_dir else os.path.dirname(csv_path)
    os.makedirs(base_out, exist_ok=True)

    out_csv_peak = os.path.join(base_out, f"{stem}_CWT_global_peak.csv")
    out_png_band_log = os.path.join(base_out, f"{stem}_CWT_global_peakband_cropped_chartonly_logY.png")
    out_png_band_lin = os.path.join(base_out, f"{stem}_CWT_global_peakband_cropped_chartonly_linearY.png")

    frac = cfg["peak_band_percent"] / 100.0
    band_period_min = global_peak_period * (1.0 - frac)
    band_period_max = global_peak_period * (1.0 + frac)

    # ensure per-file CSV includes all useful columns
    df_global_peak["peak_band_percent"] = cfg["peak_band_percent"]
    df_global_peak["band_period_min"] = band_period_min
    df_global_peak["band_period_max"] = band_period_max
    df_global_peak["cropped_n_period_bins"] = len(cropped_period_grid)
    df_global_peak.to_csv(out_csv_peak, index=False)

    save_chart_only_scalogram(
        out_png=out_png_band_log,
        t_grid=t_grid,
        period_grid=cropped_period_grid,
        power_grid=cropped_power_grid,
        y_mode="log",
        dpi=cfg["dpi"],
        width_px=cfg["width_px"],
        height_px=cfg["height_px"],
        cmap=cfg["cmap"],
        transparent=cfg["transparent_png"],
        robust_color=cfg["robust_color"],
    )

    save_chart_only_scalogram(
        out_png=out_png_band_lin,
        t_grid=t_grid,
        period_grid=cropped_period_grid,
        power_grid=cropped_power_grid,
        y_mode="linear",
        dpi=cfg["dpi"],
        width_px=cfg["width_px"],
        height_px=cfg["height_px"],
        cmap=cfg["cmap"],
        transparent=cfg["transparent_png"],
        robust_color=cfg["robust_color"],
    )

    summary_row = {
        "file_name": os.path.basename(csv_path),
        "file_path": csv_path,
        "output_dir": base_out,
        "global_peak_period": global_peak_period,
        "global_peak_freq": float(df_global_peak["global_peak_freq"].iloc[0]),
        "global_peak_band_mean_power": float(df_global_peak["global_peak_band_mean_power"].iloc[0]),
        "peak_band_percent": cfg["peak_band_percent"],
        "band_period_min": band_period_min,
        "band_period_max": band_period_max,
        "cropped_n_period_bins": int(len(cropped_period_grid)),
        "n_time_points": int(len(t_grid)),
        "dt": float(dt),
        "period_min_used": float(pmin),
        "period_max_used": float(pmax),
        "n_period": int(cfg["n_period"]),
        "period_scale": cfg["period_scale"],
        "cwt_wavelet": cfg["cwt_wavelet"],
        "cwt_dj": float(cfg["cwt_dj"]),
        "detrend": cfg["detrend"],
        "zscore": bool(cfg["zscore"]),
        "resample_uniform": bool(cfg["resample_uniform"]),
        "png_log_path": out_png_band_log,
        "png_linear_path": out_png_band_lin,
        "per_file_csv_path": out_csv_peak,
    }

    return out_csv_peak, out_png_band_log, out_png_band_lin, summary_row


# -----------------------
# GUI
# -----------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CWT global-peak band cropped chart-only image exporter")
        self.geometry("1060x840")

        self.selection = {"mode": "files", "files": [], "folder": ""}

        self.var_out_dir = tk.StringVar(value="")

        self.var_resample = tk.BooleanVar(value=True)
        self.var_detrend = tk.StringVar(value="median")
        self.var_zscore = tk.BooleanVar(value=True)

        self.var_pmin = tk.StringVar(value="")
        self.var_pmax = tk.StringVar(value="")
        self.var_nperiod = tk.StringVar(value="160")
        self.var_pscale = tk.StringVar(value="log")

        self.var_cwt_wavelet = tk.StringVar(value="morl")
        self.var_cwt_dj = tk.StringVar(value="0.02")

        self.var_peak_band_percent = tk.StringVar(value="10")

        self.var_dpi = tk.StringVar(value="200")
        self.var_width_px = tk.StringVar(value="2800")
        self.var_height_px = tk.StringVar(value="1160")
        self.var_cmap = tk.StringVar(value="viridis")
        self.var_transparent = tk.BooleanVar(value=False)
        self.var_robust_color = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        frm_sel = ttk.LabelFrame(self, text="Input selection")
        frm_sel.pack(fill="x", padx=10, pady=8)

        ttk.Button(frm_sel, text="Select CSV files...", command=self.pick_files).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Button(frm_sel, text="Select folder...", command=self.pick_folder).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        self.lbl_sel = ttk.Label(frm_sel, text="No files/folder selected.")
        self.lbl_sel.grid(row=1, column=0, columnspan=6, padx=6, pady=6, sticky="w")

        frm_pre = ttk.LabelFrame(self, text="Preprocess")
        frm_pre.pack(fill="x", padx=10, pady=8)

        ttk.Checkbutton(frm_pre, text="Resample to uniform dt (recommended)", variable=self.var_resample)\
            .grid(row=0, column=0, padx=6, pady=4, sticky="w")
        ttk.Checkbutton(frm_pre, text="Z-score normalize", variable=self.var_zscore)\
            .grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_pre, text="Detrend").grid(row=1, column=0, padx=6, pady=4, sticky="e")
        cb_d = ttk.Combobox(frm_pre, textvariable=self.var_detrend, width=16, state="readonly")
        cb_d["values"] = ("median", "linear", "none")
        cb_d.grid(row=1, column=1, padx=6, pady=4, sticky="w")

        frm_p = ttk.LabelFrame(self, text="Period axis / interpolation grid")
        frm_p.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_p, text="Period min (blank=auto)").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_p, textvariable=self.var_pmin, width=16).grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_p, text="Period max (blank=auto)").grid(row=0, column=2, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_p, textvariable=self.var_pmax, width=16).grid(row=0, column=3, padx=6, pady=4, sticky="w")

        ttk.Label(frm_p, text="#period grid").grid(row=1, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_p, textvariable=self.var_nperiod, width=16).grid(row=1, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_p, text="grid scale").grid(row=1, column=2, padx=6, pady=4, sticky="e")
        cb_s = ttk.Combobox(frm_p, textvariable=self.var_pscale, width=16, state="readonly")
        cb_s["values"] = ("log", "linear")
        cb_s.grid(row=1, column=3, padx=6, pady=4, sticky="w")

        frm_c = ttk.LabelFrame(self, text="CWT parameters")
        frm_c.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_c, text="wavelet").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        cb_cw = ttk.Combobox(frm_c, textvariable=self.var_cwt_wavelet, width=18, state="readonly")
        cb_cw["values"] = ("morl", "mexh", "gaus1", "gaus2", "cmor1.5-1.0", "cmor2.0-1.0")
        cb_cw.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_c, text="dj").grid(row=0, column=2, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_c, textvariable=self.var_cwt_dj, width=12).grid(row=0, column=3, padx=6, pady=4, sticky="w")
        ttk.Label(frm_c, text="(smaller = denser / slower)").grid(row=0, column=4, padx=6, pady=4, sticky="w")

        frm_peak = ttk.LabelFrame(self, text="Global peak band")
        frm_peak.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_peak, text="Peak band (± %) used both for selection and cropping").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Entry(frm_peak, textvariable=self.var_peak_band_percent, width=12).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(frm_peak, text="Peak is chosen by maximal band-mean power").grid(row=0, column=2, padx=10, pady=6, sticky="w")

        frm_img = ttk.LabelFrame(self, text="Chart-only output image")
        frm_img.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_img, text="Output folder (optional)").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Entry(frm_img, textvariable=self.var_out_dir, width=70).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Button(frm_img, text="Browse...", command=self.pick_out_dir).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(frm_img, text="Width (px)").grid(row=1, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_img, textvariable=self.var_width_px, width=12).grid(row=1, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_img, text="Height (px)").grid(row=1, column=2, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_img, textvariable=self.var_height_px, width=12).grid(row=1, column=3, padx=6, pady=4, sticky="w")

        ttk.Label(frm_img, text="DPI").grid(row=2, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_img, textvariable=self.var_dpi, width=12).grid(row=2, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_img, text="Colormap").grid(row=2, column=2, padx=6, pady=4, sticky="e")
        cb_map = ttk.Combobox(frm_img, textvariable=self.var_cmap, width=16, state="readonly")
        cb_map["values"] = ("viridis", "plasma", "inferno", "magma", "cividis", "turbo", "jet")
        cb_map.grid(row=2, column=3, padx=6, pady=4, sticky="w")

        ttk.Checkbutton(frm_img, text="Transparent PNG background", variable=self.var_transparent)\
            .grid(row=3, column=0, padx=6, pady=4, sticky="w")
        ttk.Checkbutton(frm_img, text="Robust color scaling (1-99 percentile)", variable=self.var_robust_color)\
            .grid(row=3, column=1, columnspan=2, padx=6, pady=4, sticky="w")

        ttk.Label(frm_img, text="Saved files: _CWT_global_peakband_cropped_chartonly_logY.png / linearY.png").grid(
            row=4, column=0, columnspan=4, padx=6, pady=6, sticky="w"
        )

        frm_run = ttk.Frame(self)
        frm_run.pack(fill="both", expand=True, padx=10, pady=8)

        self.btn_run = ttk.Button(
            frm_run,
            text="Run batch (per-file CSV + cropped chart-only PNGs + summary.csv)",
            command=self.run_batch
        )
        self.btn_run.pack(anchor="w")

        self.txt = tk.Text(frm_run, height=14, wrap="word")
        self.txt.pack(fill="both", expand=True, pady=8)
        self._log("Ready.\n")

    def _log(self, s: str):
        self.txt.insert("end", s)
        self.txt.see("end")
        self.update_idletasks()

    def pick_files(self):
        files = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        files = list(files)
        if files:
            self.selection = {"mode": "files", "files": files, "folder": ""}
            self.lbl_sel.config(text=f"Selected {len(files)} file(s).")
            self._log(f"[SELECT] {len(files)} file(s)\n")

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing CSVs")
        if folder:
            self.selection = {"mode": "folder", "files": [], "folder": folder}
            self.lbl_sel.config(text=f"Selected folder: {folder}")
            self._log(f"[SELECT] Folder: {folder}\n")

    def pick_out_dir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.var_out_dir.set(folder)

    def _parse_float_or_none(self, s: str):
        s = s.strip()
        if s == "":
            return None
        return float(s)

    def _cfg(self) -> dict:
        cfg = dict(
            resample_uniform=bool(self.var_resample.get()),
            detrend=self.var_detrend.get().strip(),
            zscore=bool(self.var_zscore.get()),
            period_min=self._parse_float_or_none(self.var_pmin.get()),
            period_max=self._parse_float_or_none(self.var_pmax.get()),
            n_period=int(float(self.var_nperiod.get().strip())),
            period_scale=self.var_pscale.get().strip(),
            cwt_wavelet=self.var_cwt_wavelet.get().strip(),
            cwt_dj=float(self.var_cwt_dj.get().strip()),
            peak_band_percent=float(self.var_peak_band_percent.get().strip()),
            dpi=int(float(self.var_dpi.get().strip())),
            width_px=int(float(self.var_width_px.get().strip())),
            height_px=int(float(self.var_height_px.get().strip())),
            cmap=self.var_cmap.get().strip(),
            transparent_png=bool(self.var_transparent.get()),
            robust_color=bool(self.var_robust_color.get()),
        )

        if cfg["n_period"] < 20:
            raise ValueError("n_period should be >= 20")
        if cfg["cwt_dj"] <= 0:
            raise ValueError("cwt_dj must be > 0")
        if cfg["peak_band_percent"] < 0:
            raise ValueError("peak_band_percent must be >= 0")
        if cfg["dpi"] <= 0:
            raise ValueError("dpi must be > 0")
        if cfg["width_px"] <= 0 or cfg["height_px"] <= 0:
            raise ValueError("width_px and height_px must be > 0")

        _ = pywt.ContinuousWavelet(cfg["cwt_wavelet"])
        if cfg["period_scale"] not in ("log", "linear"):
            raise ValueError("period_scale must be 'log' or 'linear'")
        return cfg

    def _paths(self):
        if self.selection["mode"] == "files":
            return self.selection.get("files", [])
        folder = self.selection.get("folder", "")
        if not folder:
            return []
        paths = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.lower().endswith(".csv")]
        paths.sort()
        return paths

    def run_batch(self):
        try:
            cfg = self._cfg()
        except Exception as e:
            messagebox.showerror("Parameter error", str(e))
            return

        paths = self._paths()
        if not paths:
            messagebox.showwarning("No input", "Please select CSV files or a folder first.")
            return

        out_dir = self.var_out_dir.get().strip()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self.btn_run.config(state="disabled")
        self._log("\n=== Start batch ===\n")
        self._log(f"Files: {len(paths)}\n")
        self._log(f"Config: {cfg}\n")
        self._log(f"Output dir: {out_dir if out_dir else '(next to each CSV)'}\n")

        ok, fail = 0, 0
        summary_rows = []

        for p in paths:
            try:
                out_csv_peak, out_png_band_log, out_png_band_lin, summary_row = run_one_file(p, cfg, out_dir)
                ok += 1
                summary_rows.append(summary_row)
                self._log(
                    f"[OK] {os.path.basename(p)}\n"
                    f"     -> {out_csv_peak}\n"
                    f"     -> {out_png_band_log}\n"
                    f"     -> {out_png_band_lin}\n"
                )
            except Exception as e:
                fail += 1
                self._log(f"[FAIL] {os.path.basename(p)} : {e}\n")

        summary_path = None
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)

            preferred_cols = [
                "file_name", "file_path", "output_dir",
                "global_peak_period", "global_peak_freq", "global_peak_band_mean_power",
                "peak_band_percent", "band_period_min", "band_period_max", "cropped_n_period_bins",
                "n_time_points", "dt",
                "period_min_used", "period_max_used", "n_period", "period_scale",
                "cwt_wavelet", "cwt_dj",
                "detrend", "zscore", "resample_uniform",
                "png_log_path", "png_linear_path", "per_file_csv_path",
            ]
            existing_cols = [c for c in preferred_cols if c in summary_df.columns]
            other_cols = [c for c in summary_df.columns if c not in existing_cols]
            summary_df = summary_df[existing_cols + other_cols]

            summary_base = out_dir if out_dir else (
                self.selection["folder"] if self.selection["mode"] == "folder"
                else os.path.dirname(paths[0])
            )
            summary_path = os.path.join(summary_base, "CWT_global_peak_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            self._log(f"[SUMMARY] -> {summary_path}\n")

        self._log(f"\n=== Done. OK={ok}, FAIL={fail} ===\n")
        self.btn_run.config(state="normal")

        if summary_path:
            messagebox.showinfo("Done", f"Finished.\nOK={ok}\nFAIL={fail}\n\nSummary:\n{summary_path}")
        else:
            messagebox.showinfo("Done", f"Finished.\nOK={ok}\nFAIL={fail}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()