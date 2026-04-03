#!/usr/bin/env python3
# cwt_only_dualY_gui.py
#
# Added:
#   - For each time slice (thin strip parallel to time axis), find the period with maximum CWT power
#   - Output CSV: <stem>_CWT_peak_period.csv (time, peak_period, peak_freq, peak_power)
#   - Optional strip width (time window points) for local averaging to stabilize peak picking
#   - Selectable scalogram colormap from GUI (including cividis)

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
    P_src shape: (n_period_src, n_time_src) with axes period_src and t_src
    Returns P_grid shape: (n_period_grid, n_time_grid)
    2-step interpolation: period then time.
    """
    order = np.argsort(period_src)
    ps = period_src[order]
    Pp = P_src[order, :]

    # period interpolation
    P1 = np.empty((len(period_grid), Pp.shape[1]), dtype=float)
    for j in range(Pp.shape[1]):
        P1[:, j] = np.interp(period_grid, ps, Pp[:, j], left=np.nan, right=np.nan)

    # time interpolation (nan-aware)
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
# Peak ridge extraction (thin strip along time axis)
# -----------------------
def _local_time_mean(P: np.ndarray, j: int, half_w: int) -> np.ndarray:
    """Average power over a thin time strip centered at j (period x time). Returns vector over period."""
    if half_w <= 0:
        return P[:, j]
    a = max(0, j - half_w)
    b = min(P.shape[1], j + half_w + 1)
    return np.nanmean(P[:, a:b], axis=1)


def extract_peak_period_series(
    t_grid: np.ndarray,
    period_grid: np.ndarray,
    power_grid: np.ndarray,
    strip_width_points: int = 1
) -> pd.DataFrame:
    """
    For each time index j:
      take a thin strip (optionally averaged over time window),
      find period index of max power (nan-aware),
      return time, peak_period, peak_freq, peak_power
    """
    if strip_width_points < 1:
        strip_width_points = 1
    # force odd
    if strip_width_points % 2 == 0:
        strip_width_points += 1
    half_w = strip_width_points // 2

    peak_period = np.full(len(t_grid), np.nan, dtype=float)
    peak_power = np.full(len(t_grid), np.nan, dtype=float)

    for j in range(len(t_grid)):
        v = _local_time_mean(power_grid, j, half_w)  # vector over period
        if not np.any(np.isfinite(v)):
            continue
        k = int(np.nanargmax(v))
        peak_period[j] = float(period_grid[k])
        peak_power[j] = float(v[k])

    peak_freq = 1.0 / np.where(np.isfinite(peak_period) & (peak_period > 0), peak_period, np.nan)
    return pd.DataFrame({
        "time": t_grid.astype(float),
        "peak_period": peak_period,
        "peak_freq": peak_freq,
        "peak_power": peak_power
    })


# -----------------------
# Plotting (two independent outputs)
# -----------------------
def _time_edges_from_centers(t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.array([t[0] - 0.5, t[0] + 0.5])
    dtg = float(np.median(np.diff(t)))
    return np.concatenate(([t[0] - dtg/2], (t[:-1] + t[1:]) / 2, [t[-1] + dtg/2]))


def _edges_from_centers(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    dx = float(np.median(np.diff(x)))
    return np.concatenate(([x[0] - dx/2], (x[:-1] + x[1:]) / 2, [x[-1] + dx/2]))


def plot_cwt_logY(out_png: str, t_grid: np.ndarray, period_grid: np.ndarray, cwt_power_grid: np.ndarray,
                  title: str, cmap: str = "viridis", dpi: int = 200):
    fig = plt.figure(figsize=(14, 5.8))
    ax = plt.gca()
    fig.suptitle(title, y=0.98)

    y = np.log2(period_grid)
    t_edges = _time_edges_from_centers(t_grid)
    y_edges = _edges_from_centers(y)

    m = ax.pcolormesh(t_edges, y_edges, cwt_power_grid, shading="auto", cmap=cmap)
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("CWT power (|coef|^2)")

    ax.set_xlabel("Time")
    ax.set_ylabel("log2(Period)")

    yt = np.linspace(y.min(), y.max(), 7)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{2**v:.3g}" for v in yt])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_cwt_linearY(out_png: str, t_grid: np.ndarray, period_grid: np.ndarray, cwt_power_grid: np.ndarray,
                     title: str, cmap: str = "viridis", dpi: int = 200):
    fig = plt.figure(figsize=(14, 5.8))
    ax = plt.gca()
    fig.suptitle(title, y=0.98)

    t_edges = _time_edges_from_centers(t_grid)
    p_edges = _edges_from_centers(period_grid)

    m = ax.pcolormesh(t_edges, p_edges, cwt_power_grid, shading="auto", cmap=cmap)
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("CWT power (|coef|^2)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Period")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# -----------------------
# Pipeline per file
# -----------------------
def run_one_file(csv_path: str, cfg: dict, out_dir: str) -> tuple[str, str, str]:
    t, y = read_two_columns_csv(csv_path)

    # resample
    if cfg["resample_uniform"]:
        t, y, dt = resample_uniform(t, y)
    else:
        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid dt (time column).")

    # preprocess
    y = detrend_signal(t, y, cfg["detrend"])
    if cfg["zscore"]:
        s = np.std(y)
        if s > 0:
            y = (y - np.mean(y)) / s

    # period range (auto if blank)
    T_total = float(t[-1] - t[0])
    nyq_period = 2.0 * dt
    pmin = cfg["period_min"] if cfg["period_min"] is not None else max(nyq_period, T_total / 200.0)
    pmax = cfg["period_max"] if cfg["period_max"] is not None else (T_total / 2.0)
    if not (np.isfinite(pmin) and np.isfinite(pmax) and pmax > pmin > 0):
        raise ValueError(f"Invalid period range: {pmin}..{pmax}")

    # common grids
    period_grid = make_common_period_grid(pmin, pmax, cfg["n_period"], cfg["period_scale"])
    t_grid = t  # uniform grid recommended

    # CWT (raw)
    per_cwt, pow_cwt = compute_cwt_power(y, dt, cfg["cwt_wavelet"], cfg["cwt_dj"], pmin, pmax)

    # Interpolate to common period grid (and time grid)
    cwt_power_grid = interp_time_period_to_grid(t, per_cwt, pow_cwt, t_grid, period_grid)

    # output paths
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    base_out = out_dir if out_dir else os.path.dirname(csv_path)
    os.makedirs(base_out, exist_ok=True)

    out_png_log = os.path.join(base_out, f"{stem}_CWT_logY.png")
    out_png_lin = os.path.join(base_out, f"{stem}_CWT_linearY.png")
    out_csv_peak = os.path.join(base_out, f"{stem}_CWT_peak_period.csv")

    title = (
        f"{stem} | period=[{pmin:.3g}, {pmax:.3g}] dt={dt:.3g} | "
        f"CWT {cfg['cwt_wavelet']}, dj={cfg['cwt_dj']} | grid={cfg['n_period']} ({cfg['period_scale']})"
    )

    plot_cwt_logY(
        out_png_log, t_grid, period_grid, cwt_power_grid,
        title + " | Y=log2(Period)",
        cmap=cfg["cmap"],
        dpi=cfg["dpi"]
    )
    plot_cwt_linearY(
        out_png_lin, t_grid, period_grid, cwt_power_grid,
        title + " | Y=Period(linear)",
        cmap=cfg["cmap"],
        dpi=cfg["dpi"]
    )

    # peak period series
    df_peak = extract_peak_period_series(
        t_grid=t_grid,
        period_grid=period_grid,
        power_grid=cwt_power_grid,
        strip_width_points=cfg["strip_width_points"]
    )
    df_peak.to_csv(out_csv_peak, index=False)

    return out_png_log, out_png_lin, out_csv_peak


# -----------------------
# GUI
# -----------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CWT only (dual Y outputs: log-period & linear-period) + peak period CSV")
        self.geometry("980x780")

        self.selection = {"mode": "files", "files": [], "folder": ""}

        # variables
        self.var_out_dir = tk.StringVar(value="")

        # preprocess
        self.var_resample = tk.BooleanVar(value=True)
        self.var_detrend = tk.StringVar(value="median")
        self.var_zscore = tk.BooleanVar(value=True)

        # period axis
        self.var_pmin = tk.StringVar(value="")
        self.var_pmax = tk.StringVar(value="")
        self.var_nperiod = tk.StringVar(value="160")
        self.var_pscale = tk.StringVar(value="log")

        # CWT
        self.var_cwt_wavelet = tk.StringVar(value="morl")
        self.var_cwt_dj = tk.StringVar(value="0.02")

        # peak extraction
        self.var_stripw = tk.StringVar(value="1")  # points, odd recommended

        # colormap
        self.var_cmap = tk.StringVar(value="cividis")

        # plot
        self.var_dpi = tk.StringVar(value="200")

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

        frm_p = ttk.LabelFrame(self, text="Period axis / interpolation grid (shared for both outputs)")
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
        ttk.Label(frm_c, text="(smaller=denser/slower)").grid(row=0, column=4, padx=6, pady=4, sticky="w")

        frm_peak = ttk.LabelFrame(self, text="Peak period extraction")
        frm_peak.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm_peak, text="Strip width (time points, odd; 1=no averaging)").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Entry(frm_peak, textvariable=self.var_stripw, width=12).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(frm_peak, text="Output: <stem>_CWT_peak_period.csv").grid(row=0, column=2, padx=10, pady=6, sticky="w")

        frm_color = ttk.LabelFrame(self, text="Scalogram color")
        frm_color.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm_color, text="Colormap").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        cb_map = ttk.Combobox(frm_color, textvariable=self.var_cmap, width=18, state="readonly")
        cb_map["values"] = ("cividis", "viridis", "plasma", "inferno", "magma", "turbo", "gray")
        cb_map.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        frm_out = ttk.LabelFrame(self, text="Output")
        frm_out.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_out, text="Output folder (optional)").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Entry(frm_out, textvariable=self.var_out_dir, width=70).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Button(frm_out, text="Browse...", command=self.pick_out_dir).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(frm_out, text="DPI").grid(row=1, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(frm_out, textvariable=self.var_dpi, width=12).grid(row=1, column=1, padx=6, pady=4, sticky="w")

        frm_run = ttk.Frame(self)
        frm_run.pack(fill="both", expand=True, padx=10, pady=8)

        self.btn_run = ttk.Button(frm_run, text="Run batch (CWT only, 2 PNGs + peak CSV each)", command=self.run_batch)
        self.btn_run.pack(anchor="w")

        self.txt = tk.Text(frm_run, height=12, wrap="word")
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
        stripw = int(float(self.var_stripw.get().strip()))
        if stripw < 1:
            stripw = 1
        if stripw % 2 == 0:
            stripw += 1

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
            strip_width_points=stripw,
            cmap=self.var_cmap.get().strip(),
            dpi=int(float(self.var_dpi.get().strip())),
        )
        if cfg["n_period"] < 20:
            raise ValueError("n_period should be >= 20")
        if cfg["cwt_dj"] <= 0:
            raise ValueError("cwt_dj must be > 0")
        _ = pywt.ContinuousWavelet(cfg["cwt_wavelet"])
        if cfg["period_scale"] not in ("log", "linear"):
            raise ValueError("period_scale must be 'log' or 'linear'")
        if not cfg["cmap"]:
            raise ValueError("Please select a colormap.")
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
        for p in paths:
            try:
                out_png_log, out_png_lin, out_csv_peak = run_one_file(p, cfg, out_dir)
                ok += 1
                self._log(
                    f"[OK] {os.path.basename(p)}\n"
                    f"     -> {out_png_log}\n"
                    f"     -> {out_png_lin}\n"
                    f"     -> {out_csv_peak}\n"
                )
            except Exception as e:
                fail += 1
                self._log(f"[FAIL] {os.path.basename(p)} : {e}\n")

        self._log(f"\n=== Done. OK={ok}, FAIL={fail} ===\n")
        self.btn_run.config(state="normal")
        messagebox.showinfo("Done", f"Finished.\nOK={ok}\nFAIL={fail}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()