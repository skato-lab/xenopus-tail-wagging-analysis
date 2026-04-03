#!/usr/bin/env python3
# fft_periodicity_batch.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from tkinter import Tk, filedialog, messagebox
import os
import traceback

# ====== SNR設定 ======
NOISE_F_MIN = 0.0
NOISE_F_MAX = None
EXCLUDE_BINS_AROUND_PEAK = 3
NOISE_STAT = "median"  # "median" or "mean"
# =====================

# ====== 入出力設定 ======
SAVE_PER_FILE_METRICS = True
SAVE_FFT_DATA_CSV = True
SAVE_AUTOCORR_PNG = True
SAVE_FFT_PNG = True
# ========================


def autocorrelation(sig: np.ndarray) -> np.ndarray:
    r = np.correlate(sig, sig, mode="full")
    r = r[r.size // 2 :]
    mx = np.max(r) if r.size else 0.0
    return r / mx if mx != 0 else r


def safe_read_two_columns_csv(path: str):
    """1列目=t, 2列目=x を想定。ヘッダあり/なし両対応。"""
    try:
        df = pd.read_csv(path, header=0)
        if df.shape[1] < 2:
            raise ValueError("Need >=2 columns")
    except Exception:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 2:
            raise ValueError("CSV must have >=2 columns.")

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    x = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]

    if len(t) < 5:
        raise ValueError("データ点が少なすぎます。")

    # tが降順なら並べ替え
    if np.any(np.diff(t) < 0):
        idx = np.argsort(t)
        t, x = t[idx], x[idx]

    return t, x


def estimate_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("時間間隔が計算できません。")
    T = float(np.mean(dt))
    if T <= 0 or not np.isfinite(T):
        raise ValueError("時間間隔 T が不正です。t列を確認してください。")
    return T


def compute_fft_metrics(t: np.ndarray, x: np.ndarray):
    """主ピークとSNRを計算して dict を返す。"""
    T = estimate_dt(t)

    # detrend (mean remove)
    x_detrended = x - np.mean(x)

    N = len(x_detrended)
    yf = fft(x_detrended)
    xf_full = fftfreq(N, T)
    xf = xf_full[: N // 2]
    amplitude = 2.0 / N * np.abs(yf[: N // 2])

    # peaks (exclude DC)
    peaks, _ = find_peaks(amplitude)
    peaks = peaks[peaks != 0]

    if len(peaks) > 0:
        max_peak_idx = int(peaks[np.argmax(amplitude[peaks])])
        peak_freq = float(xf[max_peak_idx])
        peak_amp = float(amplitude[max_peak_idx])
        period = 1.0 / peak_freq if peak_freq != 0 else np.nan
    else:
        max_peak_idx = None
        peak_freq = np.nan
        peak_amp = np.nan
        period = np.nan

    # SNR
    snr = np.nan
    noise_level = np.nan

    if max_peak_idx is not None and np.isfinite(peak_freq):
        fmin = NOISE_F_MIN
        fmax = NOISE_F_MAX if NOISE_F_MAX is not None else float(np.max(xf))
        band_mask = (xf >= fmin) & (xf <= fmax)
        band_mask &= (xf > 0)  # DC除外

        lo = max(0, max_peak_idx - EXCLUDE_BINS_AROUND_PEAK)
        hi = min(len(amplitude) - 1, max_peak_idx + EXCLUDE_BINS_AROUND_PEAK)
        exclude_mask = np.ones_like(band_mask, dtype=bool)
        exclude_mask[lo : hi + 1] = False

        noise_candidates = amplitude[band_mask & exclude_mask]

        if noise_candidates.size > 5:
            if NOISE_STAT.lower() == "median":
                noise_level = float(np.median(noise_candidates))
            else:
                noise_level = float(np.mean(noise_candidates))

            if noise_level > 0:
                snr = float(peak_amp / noise_level)

    return {
        "dt(time step)": T,
        "PeakFrequency [1/time]": peak_freq,
        "PeakAmplitude": peak_amp,
        "NoiseLevel": noise_level,
        "FFT_SNR": snr,
        "Period [time units]": period,
        "xf": xf,
        "amplitude": amplitude,
        "x_detrended": x_detrended,
    }


def save_autocorr_png(out_png: str, x_detrended: np.ndarray):
    acf = autocorrelation(x_detrended)
    plt.figure()
    plt.plot(acf)
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_fft_png(out_png: str, xf: np.ndarray, amplitude: np.ndarray, peak_freq: float):
    plt.figure()
    plt.plot(xf, amplitude, label="FFT Amplitude")
    if np.isfinite(peak_freq):
        plt.axvline(peak_freq, color="red", linestyle="--", label=f"Peak: {peak_freq:.4f}")
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency [1/time]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def choose_inputs():
    root = Tk()
    root.withdraw()

    # まず複数ファイル選択を試す
    files = filedialog.askopenfilenames(
        title="CSVファイルを複数選択（キャンセルでフォルダ選択へ）",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    files = list(files) if files else []

    if files:
        return files

    # キャンセルされたらフォルダ選択
    folder = filedialog.askdirectory(title="フォルダを選択（フォルダ内のCSVを一括処理）")
    if not folder:
        return []

    paths = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(".csv")
    ]
    paths.sort()
    return paths


def main():
    paths = choose_inputs()
    if not paths:
        print("入力が選択されませんでした。")
        return

    # 出力先：最初のファイルと同じフォルダ
    output_dir = os.path.dirname(paths[0])
    summary_csv = os.path.join(output_dir, "fft_peak_summary.csv")

    results = []
    ok, fail = 0, 0

    print(f"=== Start batch ===")
    print(f"Files: {len(paths)}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_csv}\n")

    for file_path in paths:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            t, x = safe_read_two_columns_csv(file_path)
            met = compute_fft_metrics(t, x)

            # per-file outputs
            if SAVE_AUTOCORR_PNG:
                autocorr_png = os.path.join(output_dir, f"{basename}_autocorrelation.png")
                save_autocorr_png(autocorr_png, met["x_detrended"])

            if SAVE_FFT_DATA_CSV:
                fft_data_csv = os.path.join(output_dir, f"{basename}_fft_data.csv")
                pd.DataFrame({"frequency": met["xf"], "amplitude": met["amplitude"]}).to_csv(
                    fft_data_csv, index=False
                )

            if SAVE_FFT_PNG:
                fft_png = os.path.join(output_dir, f"{basename}_fft_spectrum.png")
                save_fft_png(fft_png, met["xf"], met["amplitude"], met["PeakFrequency [1/time]"])

            row = {
                "File": basename,
                **{k: met[k] for k in [
                    "dt(time step)",
                    "PeakFrequency [1/time]",
                    "PeakAmplitude",
                    "NoiseLevel",
                    "FFT_SNR",
                    "Period [time units]",
                ]},
            }
            results.append(row)

            if SAVE_PER_FILE_METRICS:
                metrics_csv = os.path.join(output_dir, f"{basename}_periodicity_metrics.csv")
                pd.DataFrame([row]).to_csv(metrics_csv, index=False)

            ok += 1
            print(f"[OK]  {basename}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {basename}: {e}")
            # デバッグしたいときは次行を有効化
            # print(traceback.format_exc())

    # ---- summary save (既存があれば同名File行を差し替え) ----
    if results:
        new_df = pd.DataFrame(results)

        if os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 0:
            try:
                old = pd.read_csv(summary_csv)
                if "File" in old.columns:
                    old = old[~old["File"].isin(new_df["File"].astype(str))]
                    out_df = pd.concat([old, new_df], ignore_index=True)
                else:
                    out_df = new_df
            except Exception:
                out_df = new_df
        else:
            out_df = new_df

        out_df.to_csv(summary_csv, index=False)
        print(f"\nSaved summary: {summary_csv}")

    print(f"\n=== Done. OK={ok}, FAIL={fail} ===")


if __name__ == "__main__":
    main()