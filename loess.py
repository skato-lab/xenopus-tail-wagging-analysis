#!/usr/bin/env python3
# loess_gui_batch_with_params_and_3plots.py

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def load_two_columns_csv(path: str, delimiter: str = ",", header_mode: str = "auto"):
    """
    CSV/TSVの1列目=時間, 2列目=値 を読み込み。
    header_mode: auto / yes / no
    """
    if header_mode == "yes":
        df = pd.read_csv(path, sep=delimiter)
        t = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()
        return t, y, df.columns[0], df.columns[1]

    if header_mode == "no":
        df = pd.read_csv(path, sep=delimiter, header=None)
        t = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()
        return t, y, "time", "value"

    # auto
    first = pd.read_csv(path, sep=delimiter, header=None, nrows=1)
    a, b = str(first.iloc[0, 0]), str(first.iloc[0, 1])
    if is_number(a) and is_number(b):
        df = pd.read_csv(path, sep=delimiter, header=None)
        t = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()
        return t, y, "time", "value"
    else:
        df = pd.read_csv(path, sep=delimiter)
        t = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()
        return t, y, df.columns[0], df.columns[1]


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text_summary(path: Path, params: dict) -> None:
    lines = [f"{k}: {v}" for k, v in params.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_xy_plot_png(
    x: np.ndarray,
    y: np.ndarray,
    png_path: Path,
    x_label: str,
    y_label: str,
    title: str | None = None,
):
    """
    x-y グラフをPNG保存する（色指定なし、シンプル）。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def apply_lowess_and_save(
    in_path: str,
    out_csv_path: Path,
    delimiter: str,
    header_mode: str,
    frac: float,
    it: int,
    raw_png_path: Path | None = None,
    trend_png_path: Path | None = None,
    detrend_png_path: Path | None = None,
):
    t, y, t_name, y_name = load_two_columns_csv(in_path, delimiter=delimiter, header_mode=header_mode)

    # NaN除去
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 5:
        raise ValueError("有効データ点が少なすぎます（5点未満）。")

    # 時間ソート（LOWESSはx昇順が安全）
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # LOWESS
    trend = lowess(endog=y, exog=t, frac=frac, it=it, return_sorted=False)
    detrended = y - trend

    out_df = pd.DataFrame(
        {
            t_name: t,
            y_name: y,
            "loess_trend": trend,
            "detrended": detrended,
        }
    )
    out_df.to_csv(out_csv_path, index=False)

    # --- 3種類のPNGを保存（指定されていれば） ---
    title = Path(in_path).name
    if raw_png_path is not None:
        save_xy_plot_png(t, y, raw_png_path, x_label=t_name, y_label=y_name, title=title)
    if trend_png_path is not None:
        save_xy_plot_png(t, trend, trend_png_path, x_label=t_name, y_label="loess_trend", title=title)
    if detrend_png_path is not None:
        save_xy_plot_png(t, detrended, detrend_png_path, x_label=t_name, y_label="detrended", title=title)

    return {
        "n_points_used": int(len(t)),
        "t_column": t_name,
        "y_column": y_name,
    }


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LOESS(LOWESS)（フォルダ一括 + パラメータ + 3種プロットをサブフォルダ保存）")
        self.geometry("960x520")

        self.in_dir = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value="")

        self.delim_mode = tk.StringVar(value="auto")  # auto / comma / tab / custom
        self.custom_delim = tk.StringVar(value=",")

        self.header_mode = tk.StringVar(value="auto")  # auto / yes / no
        self.frac = tk.StringVar(value="0.2")
        self.it = tk.StringVar(value="3")

        # 追加: プロット保存のON/OFF
        self.save_plots = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="入力フォルダ（内部の *.csv *.tsv *.txt を一括処理）").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.in_dir, width=86).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="選択…", command=self.pick_input_dir).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="保存先フォルダ（*_loess.csv + run_params.* + plots_*/）").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.out_dir, width=86).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm, text="選択…", command=self.pick_output_dir).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="区切り文字").grid(row=2, column=0, sticky="w", **pad)
        dbox = ttk.Frame(frm)
        dbox.grid(row=2, column=1, sticky="w", **pad)
        ttk.Radiobutton(dbox, text="自動（拡張子で推定）", value="auto", variable=self.delim_mode,
                        command=self._sync_custom_state).pack(side="left", padx=4)
        ttk.Radiobutton(dbox, text="CSV( , )", value="comma", variable=self.delim_mode,
                        command=self._sync_custom_state).pack(side="left", padx=4)
        ttk.Radiobutton(dbox, text="TSV(\\t)", value="tab", variable=self.delim_mode,
                        command=self._sync_custom_state).pack(side="left", padx=4)
        ttk.Radiobutton(dbox, text="カスタム:", value="custom", variable=self.delim_mode,
                        command=self._sync_custom_state).pack(side="left", padx=4)
        self.custom_entry = ttk.Entry(dbox, textvariable=self.custom_delim, width=6)
        self.custom_entry.pack(side="left", padx=4)

        ttk.Label(frm, text="ヘッダ").grid(row=3, column=0, sticky="w", **pad)
        hbox = ttk.Frame(frm)
        hbox.grid(row=3, column=1, sticky="w", **pad)
        ttk.Radiobutton(hbox, text="auto", value="auto", variable=self.header_mode).pack(side="left", padx=6)
        ttk.Radiobutton(hbox, text="あり", value="yes", variable=self.header_mode).pack(side="left", padx=6)
        ttk.Radiobutton(hbox, text="なし", value="no", variable=self.header_mode).pack(side="left", padx=6)

        ttk.Label(frm, text="LOWESS パラメータ").grid(row=4, column=0, sticky="w", **pad)
        pbox = ttk.Frame(frm)
        pbox.grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(pbox, text="frac (0〜1):").pack(side="left", padx=6)
        ttk.Entry(pbox, textvariable=self.frac, width=10).pack(side="left", padx=4)
        ttk.Label(pbox, text="it (0以上):").pack(side="left", padx=6)
        ttk.Entry(pbox, textvariable=self.it, width=10).pack(side="left", padx=4)

        ttk.Checkbutton(frm, text="元データ / trend / detrend の3種プロット（PNG）をサブフォルダに保存",
                        variable=self.save_plots).grid(row=5, column=1, sticky="w", **pad)

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=1, sticky="e", **pad)
        ttk.Button(btns, text="実行（フォルダ一括）", command=self.run_batch).pack(side="left", padx=6)
        ttk.Button(btns, text="終了", command=self.destroy).pack(side="left", padx=6)

        frm.columnconfigure(1, weight=1)
        self._sync_custom_state()

        hint = (
            "プロット保存ON: 保存先に以下のサブフォルダを作り、PNGを分けて保存します。\n"
            "- plots_raw/   (time–元データ)\n"
            "- plots_trend/ (time–trend)\n"
            "- plots_detrend/ (time–detrended)\n"
            "パラメータ保存: run_params.json / run_params.txt / run_results.json / 各 *_loess.meta.json"
        )
        ttk.Label(frm, text=hint, foreground="#444").grid(row=7, column=0, columnspan=3, sticky="w", padx=8, pady=10)

    def _sync_custom_state(self):
        self.custom_entry.configure(state=("normal" if self.delim_mode.get() == "custom" else "disabled"))

    def pick_input_dir(self):
        path = filedialog.askdirectory(title="入力フォルダを選択")
        if not path:
            return
        self.in_dir.set(path)
        if not self.out_dir.get().strip():
            self.out_dir.set(str(Path(path) / "out_loess"))

    def pick_output_dir(self):
        path = filedialog.askdirectory(title="保存先フォルダを選択")
        if not path:
            return
        self.out_dir.set(path)

    def _get_delimiter_for_file(self, file_path: str) -> str:
        mode = self.delim_mode.get()
        if mode == "comma":
            return ","
        if mode == "tab":
            return "\t"
        if mode == "custom":
            v = self.custom_delim.get()
            if v == "\\t":
                return "\t"
            if v == "":
                raise ValueError("カスタム区切り文字が空です。")
            return v
        ext = os.path.splitext(file_path)[1].lower()
        return "\t" if ext == ".tsv" else ","

    def run_batch(self):
        try:
            in_dir = Path(self.in_dir.get().strip())
            out_dir = Path(self.out_dir.get().strip())
            if not in_dir.as_posix():
                raise ValueError("入力フォルダが未選択です。")
            if not in_dir.exists() or not in_dir.is_dir():
                raise ValueError("入力フォルダが存在しないか、フォルダではありません。")
            if not out_dir.as_posix():
                raise ValueError("保存先フォルダが未選択です。")

            frac = float(self.frac.get())
            if not (0 < frac <= 1.0):
                raise ValueError("frac は 0 より大きく 1 以下にしてください。")
            it = int(float(self.it.get()))
            if it < 0:
                raise ValueError("it は 0 以上にしてください。")

            header_mode = self.header_mode.get()
            delim_mode = self.delim_mode.get()
            custom_delim = self.custom_delim.get()

            out_dir.mkdir(parents=True, exist_ok=True)

            targets = []
            for ext in (".csv", ".tsv", ".txt"):
                targets.extend(sorted(in_dir.glob(f"*{ext}")))
            if not targets:
                raise ValueError("入力フォルダに *.csv *.tsv *.txt が見つかりません。")

            # プロット用サブフォルダ
            save_plots = bool(self.save_plots.get())
            raw_dir = out_dir / "plots_raw"
            trend_dir = out_dir / "plots_trend"
            detrend_dir = out_dir / "plots_detrend"
            if save_plots:
                raw_dir.mkdir(parents=True, exist_ok=True)
                trend_dir.mkdir(parents=True, exist_ok=True)
                detrend_dir.mkdir(parents=True, exist_ok=True)

            run_info = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_dir": str(in_dir),
                "output_dir": str(out_dir),
                "header_mode": header_mode,
                "frac": frac,
                "it": it,
                "delimiter_mode": delim_mode,
                "custom_delimiter_raw": custom_delim,
                "delimiter_auto_rule": "'.tsv' -> TAB, otherwise COMMA",
                "save_3plots_png": save_plots,
                "plot_subdirs": {
                    "raw": "plots_raw",
                    "trend": "plots_trend",
                    "detrend": "plots_detrend",
                },
                "target_extensions": [".csv", ".tsv", ".txt"],
                "n_targets": int(len(targets)),
                "script": Path(__file__).name if "__file__" in globals() else "loess_gui_batch_with_params_and_3plots.py",
            }
            save_json(out_dir / "run_params.json", run_info)
            save_text_summary(out_dir / "run_params.txt", run_info)

            ok = 0
            failed = []
            meta_list = []

            for f in targets:
                try:
                    delimiter = self._get_delimiter_for_file(str(f))

                    out_csv = out_dir / f"{f.stem}_loess.csv"

                    raw_png = (raw_dir / f"{f.stem}_raw.png") if save_plots else None
                    trend_png = (trend_dir / f"{f.stem}_trend.png") if save_plots else None
                    detrend_png = (detrend_dir / f"{f.stem}_detrend.png") if save_plots else None

                    extra = apply_lowess_and_save(
                        in_path=str(f),
                        out_csv_path=out_csv,
                        delimiter=delimiter,
                        header_mode=header_mode,
                        frac=frac,
                        it=it,
                        raw_png_path=raw_png,
                        trend_png_path=trend_png,
                        detrend_png_path=detrend_png,
                    )

                    per_file_meta = {
                        "input_file": str(f),
                        "output_csv": str(out_csv),
                        "plots": {
                            "raw_png": (str(raw_png) if raw_png is not None else None),
                            "trend_png": (str(trend_png) if trend_png is not None else None),
                            "detrend_png": (str(detrend_png) if detrend_png is not None else None),
                        },
                        "delimiter_used": "\\t" if delimiter == "\t" else delimiter,
                        "header_mode": header_mode,
                        "frac": frac,
                        "it": it,
                        **extra,
                    }
                    save_json(out_dir / f"{f.stem}_loess.meta.json", per_file_meta)

                    meta_list.append(per_file_meta)
                    ok += 1

                except Exception as e:
                    failed.append((f.name, str(e)))

            save_json(out_dir / "run_results.json", {"results": meta_list, "failed": failed})

            msg = f"完了: {ok}/{len(targets)} 件を処理しました。\n保存先: {out_dir}\nパラメータ: frac={frac}, it={it}, header={header_mode}"
            msg += "\n\n生成物:\n- *_loess.csv\n- （ON時）plots_raw/, plots_trend/, plots_detrend/ にPNG\n- run_params.json / run_params.txt / run_results.json / 各 *_loess.meta.json"

            if failed:
                msg += "\n\n失敗したファイル（先頭10件）:\n" + "\n".join([f"- {n}: {err}" for n, err in failed[:10]])
                if len(failed) > 10:
                    msg += f"\n...（他 {len(failed)-10} 件）"
                messagebox.showwarning("一部失敗", msg)
            else:
                messagebox.showinfo("完了", msg)

        except Exception as e:
            messagebox.showerror("エラー", str(e))


def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"GUI起動に失敗: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()