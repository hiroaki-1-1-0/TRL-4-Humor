#!/usr/bin/env python3
"""
各CSVファイルの「最初のレコード」の「得票数」を収集してヒストグラムを作成するスクリプト。
さらに、ターミナル上で a を何度でも入力でき、a 未満のレコードを除外したときの
「レコードのカバー率（残存件数 / 全体件数）」を逐次出力します。

使い方:
    python scripts/data_selection/make_chaya_hist_from_first_vote.py \
        --input data/texts/Oogirichaya_raw_data \
        --out-csv scripts/data_selection/fig/first_record_votes_extracted_new.csv \
        --out-png scripts/data_selection/fig/first_record_votes_histogram_new.png

オプション:
    --no-loop : 実行後のインタラクティブ入力を無効にします（既定では有効）。

備考:
    - 各CSVの1行目はヘッダとみなし、その直後(0番目のデータ行)の「得票数」列の値を抽出します。
    - 文字コードは UTF-8 / UTF-8-SIG / CP932 の順に試行します。
    - 列名は空白や全角スペースを除去して「得票数」と一致するものを優先します。
    - 「お題画像」が NA ではない CSV は集計対象から除外します。
"""
import argparse
import sys
import zipfile
from pathlib import Path
import glob

import pandas as pd
import matplotlib.pyplot as plt


def is_na_like(value) -> bool:
    """NA, NaN, 空文字を NA とみなすための補助関数。"""
    if pd.isna(value):
        return True
    if isinstance(value, str):
        normalized = value.replace("\u3000", " ").strip().lower()
        return normalized in {"", "na", "nan", "none"}
    return False

def find_csvs(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        # unzip to a sibling folder
        unzip_dir = input_path.with_suffix("")
        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(unzip_dir)
        base = unzip_dir
    else:
        base = input_path

    csvs = sorted([Path(p) for p in glob.glob(str(base / "**/*.csv"), recursive=True)])
    return csvs

def read_first_vote_count(path: Path):
    encodings = ["utf-8", "utf-8-sig", "cp932"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, nrows=1)
            df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
            # 「お題画像」が NA でないファイルは対象外
            odai_col = None
            for c in df.columns:
                c_norm = str(c).replace(" ", "")
                if c == "お題画像" or c_norm == "お題画像":
                    odai_col = c
                    break
            if odai_col is not None:
                odai_val = df.iloc[0][odai_col]
                if isinstance(odai_val, str):
                    odai_val = odai_val.replace("\u3000", " ").strip()
                if not is_na_like(odai_val):
                    return None

            # 決め打ち列名・ゆるい一致
            candidates = ["得票数", "票数", "得 票 数", "得 票数", "得票 数"]
            col = None
            for c in df.columns:
                c_norm = str(c).replace(" ", "")
                if c == "得票数" or c_norm == "得票数" or c in candidates:
                    col = c; break
            if col is None:
                for c in df.columns:
                    if "票" in str(c):
                        col = c; break
            if col is None:
                return pd.NA

            val = df.iloc[0][col]
            if isinstance(val, str):
                val = val.replace(",", "").replace("\u3000", " ").strip()
                tran = str.maketrans("０１２３４５６７８９－ー", "0123456789--")
                val = val.translate(tran)
            return pd.to_numeric(val, errors="coerce")
        except Exception:
            continue
    return pd.NA

def build_dataframe(csvs):
    records = []
    for p in csvs:
        v = read_first_vote_count(p)
        if v is None:
            continue
        records.append({"file": str(p), "first_record_得票数": v})
    if not records:
        return pd.DataFrame(columns=["file", "first_record_得票数"])
    df = pd.DataFrame(records).dropna(subset=["first_record_得票数"]).copy()
    df["first_record_得票数"] = df["first_record_得票数"].astype(float)
    return df

def run_interactive_loop(df: pd.DataFrame):
    total = len(df)
    print("\n=== 反復モード ===")
    print("a を入力すると『a 未満のレコードを削除』した際のカバー率を表示します。")
    print("終了するには 'q' または空行で Enter。例: 10, 12.5 など")
    while True:
        try:
            s = input("a = ").strip()
        except EOFError:
            print()  # newline for clean exit
            break
        if s == "" or s.lower() == "q":
            print("終了します。")
            break
        try:
            a = float(s)
        except ValueError:
            print("[ERROR] 数値を入力してください。")
            continue
        remain = (df["first_record_得票数"] >= a).sum()
        coverage = remain / total * 100.0 if total > 0 else 0.0
        print(f"a = {a:g} → カバー率: {coverage:.2f}% （{remain}/{total} レコード）")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/texts/Oogirichaya_raw_data", help="CSVディレクトリ or ZIPファイル（省略時はカレントディレクトリ）")
    ap.add_argument("--out-csv", default="scripts/data_selection/fig/first_record_votes_extracted.csv")
    ap.add_argument("--out-png", default="scripts/data_selection/fig/first_record_votes_histogram.png")
    ap.add_argument("--no-loop", action="store_true", help="インタラクティブな a 入力を行わない")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 入力が存在しません: {input_path}", file=sys.stderr)
        sys.exit(1)

    csvs = find_csvs(input_path)
    if not csvs:
        print("[ERROR] CSVが見つかりませんでした。", file=sys.stderr)
        sys.exit(2)

    df = build_dataframe(csvs)

    # 保存
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

    # ヒストグラム作成 (matplotlib/単一プロット/色指定なし)
    plt.figure(figsize=(8, 5))
    plt.hist(df["first_record_得票数"].values, bins="auto")
    plt.xlabel("Maximum Vote Share")
    plt.ylabel("Frequency")
    plt.title("Maximum Vote Share Histogram")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"Saved: {args.out_png}")

    # 反復モード（既定: 有効）
    if not args.no_loop:
        run_interactive_loop(df)

if __name__ == "__main__":
    main()
