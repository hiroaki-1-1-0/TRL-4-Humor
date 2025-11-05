#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
texts/go/t2t_en.jsonl を読み込み、各「お題（question）」ごとに
回答の star の最大値（max_star）を求めてヒストグラムを描画・保存するスクリプト。
付属の chaya_make_hist_from_first_vote.py と同様に、しきい値 a を対話入力して
「a 以上を満たすお題のカバー率」を繰り返し表示するモードも備えています。

使い方例:
    python go_make_hist_from_first_vote.py \
        --input texts/go/t2t_en.jsonl \
        --out-csv test/go_max_star_per_question.csv \
        --out-png test/go_max_star_histogram.png

オプション:
    --no-loop : 実行後の対話モードを無効にします（既定では有効）。

備考:
    - 入力は .jsonl も .jsonl.gz も可。
    - 各行は少なくとも { "question": <str>, "star": <number> } を含むことを想定します。
    - question / star が欠損(None/NaN)の行は無視します。
    - question の前後空白および全角空白は正規化します（strip + 全角空白→半角空白）。
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple, Any, TextIO, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ---------- I/O ユーティリティ ----------

def smart_open(path: str, mode: str) -> Union[TextIO, gzip.GzipFile]:
    """
    拡張子が .gz の場合は gzip.open、それ以外は通常の open を使う。
    mode はテキストモード（'rt'/'wt'）を前提とし、encoding は UTF-8 固定。
    """
    if path.endswith(".gz"):
        if "b" in mode:
            mode = mode.replace("b", "")
        tmode = mode if "t" in mode else (mode + "t")
        return gzip.open(path, tmode, encoding="utf-8")
    return open(path, mode, encoding="utf-8", newline="\n")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """JSON Lines を1行ずつ dict で返す。壊れた行はスキップ。"""
    with smart_open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


# ---------- 前処理・集計 ----------

def normalize_question(q: Any) -> str | None:
    """question を正規化（前後空白・全角空白の調整）。"""
    if q is None:
        return None
    q = str(q).replace("\u3000", " ")
    q = q.strip()
    return q if q != "" else None


def to_number(x: Any) -> float | None:
    """star を数値へ。全角数字/マイナス等も簡易変換。失敗したら None。"""
    if x is None:
        return None
    s = str(x).replace("\u3000", " ").strip()
    if s == "":
        return None
    tran = str.maketrans("０１２３４５６７８９－ー．，", "0123456789--..")
    s = s.translate(tran)
    try:
        return float(s)
    except Exception:
        # JSONが数値型の場合はそのまま再試行
        try:
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return None


def build_max_star_per_question(input_path: str) -> pd.DataFrame:
    """
    入力JSONLを1パスで走査し、お題ごとに star の最大値を計算して DataFrame を返す。
    返り値の列: ["question", "max_star", "n_records"]
    """
    agg: Dict[str, Tuple[float, int]] = {}  # question -> (max_star, n_records)

    for rec in iter_jsonl(input_path):
        q = normalize_question(rec.get("question"))
        if q is None:
            continue
        star = to_number(rec.get("star"))
        if star is None:
            continue

        if q in agg:
            cur_max, n = agg[q]
            if star > cur_max:
                cur_max = star
            agg[q] = (cur_max, n + 1)
        else:
            agg[q] = (star, 1)

    if not agg:
        return pd.DataFrame(columns=["question", "max_star", "n_records"])

    data = [{"question": q, "max_star": v[0], "n_records": v[1]} for q, v in agg.items()]
    df = pd.DataFrame(data)
    # max_star は float 扱い
    df["max_star"] = pd.to_numeric(df["max_star"], errors="coerce")
    df = df.dropna(subset=["max_star"]).copy()
    return df


# ---------- 対話モード ----------

def run_interactive_loop(df: pd.DataFrame) -> None:
    """
    しきい値 a を入力し、max_star >= a を満たすお題のカバー率を表示するループ。
    """
    total = len(df)
    print("\n=== 反復モード ===")
    print("a を入力すると『max_star >= a』のお題のカバー率を表示します。")
    print("終了するには 'q' または空行で Enter。例: 1, 1.5, 2 など")
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
        remain = (df["max_star"] >= a).sum()
        coverage = (remain / total * 100.0) if total > 0 else 0.0
        print(f"a = {a:g} → カバー率: {coverage:.2f}% （{remain}/{total} お題）")


# ---------- メイン ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="texts/go/t2t_en.jsonl",
                    help="入力 JSONL（.jsonl または .jsonl.gz）")
    ap.add_argument("--out-csv", default="test/fig/go_max_star_per_question.csv",
                    help="集計結果 CSV の出力先")
    ap.add_argument("--out-png", default="test/fig/go_max_star_histogram.png",
                    help="ヒストグラム PNG の出力先")
    ap.add_argument("--no-loop", action="store_true", help="インタラクティブな a 入力を行わない")
    args = ap.parse_args()

    # 集計
    df = build_max_star_per_question(args.input)

    # 保存
    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"Saved: {out_csv_path}  ({len(df)} rows)")

    # ヒストグラム作成（matplotlib / 単一プロット / 色指定なし）
    out_png_path = Path(args.out_png)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    # bins は chaya_* に倣い "auto" にしています
    plt.hist(df["max_star"].values, bins="auto")
    plt.xlim(0, 400)
    plt.ylim(0, 600)
    plt.xlabel("Max Star per Question")
    plt.ylabel("Frequency")
    plt.title("Histogram of Max Star per Question")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    print(f"Saved: {out_png_path}")

    # 反復モード（既定: 有効）
    if not args.no_loop:
        run_interactive_loop(df)


if __name__ == "__main__":
    main()
