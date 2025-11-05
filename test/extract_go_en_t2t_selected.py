#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
texts/go/t2t_en.jsonl を基に、
「お題（question）ごとの star の最大値 >= 10」のお題のみを選定し、
該当するお題に属するレコードだけを JSONL として出力します。
（＝お題単位でフィルタリングし、元レコードの内容はそのまま保ちます。）

使い方:
  python go_make_selected_from_first_vote.py \
    --input texts/go/t2t_en.jsonl \
    --output texts/go/t2t_en_selected.jsonl \
    --threshold 10

備考:
  - 入力/出力とも .jsonl と .jsonl.gz に対応
  - questionは前後/全角空白を正規化して集計します（出力は元のレコードをそのまま書き出し）
"""

from __future__ import annotations

import argparse
import gzip
import json
from typing import Dict, Iterator, Tuple, Any, TextIO, Union, Set

from collections import defaultdict
from pathlib import Path


# ---------- I/O ユーティリティ ----------

def smart_open(path: str, mode: str) -> Union[TextIO, gzip.GzipFile]:
    """拡張子が .gz の場合は gzip.open、それ以外は通常の open を使う。"""
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
    if q is None:
        return None
    q = str(q).replace("\u3000", " ")
    q = q.strip()
    return q if q != "" else None


def to_number(x: Any) -> float | None:
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
        try:
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return None


def compute_max_star_per_question(input_path: str) -> Dict[str, float]:
    """各 question の max(star) を返す。"""
    max_star: Dict[str, float] = {}
    for rec in iter_jsonl(input_path):
        q = normalize_question(rec.get("question"))
        if q is None:
            continue
        star = to_number(rec.get("star"))
        if star is None:
            continue
        if (q not in max_star) or (star > max_star[q]):
            max_star[q] = star
    return max_star


def filter_by_questions(input_path: str, output_path: str, keep_questions: Set[str]) -> int:
    """keep_questions に含まれる question のレコードのみ出力。戻り値は書き出し件数。"""
    n_write = 0
    with smart_open(output_path, "wt") as out_f:
        for rec in iter_jsonl(input_path):
            q_raw = rec.get("question")
            q_norm = normalize_question(q_raw)
            if q_norm is None:
                continue
            if q_norm in keep_questions:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_write += 1
    return n_write


# ---------- メイン ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="texts/go/t2t_en.jsonl",
                    help="入力ファイル（.jsonl / .jsonl.gz）")
    ap.add_argument("--output", "-o", default="texts/go/t2t_en_selected.jsonl",
                    help="出力ファイル（.jsonl / .jsonl.gz）")
    ap.add_argument("--threshold", "-t", type=float, default=10.0,
                    help="お題の max(star) の下限 (既定: 10)")
    args = ap.parse_args()

    # 1) questionごとの max(star) を計算
    max_star = compute_max_star_per_question(args.input)
    if not max_star:
        print("入力に有効データが見つかりませんでした。")
        return

    # 2) しきい値で選定
    keep_questions = {q for q, m in max_star.items() if m is not None and m >= args.threshold}
    print(f"Total questions: {len(max_star)}  ->  Selected: {len(keep_questions)} (threshold={args.threshold:g})")

    # 3) 選定されたお題に属するレコードだけを書き出し
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_out = filter_by_questions(args.input, args.output, keep_questions)
    print(f"Wrote {n_out} records to: {args.output}")


if __name__ == "__main__":
    main()
