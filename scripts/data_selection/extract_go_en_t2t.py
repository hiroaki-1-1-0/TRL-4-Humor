#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
en.jsonl を2パスで走査して、以下を満たすレコードのみ data/texts/go/t2t_en.jsonl に抽出します。
  - type が "T2T"
  - 同一 question を持つレコードが複数存在（>= 2）
  - question も text も null ではない（空文字は許容。要件が null のみだったため）

使い方:
  python make_go/t2t_en.py \
    --input data/texts/go/en.jsonl \
    --output data/texts/go/t2t_en.jsonl

gz圧縮(.jsonl.gz)にも自動対応します。
"""

import argparse
import json
import gzip
from collections import Counter
from json import JSONDecodeError
from typing import Iterator, Dict, Any, TextIO, Union
import os


def smart_open(path: str, mode: str) -> Union[TextIO, gzip.GzipFile]:
    """
    拡張子が .gz の場合は gzip.open、それ以外は通常の open を使う。
    mode はテキストモードを想定（'r', 'w' など）。encoding は UTF-8 固定。
    """
    if path.endswith(".gz"):
        # gzip はバイナリモード + encoding 指定のため 'rt'/'wt' を用いる
        if "b" in mode:
            mode = mode.replace("b", "")
        tmode = mode if "t" in mode else (mode + "t")
        return gzip.open(path, tmode, encoding="utf-8")
    else:
        return open(path, mode, encoding="utf-8", newline="\n")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """
    JSON Lines を1行ずつ dict で返すイテレータ。空行やパースエラー行はスキップ。
    """
    with smart_open(path, "rt") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except JSONDecodeError:
                # 壊れた行はスキップ（必要ならログ出力に変更）
                continue
            if isinstance(obj, dict):
                yield obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/texts/go/en.jsonl", help="入力 en.jsonl（または .jsonl.gz）")
    parser.add_argument("--output", "-o", default="data/texts/go/t2t_en.jsonl",
                        help="出力ファイル名（既定: data/texts/go/t2t_en.jsonl。拡張子 .gz なら gzip 出力）")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    # 1パス目: T2T かつ question が None でないものをカウント
    q_counter = Counter()
    total_lines = 0
    t2t_nonnull_q = 0

    for rec in iter_jsonl(in_path):
        total_lines += 1
        if rec.get("type") == "T2T":
            q = rec.get("question")
            if q is not None:
                q_counter[q] += 1
                t2t_nonnull_q += 1

    # 2パス目: 条件を満たすものを書き出し
    written = 0
    t2t_nonnull_q_text = 0

    with smart_open(out_path, "wt") as out_f:
        for rec in iter_jsonl(in_path):
            if rec.get("type") != "T2T":
                continue
            q = rec.get("question")
            t = rec.get("text")

            # question と text がどちらも None ではない
            if q is None or t is None:
                continue
            t2t_nonnull_q_text += 1

            # 同一 question が2件以上存在
            if q_counter.get(q, 0) >= 2:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    # 簡易サマリ出力
    # （必要なければ print を削ってもOK）
    print("----- Summary -----")
    print(f"Input file         : {in_path}")
    print(f"Output file        : {out_path} ({'gz' if out_path.endswith('.gz') else 'plain'})")
    print(f"Total lines read   : {total_lines}")
    print(f"T2T & question!=null (pass1): {t2t_nonnull_q}")
    print(f"T2T & question/text!=null (pass2 candidates): {t2t_nonnull_q_text}")
    print(f"Written records    : {written}")


if __name__ == "__main__":
    main()