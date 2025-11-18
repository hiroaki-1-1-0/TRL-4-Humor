#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL を question の値に基づく決定的なハッシュで 9:1 に分割します。

- question の値の SHA256 ハッシュを整数化し、(値 % 10) が 0〜8 なら 9割側、9 なら 1割側へ出力します。
- question が存在しない／JSONとして不正な行はデータ損失を避けるため 9割側へ出力します（統計にカウント）。
- 入力は JSONL（1行=1 JSONオブジェクト）を想定。出力は元行そのまま（余計な整形なし）。

使い方:
    python split_by_question.py \
        --input data/texts/go/t2t_en.jsonl \
        --out9 data/texts/go/t2t_en_selected_9.jsonl \
        --out1 data/texts/go/t2t_en_selected_9.jsonl
"""

import argparse
import hashlib
import json
import sys

def assign_bucket(question: str) -> str:
    # 決定的なハッシュで 10 分割
    h = hashlib.sha256(question.encode("utf-8")).hexdigest()
    v = int(h, 16) % 10
    return "nine" if v < 9 else "one"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="入力 JSONL ファイルパス")
    ap.add_argument("--out9", required=True, help="9割側の出力 JSONL")
    ap.add_argument("--out1", required=True, help="1割側の出力 JSONL")
    args = ap.parse_args()

    total = 0
    to_nine = 0
    to_one = 0
    missing_q = 0
    invalid_json = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out9, "w", encoding="utf-8") as f9, \
         open(args.out1, "w", encoding="utf-8") as f1:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            # 行をそのまま出力したいので、まず JSON 判定だけする
            try:
                obj = json.loads(line)
            except Exception:
                # 不正行は 9割側へ
                invalid_json += 1
                f9.write(line if line.endswith("\n") else (line + "\n"))
                to_nine += 1
                continue

            if "question" not in obj or obj["question"] is None:
                missing_q += 1
                # question 不在は 9割側へ
                f9.write(line if line.endswith("\n") else (line + "\n"))
                to_nine += 1
                continue

            q = str(obj["question"])
            bucket = assign_bucket(q)
            if bucket == "nine":
                f9.write(line if line.endswith("\n") else (line + "\n"))
                to_nine += 1
            else:
                f1.write(line if line.endswith("\n") else (line + "\n"))
                to_one += 1

    # 統計を標準出力へ
    print("Done.")
    print(f"Total lines: {total}")
    print(f"9/10 file lines: {to_nine}")
    print(f"1/10 file lines: {to_one}")
    print(f"Missing 'question': {missing_q}")
    print(f"Invalid JSON lines: {invalid_json}")

if __name__ == "__main__":
    main()
