#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oogirichaya CSV 選別 → JSONL 変換スクリプト
=================================================
- 入力:   data/texts/Oogirichaya_raw_data （make_chaya_hist_from_first_vote.py を踏襲）
- 出力:   data/texts/chaya/t2t_jp_selected.jsonl.jsonl
- 仕様:
    * 各 CSV の 1 行目はヘッダとみなす。
    * 「お題画像」が NA ではない CSV は抽出対象から除外する（= NA のものだけ採用）。
      ※ NA 判定: pandas の欠損値 / 空文字 / 'NA' 'N/A' 'NaN' 'None' 'Null'（大文字小文字無視）
    * 「得票数」の最大値が 22 未満の CSV は除外する。
    * 採用 CSV からは、
        - 得票数 最大の行（同値多数なら“上から最初の行”）
        - 得票数 最小の行（同値多数なら“下から最後の行”）
      の “2 行のみ” を抽出して JSONL に書き出す。
    * 書き出し時は以下のヘッダ名をリネームする（他はそのまま）:
        - 「お題」   → question
        - 「ボケ」   → text
        - 「得票数」 → star  （列名が「票数」などの近似でも、この列を star にする）

使い方:
    python scripts/make_t2t_jp_selected.py \
        --input data/texts/Oogirichaya_raw_data \
        --output data/texts/chaya/t2t_jp_selected.jsonl \
        --min-votes 22
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path
import glob
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd


# ---------- helpers ----------
def _normalize_colname(name: str) -> str:
    """全角空白→半角、strip。"""
    return str(name).replace("\u3000", " ").strip()


def _remove_spaces(name: str) -> str:
    return _normalize_colname(name).replace(" ", "")


def _find_csvs(input_path: Path) -> List[Path]:
    """make_chaya_hist_from_first_vote.py と同様に、
    入力が ZIP なら隣に展開してから再帰的に CSV を集める。
    """
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        unzip_dir = input_path.with_suffix("")
        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(unzip_dir)
        base = unzip_dir
    else:
        base = input_path
    return sorted([Path(p) for p in glob.glob(str(base / "**/*.csv"), recursive=True)])


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """utf-8 / utf-8-sig / cp932 の順に試す。"""
    encodings = ["utf-8", "utf-8-sig", "cp932"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            # 列名の軽い正規化
            df.columns = [_normalize_colname(c) for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV 読み込み失敗: {path} ({last_err})")


def _detect_vote_col(df: pd.DataFrame) -> Optional[str]:
    """「得票数」列を推定して列名（元の列名）を返す。見つからなければ None。"""
    candidates_exact = ["得票数", "票数", "得 票 数", "得 票数", "得票 数"]
    # 1) 厳密: 空白除去後が「得票数」
    for c in df.columns:
        c_rm = _remove_spaces(c)
        if c_rm == "得票数":
            return c
    # 2) 候補一覧
    for c in df.columns:
        if c in candidates_exact or _remove_spaces(c) in candidates_exact:
            return c
    # 3) ゆるく: 「票」という文字が入る最初の列
    for c in df.columns:
        if "票" in str(c):
            return c
    return None


def _detect_image_col(df: pd.DataFrame) -> Optional[str]:
    """「お題画像」列（あれば）を返す。"""
    for c in df.columns:
        if _remove_spaces(c) == "お題画像":
            return c
    return None


def _to_numeric_votes(series: pd.Series) -> pd.Series:
    """日本語圏フォーマットを考慮して数値化。"""
    def _clean(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).replace(",", "").replace("\\u3000", " ").strip()
        tran = str.maketrans("０１２３４５６７８９－ー", "0123456789--")
        s = s.translate(tran)
        # 空文字は NA
        if s == "":
            return pd.NA
        return s
    cleaned = series.map(_clean)
    return pd.to_numeric(cleaned, errors="coerce")


def _is_na_like(v: Any) -> bool:
    """NA 的な値（欠損/空/'NA' 等）を True とみなす。"""
    if pd.isna(v):
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"", "na", "n/a", "nan", "none", "null"}
    return False


def _select_rows(df: pd.DataFrame, vote_col: str) -> Tuple[pd.Series, pd.Series]:
    """最大票の“上から最初”と、最小票の“下から最後”の 2 行を返す。"""
    votes = _to_numeric_votes(df[vote_col])
    valid = df.loc[~votes.isna()].copy()
    valid_votes = votes.loc[~votes.isna()].astype(float)

    # ここまでで行がなければ例外
    if valid.empty:
        raise ValueError("有効な得票数の行がありません。")

    max_val = float(valid_votes.max())
    min_val = float(valid_votes.min())

    # 最大: idxmax() は先頭を返すので、そのままで OK
    idx_max_first = valid_votes.idxmax()
    row_max = df.loc[idx_max_first]

    # 最小: “下から最後”が必要 → 最小値に等しいインデックスの最後を取る
    idx_min_all = valid_votes[valid_votes == min_val].index
    idx_min_last = idx_min_all[-1]
    row_min = df.loc[idx_min_last]

    # star を明示数値化（書き出しで利用）
    row_max = row_max.copy()
    row_min = row_min.copy()
    row_max[vote_col] = int(max_val) if max_val.is_integer() else max_val
    row_min[vote_col] = int(min_val) if min_val.is_integer() else min_val

    return row_max, row_min


def _rename_for_json(row: pd.Series, vote_col: str) -> Dict[str, Any]:
    """指定列を英名に差し替えた dict を返す。他の列はそのまま。"""
    d: Dict[str, Any] = {}
    for c, v in row.items():
        if c == vote_col:
            key = "star"
        else:
            norm = _remove_spaces(c)
            if norm == "お題":
                key = "question"
            elif norm == "ボケ":
                key = "text"
            else:
                key = c
        d[key] = v
    return d


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/texts/Oogirichaya_raw_data",
                    help="CSV ディレクトリ or ZIP ファイル")
    ap.add_argument("--output", default="data/texts/chaya/t2t_jp_selected.jsonl.jsonl",
                    help="出力 JSONL ファイルパス（ディレクトリが無い場合は作成）")
    ap.add_argument("--min-votes", type=float, default=22.0,
                    help="この値未満しか最大得票がない CSV は除外（既定: 22）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] 入力が存在しません: {in_path}", file=sys.stderr)
        sys.exit(1)

    csvs = _find_csvs(in_path)
    if not csvs:
        print("[ERROR] CSV が見つかりませんでした。", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # chaya フォルダ作成

    total = 0
    used = 0
    skipped_image = 0
    skipped_votes = 0
    skipped_structure = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as fw:
        for p in csvs:
            total += 1
            try:
                df = _read_csv_any_encoding(p)
            except Exception as e:
                skipped_structure += 1
                print(f"[SKIP:READ] {p} ({e})", file=sys.stderr)
                continue

            vote_col = _detect_vote_col(df)
            if vote_col is None:
                skipped_structure += 1
                print(f"[SKIP:NOVOTE] {p} （得票列が見つかりません）", file=sys.stderr)
                continue

            # 「お題画像」による除外（列があれば評価。無ければ通す）
            img_col = _detect_image_col(df)
            if img_col is not None:
                first_val = df.iloc[0][img_col] if len(df) > 0 else pd.NA
                if not _is_na_like(first_val):
                    skipped_image += 1
                    # print(f"[SKIP:IMAGE] {p} （お題画像が NA ではない）", file=sys.stderr)
                    continue

            # 票数の前処理
            votes_num = _to_numeric_votes(df[vote_col])
            if votes_num.dropna().empty:
                skipped_structure += 1
                print(f"[SKIP:NOVALIDVOTES] {p} （得票数が読み取れない）", file=sys.stderr)
                continue

            max_vote = float(votes_num.dropna().max())
            if max_vote < float(args.min_votes):
                skipped_votes += 1
                # print(f"[SKIP:LOWMAX] {p} （最大得票 {max_vote} < {args.min_votes}）", file=sys.stderr)
                continue

            # 行選択
            try:
                row_max, row_min = _select_rows(df, vote_col)
            except Exception as e:
                skipped_structure += 1
                print(f"[SKIP:SELECT] {p} ({e})", file=sys.stderr)
                continue

            # 書き出し（お題/ボケ/得票数を英名にリネーム）
            rec_max = _rename_for_json(row_max, vote_col)
            rec_min = _rename_for_json(row_min, vote_col)

            fw.write(json.dumps(rec_max, ensure_ascii=False) + "\n")
            fw.write(json.dumps(rec_min, ensure_ascii=False) + "\n")
            written += 2
            used += 1

    print("---- Summary ----")
    print(f"  CSV 総数         : {total}")
    print(f"  対象として使用   : {used}")
    print(f"  出力行数         : {written}")
    print(f"  除外（お題画像） : {skipped_image}")
    print(f"  除外（最大得票< {args.min_votes:g}）: {skipped_votes}")
    print(f"  除外（構造/読取など）: {skipped_structure}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
