#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DATA_URL = "https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO/resolve/main/en.jsonl"

def human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit=="TB":
            return f"{n:.1f}{unit}"
        n /= 1024

def download(url, out_path: Path, token: str|None, force: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[skip] {out_path} は既に存在します。--force で上書き可能。")
        return

    headers = {
        "User-Agent": "python-urllib/3",
        "Accept": "*/*",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            total = resp.length or 0
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp, "wb") as f:
                downloaded = 0
                last_print = 0.0
                chunk = 1024 * 64
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    now = time.time()
                    # 進捗は0.2秒に1回だけ表示
                    if now - last_print > 0.2:
                        if total:
                            pct = downloaded * 100 / total
                            sys.stdout.write(f"\rDownloading {out_path.name}: {pct:5.1f}% ({human(downloaded)}/{human(total)})")
                        else:
                            sys.stdout.write(f"\rDownloading {out_path.name}: {human(downloaded)}")
                        sys.stdout.flush()
                        last_print = now
            # 進捗の最終行
            if total:
                print(f"\rDownloading {out_path.name}: 100.0% ({human(total)}/{human(total)})")
            else:
                print(f"\rDownloading {out_path.name}: {human(downloaded)} (done)")
            tmp.replace(out_path)
    except HTTPError as e:
        print(f"[HTTP {e.code}] {e.reason} — {e.url}", file=sys.stderr); sys.exit(1)
    except URLError as e:
        print(f"[URL Error] {e.reason}", file=sys.stderr); sys.exit(1)

def main():
    p = argparse.ArgumentParser(description="Download en.jsonl from Hugging Face without extra deps")
    p.add_argument("--out", default=str(Path("texts") / "go/en.jsonl"), help="保存先（既定: ./data/en.jsonl）")
    p.add_argument("--force", action="store_true", help="既存ファイルを上書き保存")
    p.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="必要ならHFトークンを指定（省略時は環境変数）")
    args = p.parse_args()
    out_path = Path(args.out)
    download(DATA_URL, out_path, args.token, args.force)
    print(f"Saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
