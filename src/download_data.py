# -*- coding: utf-8 -*-
"""
语料自动下载脚本 (Data Downloader)
==================================
从公开中文 NLP 资料库下载本文使用的标注语料。
所有语料均为公开可校验数据，下载后请遵循原始许可条款使用。

用法：
    python src/download_data.py
"""
import os
import sys
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

BASE_URL = ("https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/"
            "master/datasets")

FILES = [
    ("ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
     "ChnSentiCorp_htl_all.csv",
     "中文酒店评论 7,766 条，人工情感标注"),
    ("waimai_10k/waimai_10k.csv",
     "waimai_10k.csv",
     "中文外卖评论 11,987 条，人工情感标注"),
]


def download(url, dest):
    """使用 curl 下载，支持断点续传与重试"""
    for attempt in range(1, 4):
        print(f"  第 {attempt} 次尝试...")
        ret = subprocess.call([
            "curl", "-sL", "--max-time", "300", "--retry", "3",
            "--retry-delay", "2", "-C", "-", "-o", dest, url])
        if ret == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000:
            return True
    return False


def main():
    print("=" * 60)
    print("下载公开中文评论语料")
    print("=" * 60)
    ok = 0
    for rel, name, desc in FILES:
        dest = os.path.join(RAW, name)
        url = f"{BASE_URL}/{rel}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[跳过] {name} 已存在 ({os.path.getsize(dest)} bytes)")
            ok += 1
            continue
        print(f"[下载] {name} —— {desc}")
        print(f"       {url}")
        if download(url, dest):
            print(f"       完成，{os.path.getsize(dest)} bytes")
            ok += 1
        else:
            print(f"       失败：请手动下载 {url} 并放入 data/raw/")
    print("=" * 60)
    print(f"完成 {ok}/{len(FILES)} 个文件")
    if ok < len(FILES):
        sys.exit(1)


if __name__ == "__main__":
    main()
