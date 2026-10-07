# -*- coding: utf-8 -*-
"""
第一步：语料获取 (Corpus Acquisition)
=====================================
本脚本负责把公开中文评论语料整理成统一的、带来源标注的结构化语料库。

数据来源（均可公开校验）：
  1. online_shopping_10_cats —— 中文电商十品类商品评论，62,774 条，人工情感标注 (1=正向, 0=负向)
     品类：书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店
     ★ 与跨境电商情境最贴近，作为主分析语料
     https://github.com/SophonPlus/ChineseNlpCorpus
  2. ChnSentiCorp_htl_all   —— 中文酒店评论，7,766 条，人工情感标注
  3. waimai_10k             —— 中文外卖评论，11,987 条，人工情感标注
     （2、3 用于跨情境稳健性检验）

输出：data/processed/corpus_all.csv
字段：doc_id, source, category, label, text, n_char
"""
import os
import csv
import zipfile
import io
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)


def load_htl():
    """ChnSentiCorp 酒店评论"""
    path = os.path.join(RAW, "ChnSentiCorp_htl_all.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text"})
    df["source"] = "ChnSentiCorp_htl"
    df["category"] = "酒店"
    return df[["source", "category", "label", "text"]]


def load_waimai():
    """waimai_10k 外卖评论"""
    path = os.path.join(RAW, "waimai_10k.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text"})
    df["source"] = "waimai_10k"
    df["category"] = "外卖"
    return df[["source", "category", "label", "text"]]


def load_shop():
    """online_shopping_10_cats 电商十品类评论（主分析语料）"""
    csv_path = os.path.join(RAW, "online_shopping_10_cats.csv")
    if not os.path.exists(csv_path):
        # 尝试从 zip 解压
        zip_path = os.path.join(RAW, "online_shopping_10_cats.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
                with open(csv_path, "wb") as f:
                    f.write(z.read(name))
        else:
            return None
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"review": "text", "cat": "category"})
    df["source"] = "online_shopping_10_cats"
    return df[["source", "category", "label", "text"]]


def main():
    frames = []
    for name, fn in [("online_shopping_10_cats", load_shop),
                     ("ChnSentiCorp_htl", load_htl),
                     ("waimai_10k", load_waimai)]:
        try:
            d = fn()
            if d is None:
                print(f"[SKIP] {name}: 文件不存在")
                continue
            print(f"[OK] {name}: {len(d)} 条")
            frames.append(d)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    df = pd.concat(frames, ignore_index=True)

    # ---- 清洗：去空、去重、去极短文本 ----
    n0 = len(df)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 4]          # 过滤极短噪声（如“好”）
    n1 = len(df)
    df = df.drop_duplicates(subset=["source", "text"])
    n2 = len(df)

    df["n_char"] = df["text"].str.len()
    df = df.reset_index(drop=True)
    df.insert(0, "doc_id", ["D%06d" % (i + 1) for i in range(len(df))])

    out = os.path.join(PROC, "corpus_all.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n===== 语料库构建完成 =====")
    print(f"原始合并: {n0}  去极短后: {n1}  去重后: {n2}")
    print(f"输出: {out}")
    print("\n按来源分布:")
    print(df.groupby("source").size())
    print("\n按来源×标签分布:")
    print(df.groupby(["source", "label"]).size())
    if "online_shopping_10_cats" in set(df["source"]):
        print("\n电商语料按品类×标签分布:")
        sh = df[df["source"] == "online_shopping_10_cats"]
        print(sh.groupby(["category", "label"]).size().unstack().to_string())
    print("\n文本长度描述:")
    print(df["n_char"].describe())


if __name__ == "__main__":
    main()
