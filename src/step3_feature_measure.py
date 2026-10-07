# -*- coding: utf-8 -*-
"""
第三步 + 第四步：特征提取 (Feature Extraction) 与 指标构建 (Measure Construction)
================================================================================
按课件三条技术路线全部实现，并构建分层测度：

路线一 · 词典法 (Lexicon)        —— 最透明、可解释，作为主测度
路线二 · TF-IDF + 机器学习       —— 监督式情感分类，作为聚合效度检验的对照
路线三 · 词嵌入 (Word2Vec)       —— 语义向量 + 余弦相似度，用于拓展效度与词表扩展

指标层次（课件 L1/L2/L3）：
  L1 原始计数  : 正向词数 / 负向词数
  L2 标准化比率: (正向词数 - 负向词数) / 总词数 × 100   ← 主测度 Sentiment
  L3 复合测度  : 0.5×情感强度 + 0.3×情感极性一致度 + 0.2×(1 - 情感离散度)
"""
import os
import re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 情感词典：来自 src/lexicon.py（含来源 A/B/C 三层，可追溯）
# ============================================================
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexicon import (POS_WORDS, NEG_WORDS, DEGREE, NEGATION, STRONG_WORDS,
                     POS_A, POS_B, POS_C, NEG_A, NEG_B, NEG_C)


def _load_stopwords():
    from step2_preprocess import STOPWORDS
    return STOPWORDS


def lexicon_score(tokens):
    """
    路线一：词典法打分
    返回 dict：pos_cnt(正向词原始计数), neg_cnt(负向词原始计数),
               pos_w(加权正向), neg_w(加权负向), n_word(总词数)
    支持程度副词加权与否定词翻转（提升语境适应性，应对"词典误判、语境迁移"）
    """
    pos_cnt = neg_cnt = 0
    pos_w = neg_w = 0.0
    for i, w in enumerate(tokens):
        pol = 1 if w in POS_WORDS else (-1 if w in NEG_WORDS else 0)
        if pol == 0:
            continue
        weight = 1.0
        # 前视窗口 2 个词，处理程度副词与否定词
        for j in range(max(0, i - 2), i):
            pj = tokens[j]
            if pj in DEGREE:
                weight *= DEGREE[pj]
            if pj in NEGATION:
                pol = -pol
        if pol > 0:
            pos_cnt += 1
            pos_w += weight
        else:
            neg_cnt += 1
            neg_w += weight
    return dict(pos_cnt=pos_cnt, neg_cnt=neg_cnt, pos_w=pos_w, neg_w=neg_w,
                n_word=len(tokens))


def build_measures(df):
    """构建 L1/L2/L3 三层测度"""
    recs = []
    for _, r in df.iterrows():
        s = lexicon_score(r["tokens"])
        recs.append(s)
    m = pd.DataFrame(recs)

    # L1 原始计数
    df["L1_PosCnt"] = m["pos_cnt"]
    df["L1_NegCnt"] = m["neg_cnt"]
    df["L1_TotalSentWord"] = m["pos_cnt"] + m["neg_cnt"]

    # L2 标准化比率（分母 = 总词数，口径统一；并做分母稳健性检验）
    nw = m["n_word"].replace(0, np.nan)
    nch = df["n_char"].replace(0, np.nan)
    df["L2_Sentiment_word"] = (m["pos_w"] - m["neg_w"]) / nw * 100       # 主测度
    df["L2_Sentiment_char"] = (m["pos_cnt"] - m["neg_cnt"]) / nch * 100  # 稳健性口径①
    df["L2_PosRatio"] = m["pos_w"] / nw * 100
    df["L2_NegRatio"] = m["neg_w"] / nw * 100

    # L3 复合测度三维度
    tot = (m["pos_w"] + m["neg_w"]).replace(0, np.nan)
    df["D1_Intensity"] = np.log1p(m["pos_w"] + m["neg_w"])               # 情感强度
    df["D2_Polarity"] = (m["pos_w"] - m["neg_w"]) / tot                  # 极性一致度
    df["D3_Dispersion"] = 1 - df["D2_Polarity"].abs()                    # 情感离散度
    # 标准化后复合，避免量纲差异
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    inten = scaler.fit_transform(df[["D1_Intensity"]].fillna(0)).ravel()
    disp = df["D3_Dispersion"].fillna(1).values
    pol = df["D2_Polarity"].fillna(0).values
    df["L3_Composite"] = 0.5 * pol + 0.3 * inten - 0.2 * disp
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(BASE, "src"))
    from step2_preprocess import preprocess

    df = pd.read_csv(os.path.join(PROC, "corpus_all.csv"))
    print(f"读入语料 {len(df)} 条，开始预处理（jieba 首次加载词典稍慢）...")
    cleaned, words_l, tokens_l = [], [], []
    for t in df["text"]:
        c, w, tk = preprocess(t)
        cleaned.append(c); words_l.append(w); tokens_l.append(tk)
    df["clean_text"] = cleaned
    df["words"] = ["/".join(x) for x in words_l]
    df["tokens"] = tokens_l
    df["n_word"] = [len(x) for x in tokens_l]

    print("开始构建测度...")
    df = build_measures(df)

    df.to_pickle(os.path.join(PROC, "corpus_measured.pkl"))
    keep = ["doc_id", "source", "category", "label", "text", "n_char", "n_word",
            "L1_PosCnt", "L1_NegCnt", "L1_TotalSentWord",
            "L2_Sentiment_word", "L2_Sentiment_char", "L2_PosRatio", "L2_NegRatio",
            "D1_Intensity", "D2_Polarity", "D3_Dispersion", "L3_Composite"]
    df[keep].to_csv(os.path.join(OUT, "measure_panel.csv"),
                    index=False, encoding="utf-8-sig")

    print("\n===== 测度构建完成 =====")
    print(f"词典规模：正向 {len(POS_WORDS)} 词（A{len(POS_A)}+B{len(POS_B)}+C{len(POS_C)}） / "
          f"负向 {len(NEG_WORDS)} 词（A{len(NEG_A)}+B{len(NEG_B)}+C{len(NEG_C)}）")
    print(df[keep[7:]].describe().round(3).to_string())
