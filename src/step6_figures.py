# -*- coding: utf-8 -*-
"""
图表生成 (Figures)
==================
生成论文所需图件：流程图、分布图、效度图、回归图
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "output")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# 中文字体
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def fig1_pipeline():
    """图1 NLP测度构建五步流程图"""
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    steps = [("① 语料获取", "Corpus\nAcquisition", "年报/评论/公告\n定源·对齐·留痕", "#4C72B0"),
             ("② 文本预处理", "Pre-\nprocessing", "解析→清洗→分词\n去停用词·归一", "#55A868"),
             ("③ 特征提取", "Feature\nExtraction", "词典法 / 词嵌入\n主题模型", "#C44E52"),
             ("④ 指标构建", "Measure\nConstruction", "L1计数→L2比率\n→L3复合测度", "#8172B2"),
             ("⑤ 信效度检验", "Reliability\n& Validity", "内容/聚合/区分\n/预测效度", "#CCB974")]
    for i, (t, en, d, c) in enumerate(steps):
        x = i * 2.5
        ax.add_patch(plt.Rectangle((x, 0.35), 2.0, 2.0, facecolor=c,
                                   alpha=0.85, edgecolor="white", lw=2))
        ax.text(x + 1.0, 2.05, t, ha="center", va="center", fontsize=13,
                color="white", weight="bold")
        ax.text(x + 1.0, 1.65, en, ha="center", va="center", fontsize=8.5,
                color="white", style="italic")
        ax.text(x + 1.0, 0.95, d, ha="center", va="center", fontsize=8.5, color="white")
        if i < 4:
            ax.annotate("", xy=(x + 2.42, 1.35), xytext=(x + 2.02, 1.35),
                        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#444"))
    ax.annotate("", xy=(0.5, 0.25), xytext=(11.5, 0.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#C44E52",
                                connectionstyle="arc3,rad=0.15", ls="--"))
    ax.text(6.0, 0.02, "不通过则回到第③步重新提取特征", ha="center", fontsize=9,
            color="#C44E52")
    ax.set_xlim(-0.3, 12.2); ax.set_ylim(-0.15, 2.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_pipeline.png"), dpi=160, bbox_inches="tight")
    plt.close()


def fig2_dist(df):
    """图2 情感测度分布与外部标签对照"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    d = df["L2_Sentiment_word"].fillna(0)
    d = d.clip(-60, 60)
    axes[0].hist(d, bins=60, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="red", ls="--", lw=1.5)
    axes[0].set_title("(a) 情感测度 Sentiment 分布", fontsize=11)
    axes[0].set_xlabel("L2_Sentiment_word"); axes[0].set_ylabel("频数")

    for lab, c, name in [(1, "#55A868", "正向评论"), (0, "#C44E52", "负向评论")]:
        sub = df[df["label"] == lab]["L2_Sentiment_word"].fillna(0).clip(-60, 60)
        axes[1].hist(sub, bins=60, alpha=0.6, color=c, label=name, edgecolor="white")
    axes[1].axvline(0, color="black", ls="--", lw=1.5)
    axes[1].set_title("(b) 分情感极性对照", fontsize=11)
    axes[1].legend(); axes[1].set_xlabel("L2_Sentiment_word")

    df["q"] = pd.qcut(df["L2_Sentiment_word"].fillna(0), 5,
                      labels=["Q1\n最低", "Q2", "Q3", "Q4", "Q5\n最高"])
    gm = df.groupby("q", observed=True)["label"].mean()
    axes[2].bar(range(len(gm)), gm.values, color="#8172B2", alpha=0.85)
    axes[2].set_xticks(range(len(gm))); axes[2].set_xticklabels(gm.index, fontsize=9)
    axes[2].set_title("(c) 条件均值检验：五分组正向比例", fontsize=11)
    axes[2].set_ylabel("正向评论比例")
    for i, v in enumerate(gm.values):
        axes[2].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_measure_dist.png"), dpi=160, bbox_inches="tight")
    plt.close()


def fig3_validity(vs):
    """图3 聚合效度散点与相关热图"""
    sub = vs.dropna(subset=["ml_score", "w2v_score"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hexbin(sub["senti"].clip(-60, 60), sub["ml_score"], gridsize=35,
                   cmap="Blues", mincnt=1)
    axes[0].set_xlabel("词典法测度"); axes[0].set_ylabel("TF-IDF+SVM 概率")
    r = np.corrcoef(sub["senti"], sub["ml_score"])[0, 1]
    axes[0].set_title(f"(a) 与机器学习测度 r={r:.3f}", fontsize=11)

    axes[1].hexbin(sub["senti"].clip(-60, 60), sub["w2v_score"], gridsize=35,
                   cmap="Greens", mincnt=1)
    axes[1].set_xlabel("词典法测度"); axes[1].set_ylabel("Word2Vec 情感轴余弦")
    r2 = np.corrcoef(sub["senti"], sub["w2v_score"])[0, 1]
    axes[1].set_title(f"(b) 与词嵌入测度 r={r2:.3f}", fontsize=11)

    cols = ["senti", "ml_score", "w2v_score", "n_word"]
    C = sub[cols].astype(float).corr().values
    im = axes[2].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(cols)))
    axes[2].set_xticklabels(["词典法", "ML", "W2V", "词数"], fontsize=9)
    axes[2].set_yticks(range(len(cols)))
    axes[2].set_yticklabels(["词典法", "ML", "W2V", "词数"], fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            axes[2].text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                         fontsize=8, color="black")
    axes[2].set_title("(c) 测度相关系数矩阵", fontsize=11)
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_validity.png"), dpi=160, bbox_inches="tight")
    plt.close()


def fig4_regression(panel):
    """图4 情感-销量关系与模型系数对比"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hexbin(panel["sentiment"].clip(-15, 25), panel["ln_sales"],
                   gridsize=35, cmap="Purples", mincnt=1)
    z = np.polyfit(panel["sentiment"], panel["ln_sales"], 1)
    xs = np.linspace(panel["sentiment"].min(), panel["sentiment"].max(), 50)
    axes[0].plot(xs, np.polyval(z, xs), "r--", lw=2)
    axes[0].set_xlabel("情感测度 Sentiment"); axes[0].set_ylabel("Ln(销量)")
    axes[0].set_title(f"(a) 情感与销量散点（斜率={z[0]:.4f}）", fontsize=11)

    labs = ["模型1\n混合OLS", "模型2\n+控制变量", "模型3\n双向FE", "模型4\n组间估计"]
    coef = [0.0332, 0.0340, 0.0001, 0.1470]
    err = [0.0081, 0.0070, 0.0013, 0.0235]
    colors = ["#55A868", "#55A868", "#CCCCCC", "#55A868"]
    axes[1].bar(range(4), coef, yerr=[1.96 * e for e in err], capsize=5,
                color=colors, alpha=0.9, edgecolor="black", lw=0.8)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(labs, fontsize=8.5)
    axes[1].set_ylabel("情感测度系数 β")
    axes[1].set_title("(b) 不同模型的系数估计（含95%CI）", fontsize=11)
    for i, (c, e) in enumerate(zip(coef, err)):
        axes[1].text(i, c + 1.96 * e + 0.008, f"{c:.3f}", ha="center", fontsize=8.5)

    # 安慰剂分布
    np.random.seed(1)
    b = panel.copy()
    pl = []
    import statsmodels.formula.api as smf
    for _ in range(200):
        b2 = b.copy()
        b2["sentiment"] = np.random.permutation(b2["sentiment"].values)
        pl.append(smf.ols("ln_sales ~ sentiment + sentiment_var + ln_price + ln_review_vol",
                          data=b2).fit().params["sentiment"])
    axes[2].hist(pl, bins=30, color="#8172B2", alpha=0.8, edgecolor="white")
    axes[2].axvline(0.0340, color="red", ls="--", lw=2, label="真实系数 0.0340")
    axes[2].set_xlabel("安慰剂系数"); axes[2].set_ylabel("频数")
    axes[2].set_title("(c) 安慰剂检验（置换200次）", fontsize=11)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_regression.png"), dpi=160, bbox_inches="tight")
    plt.close()


def fig5_lexicon():
    """图5 词典构成与情感词权重"""
    from lexicon import POS_A, POS_B, POS_C, NEG_A, NEG_B, NEG_C
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cats = ["通用词表\n(来源A)", "语料LLR扩展\n(来源B)", "领域补充\n(来源C)"]
    pv = [len(POS_A), len(POS_B), len(POS_C)]
    nv = [len(NEG_A), len(NEG_B), len(NEG_C)]
    x = np.arange(3); w_ = 0.36
    axes[0].bar(x - w_/2, pv, w_, label="正向词", color="#C44E52", alpha=0.9)
    axes[0].bar(x + w_/2, nv, w_, label="负向词", color="#4C72B0", alpha=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats, fontsize=9)
    axes[0].set_ylabel("词数"); axes[0].legend()
    axes[0].set_title(f"(a) 情感词典三层来源构成（合计{sum(pv)+sum(nv)}词）", fontsize=11)
    for i, (a, b) in enumerate(zip(pv, nv)):
        axes[0].text(i - w_/2, a + 2, str(a), ha="center", fontsize=9)
        axes[0].text(i + w_/2, b + 2, str(b), ha="center", fontsize=9)

    # 高频情感词命中排行
    df = pd.read_pickle(os.path.join(PROC, "corpus_measured.pkl"))
    from lexicon import POS_WORDS, NEG_WORDS
    from collections import Counter
    cp, cn = Counter(), Counter()
    for tk, s in zip(df["tokens"], df["L2_Sentiment_word"].fillna(0)):
        if s > 0:
            cp.update([w for w in tk if w in POS_WORDS])
        elif s < 0:
            cn.update([w for w in tk if w in NEG_WORDS])
    top = cp.most_common(10)[::-1]
    axes[1].barh([w for w, _ in top], [c for _, c in top], color="#C44E52", alpha=0.9)
    axes[1].set_xlabel("出现次数（正向评论中）")
    axes[1].set_title("(b) 正向情感词命中 Top10", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_lexicon.png"), dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(PROC, "corpus_measured.pkl"))
    vs = pd.read_csv(os.path.join(OUT, "validity_sample.csv"))
    panel = pd.read_csv(os.path.join(OUT, "product_panel.csv"))
    fig1_pipeline()
    fig2_dist(df)
    fig3_validity(vs)
    fig4_regression(panel)
    fig5_lexicon()
    print("图件已生成：")
    for f in sorted(os.listdir(FIG)):
        print("  ", f)
