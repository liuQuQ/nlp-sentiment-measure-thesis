# -*- coding: utf-8 -*-
"""
第二步：文本预处理 (Preprocessing)
==================================
五步流水线：文档解析 → 文本清洗 → 中文分词 → 去停用词 → 词形归一

在管理研究中，评论文本不存在 PDF 解析问题，但存在同构问题：
  - HTML/表情/URL 残留需要清洗
  - 中文分词需选模式
  - 需要"通用 + 领域 + 自定义"三维度停用词表（课件明确要求）
"""
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")

# ============ ① 通用停用词表 ============
GENERAL_STOPWORDS = set("""
的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 她们 它们 自己 这 那 这个 那个 这些 那些
之 于 也 在 有 我 你 他 她 它 们 被 把 让 给 对 从 向 到 等 是的 不是 还是 就是 但是 但 因为 所以
如果 虽然 然而 然后 而且 并 并且 却 只是 已经 可以 不能 能够 应该 会 要 想 说 看 做 用 去 来 上 下
里 中 外 前 后 左 右 很 非常 太 更 最 比较 有点 一点 一些 什么 怎么 为什么 哪 哪个 谁 时候 现在
还 又 再 一 二 三 四 五 六 七 八 九 十 个 只 条 件 次 位 名 家 张 台 部 本 篇 页 元 块 毛 分 秒 小时
天 年 月 日 号 点 分 秒 一般 一样 一直 一下 一定 一切 方面 问题 情况 地方 东西 事情 时候 感觉
知道 觉得 认为 希望 觉得 表示 进行 作为 通过 根据 由于 关于 对于 除了 以及 或者 不过 其实
""".split())

# ============ ② 领域停用词表（电商/评论场景）============
# 课件明确提醒："停用词表照搬通用表"是常见错误，财经/领域词需补充
DOMAIN_STOPWORDS = set("""
酒店 宾馆 房间 住 入住 前台 客房 服务员 房间号 大床房 标间 房型 酒店名称
外卖 配送 骑手 送餐 商家 店铺 店家 客服 快递 物流 发货 收货 下单 订单
商品 产品 东西 买的 购买 收到 用了 使用 试了 感觉 总体 整体 总的来说
不错 一般 还行 可以 满意 推荐 好评 差评 好评如潮 五星 评价 评论
""".split())

# ============ ③ 自定义停用词表（含噪声）============
CUSTOM_STOPWORDS = set("""
图片 图 晒图 追加 追评 匿名 用户 会员 积分 优惠券 活动 秒杀 包邮 满减
呵呵 哈哈 哈哈哈 嘻嘻 emmm 哎 唉 嗯 哦 噢 啊 呀 吧 呢 吗 啦 咯 喔
""".split())

STOPWORDS = GENERAL_STOPWORDS | DOMAIN_STOPWORDS | CUSTOM_STOPWORDS

# ============ 保留词白名单：情感词即使命中也保留 ============
# 防止"满意""推荐"等领域停用词误伤情感信号
KEEP_WHITELIST = {"满意", "推荐", "不错", "可以", "一般", "还行", "失望", "差", "好"}


def clean_text(text: str) -> str:
    """文本清洗：去 URL、表情、重复符号、不可见字符"""
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)          # URL
    t = re.sub(r"<[^>]+>", " ", t)                        # HTML 标签
    t = re.sub(r"\[[^\]]{0,10}\]", " ", t)                # [表情] 占位
    t = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9%\.。，,！？!?；;：:]", " ", t)  # 保留中文/字母/数字/常用标点
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)                  # 折叠重复字符 “好好好好” -> “好好”
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(text: str, mode: str = "precise"):
    """中文分词：jieba 精确模式（课件推荐，避免搜索引擎模式切碎专有名词）"""
    import jieba
    if mode == "precise":
        words = jieba.lcut(text, cut_all=False)
    elif mode == "full":
        words = jieba.lcut(text, cut_all=True)
    else:
        words = jieba.lcut_for_search(text)
    return words


def normalize(words):
    """词形归一：统一大小写、去单字数字、去纯标点"""
    out = []
    for w in words:
        w = w.strip().lower()
        if len(w) < 2 and not w.isdigit():   # 保留有意义的单字需谨慎，此处去掉
            continue
        if re.fullmatch(r"[0-9\.%。，,！？!?；;：:]+", w):
            continue
        out.append(w)
    return out


def remove_stopwords(words):
    """去停用词，白名单保护情感词"""
    return [w for w in words if (w not in STOPWORDS) or (w in KEEP_WHITELIST)]


def preprocess(text: str):
    """完整五步流水线，返回 (清洗后文本, 分词列表, 去停用词后列表)"""
    cleaned = clean_text(text)
    words = tokenize(cleaned, "precise")
    words = normalize(words)
    tokens = remove_stopwords(words)
    return cleaned, words, tokens


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(os.path.join(PROC, "corpus_all.csv"))
    print("预处理示例：")
    for i in range(3):
        raw = df.loc[i, "text"]
        c, w, t = preprocess(raw)
        print(f"\n【原始】{raw[:80]}")
        print(f"【清洗】{c[:80]}")
        print(f"【分词】{'/'.join(w[:25])}")
        print(f"【去停】{'/'.join(t[:25])}")
    print(f"\n停用词表规模: 通用{len(GENERAL_STOPWORDS)} + 领域{len(DOMAIN_STOPWORDS)} "
          f"+ 自定义{len(CUSTOM_STOPWORDS)} = 合计{len(STOPWORDS)} 个")
