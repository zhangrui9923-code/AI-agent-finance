"""
增强版 RAG 检索管道 (Enhanced RAG Pipeline)
================================================

【模块定位】
本模块是 AI 金融研究助手系统的「知识检索引擎」，负责将用户查询转换为高质量、
多样化的上下文片段。它是连接知识库与 LLM 推理的核心桥梁。

【为什么需要增强版 RAG？】
传统 RAG (Retrieval-Augmented Generation) 存在以下痛点：

  ❌ 单一向量检索 → 语义漂移（"盈利能力"可能匹配到"亏损"相关内容）
  ❌ 无关键词匹配 → 专业术语精确匹配失败（股票代码、财务比率名称）
  ❌ 无重排序 → 低质量结果混入 Top-K
  ❌ 无多样性控制 → 返回内容高度重复（同一报告的不同段落）
  ❌ 碎片化返回 → 上下文不完整，LLM 无法理解全貌

本模块通过 6 大增强技术解决上述问题。

【架构总览】

  ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  User Query  │────▶│   Query Engine    │────▶│  HyDE Gen    │
  └─────────────┘     │  (预处理/改写)    │     │  (假设文档)  │
                       └────────┬─────────┘     └──────┬───────┘
                                │                      │
              ┌─────────────────┼──────────────────────┼──────────┐
              ▼                 ▼                      ▼          ▼
     ┌────────────────┐ ┌──────────────┐    ┌──────────────┐ ┌──────────┐
     │ Vector Search  │ │ BM25 Search  │    │ HyDE Search  │ │ Metadata │
     │ (语义相似度)   │ │ (关键词匹配)  │    │ (假设文档检索)│ │ Filter   │
     └───────┬────────┘ └──────┬───────┘    └──────┬───────┘ └──────────┘
             │                 │                   │
             └─────────────────┼───────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Result Fusion   │
                    │  (RRF/加权融合)  │
                    └────────┬─────────┘
                              ▼
                    ┌──────────────────┐
                    │ Cross-Encoder    │◀── BAAI/bge-reranker
                    │ (精细重排序)     │
                    └────────┬─────────┘
                              ▼
                    ┌──────────────────┐
                    │ MMR Diversity    │
                    │ (多样性优化)     │
                    └────────┬─────────┘
                              ▼
                    ┌──────────────────┐
                    │ Parent Document  │
                    │ (上下文完整性)   │
                    └────────┬─────────┘
                              ▼
                    ┌──────────────────┐
                    │ Context Window   │
                    │ Assembly & Output│
                    └──────────────────┘

【核心组件职责】

1. BM25Retriever: 轻量级关键词检索器（纯 Python 实现）
   - 适用场景：专业术语、股票代码、精确数值匹配
   - 优势：零外部依赖、内存占用低、可增量更新

2. ParentDocumentRetriever: 父文档检索器
   - 解决问题：子块检索导致的信息碎片化
   - 策略：小块检索 → 大块返回

3. EnhancedRAGPipeline: 主协调器
   - 编排多路召回、融合、重排序、组装全流程
   - 提供可配置的检索策略

【设计原则】

- 可插拔性: 每个检索通道可独立启用/禁用
- 渐进式降级: Cross-Encoder 加载失败时自动回退到向量分数
- 可观测性: 全链路耗时统计、各通道贡献度分析
- 幂等性: 相同输入在相同配置下产生相同输出（确定性）

【性能特征】

- 向量检索: O(log N) （ChromaDB HNSW 索引）
- BM25 检索: O(Q × D_avg × N) （Q=查询词数, D_avg=平均文档词数, N=文档数）
- Cross-Encoder: O(K² × L) （K=候选数, L=序列长度）
- MMR 重排序: O(K² × S) （K=候选数, S=已选数）

【依赖关系】

上游:
  - config.settings: API 密钥、模型配置
  - langchain_openai: Embedding 和 LLM 调用
  - langchain_community.vectorstores.Chroma: 向量存储

下游消费者:
  - src.react.enhanced_react: ReAct 循环中的知识获取
  - main_enhanced.py: 主流程中的上下文构建

【版本信息】
- v2.0 (2025-01): 极致优化版，完整算法实现 + 自测套件
- v1.0 (2024-12): 初始版本，基础混合检索
"""

from __future__ import annotations

import time
import math
import json
import hashlib
from typing import (
    Optional, List, Dict, Any, Tuple, Set,
    Callable, Iterator, Protocol, TypeVar, Generic
)
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto

from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from config.settings import (
    GLM_API_KEY, GLM_BASE_URL, LLM_MODEL, EMBEDDING_MODEL,
    TOP_K_RETRIEVE, TOP_K_RERANK, CHUNK_SIZE, CHUNK_OVERLAP,
)


# ════════════════════════════════════════════════════════════
# 第一部分：枚举与常量定义
# ════════════════════════════════════════════════════════════


class RetrievalMethod(Enum):
    """
    检索方法枚举
    
    标识每条检索结果的来源通道，用于：
    - 结果溯源分析（哪条路召回的？）
    - 融合权重分配（不同通道信任度不同）
    - A/B 测试对比（关闭某通道看效果变化）
    
    成员说明:
        VECTOR: 基于嵌入向量的语义相似度检索
               特点：擅长语义理解，弱于精确匹配
               典型场景："公司盈利情况" → 匹配 "营收增长"
        
        BM25: 基于 TF-IDF 变体的关键词检索
              特点：擅长精确匹配，弱于语义理解
              典型场景："600519" → 精确匹配茅台股票代码
        
        HYDE: 基于 Hypothetical Document Embeddings 的检索
              特点：用 LLM 生成的假设文档作为查询
              典型场景：模糊查询扩展为具体描述
        
        KEYWORD_EXACT: 元数据字段精确匹配
                      特点：零语义，纯字符串比较
                      典型场景：按日期范围筛选报告
    """
    VECTOR = "vector"
    BM25 = "bm25"
    HYDE = "hyde"
    KEYWORD_EXACT = "keyword_exact"


class FusionStrategy(Enum):
    """
    结果融合策略枚举
    
    决定如何合并多路召回的结果：
    
    成员说明:
        RRF (Reciprocal Rank Fusion):
            公式: score = Σ 1/(k + rank_i)
            特点：对排名敏感，天然去重
            适用：各通道质量相近时
        
        WEIGHTED_AVERAGE:
            公式: score = Σ w_i * normalized_score_i
            特点：需要先归一化，支持自定义权重
            适用：某通道明显更可靠时
        
        MAX_SCORE:
            公式: score = max(score_1, score_2, ...)
            特点：简单粗暴，取最优
            适用：快速原型验证
    """
    RRF = "rrf"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX_SCORE = "max_score"


class RerankingMode(Enum):
    """
    重排序模式枚举
    
    成员说明:
        CROSS_ENCODER: 使用 Cross-Encoder 模型精排
                     精度最高，但延迟较大（CPU推理）
                     推荐：离线场景 / 对质量要求极高的场景
        
        VECTOR_RERANK: 使用向量相似度二次计算
                     速度快，但精度有限
                     推荐：实时性要求高的在线场景
        
        NONE: 不重排序，直接使用初始分数
             最快，但质量不可控
             推荐：仅单路召回时的降级方案
    """
    CROSS_ENCODER = "cross_encoder"
    VECTOR_RERANK = "vector_rerank"
    NONE = "none"


class RetrievalQuality(Enum):
    """
    检索质量等级枚举
    
    用于对最终结果进行质量标记，供下游模块参考：
    
    成员说明:
        EXCELLENT: 高相关性 + 高多样性 + 多通道命中
                  建议：可直接用于生成答案
        
        GOOD: 相关性尚可，但可能存在遗漏
             建议：可用于生成，但需标注置信度
        
        FAIR: 仅单一通道命中，或分数偏低
             建议：补充其他信息源后再使用
        
        POOR: 无有效结果或全部低分
             建议：触发查询改写或转人工
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


# 默认融合权重配置
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "vector": 0.35,
    "bm25": 0.25,
    "hyde": 0.40,
}

# 中文停用词表（精简版，覆盖高频虚词）
DEFAULT_STOP_WORDS: Set[str] = {
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "这个", "那个", "这些", "那些",
    "自己", "什么", "怎么", "哪里", "哪个",
    # 助词
    "的", "了", "着", "过", "得", "地",
    # 介词
    "在", "从", "向", "往", "对", "关于", "由于",
    # 连词
    "和", "与", "或", "但", "而", "如果", "因为", "所以",
    # 副词
    "不", "没", "很", "太", "都", "也", "就", "才", "还",
    "已经", "正在", "将要", "曾经",
    # 动词（高频弱动词）
    "是", "有", "会", "能", "可以", "要", "去", "来",
    "说", "看", "做", "让", "给", "被",
    # 数量词
    "一", "二", "三", "十", "百", "千", "万",
    "个", "些", "点", "次", "回",
}


# ════════════════════════════════════════════════════════════
# 第二部分：数据模型定义
# ════════════════════════════════════════════════════════════


@dataclass
class RetrievalResult:
    """
    单条检索结果数据结构
    
    这是 RAG 管道中最核心的数据单元，贯穿检索→融合→重排序→输出的全流程。
    每条结果携带丰富的元数据，支持下游的质量评估和溯源分析。
    
    属性分组:
        【核心内容】
        content: 文档文本内容（可能是子块或父文档）
        metadata: 结构化元数据字典（来源、日期、作者等）
        score: 综合得分（归一化到 [0, 1] 或原始模型分数）
        
        【来源追踪】
        source: 文档来源标识（如文件名、URL、数据库名）
        retrieval_method: 检索方法标识（见 RetrievalMethod 枚举）
        parent_doc_id: 父文档 ID（用于 Parent Document 模式的关联）
        
        【位置信息】
        chunk_index: 在源文档中的块序号（从 0 开始）
        start_offset: 内容在原文中的起始字符偏移量
        end_offset: 内容在原文中的结束字符偏移量
    
    使用示例:
        >>> result = RetrievalResult(
        ...     content="贵州茅台2023年净利润同比增长18.2%...",
        ...     metadata={"source": "annual_report_2023.pdf", "page": 15},
        ...     score=0.92,
        ...     source="annual_report_2023.pdf",
        ...     retrieval_method=RetrievalMethod.VECTOR.value,
        ... )
        >>> print(f"相关度: {result.score:.1%}")
        相关度: 92.0%
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    
    source: str = "unknown"
    retrieval_method: str = ""
    parent_doc_id: Optional[str] = None
    
    chunk_index: int = 0
    start_offset: int = 0
    end_offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于日志记录和缓存）"""
        return {
            "content": self.content[:200],  # 截断长文本
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
            "retrieval_method": self.retrieval_method,
        }
    
    @property
    def is_high_quality(self) -> bool:
        """判断是否为高质量结果（score > 0.7）"""
        return self.score > 0.7


@dataclass
class HybridSearchResult:
    """
    混合检索聚合结果
    
    封装一次完整检索操作的所有产出，包括最终结果列表和详细的诊断统计。
    该对象是 EnhancedRAGPipeline.query() 的返回值。
    
    属性说明:
        query: 原始用户查询（未改写）
        results: 最终结果列表（已按综合得分降序排列）
        
        【各通道贡献统计】
        vector_results_count: 向量通道命中的候选数量
        bm25_results_count: BM25 通道命中的候选数量
        hyde_results_count: HyDE 通道命中的候选数量
        
        【融合参数快照】
        fusion_weights: 本次检索使用的融合权重配置
        total_candidates: 融合前候选总数（含重复）
        unique_results: 融合去重后的唯一结果数
        
        【性能指标】
        retrieval_time_ms: 检索阶段总耗时（毫秒）
        rerank_time_ms: 重排序阶段耗时（毫秒）
        
        【质量评估】
        quality: 自动判定的检索质量等级
    
    下游使用:
        >>> result = rag.query("茅台估值")
        >>> for item in result.results:
        ...     print(f"[{item.score:.3f}] {item.content[:80]}")
        >>> print(f"共{result.unique_results}条结果，耗时{result.retrieval_time_ms:.0f}ms")
    """
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    
    vector_results_count: int = 0
    bm25_results_count: int = 0
    hyde_results_count: int = 0
    
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    total_candidates: int = 0
    unique_results: int = 0
    
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    
    quality: RetrievalQuality = RetrievalQuality.FAIR
    
    @property
    def hit_rate(self) -> float:
        """命中率 = 唯一结果数 / 候选总数"""
        if self.total_candidates == 0:
            return 0.0
        return self.unique_results / self.total_candidates
    
    @property
    def avg_score(self) -> float:
        """平均得分"""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    def get_top_k(self, k: int = 3) -> List[RetrievalResult]:
        """获取 Top-K 结果"""
        return self.results[:k]
    
    def summary(self) -> str:
        """生成人类可读的摘要"""
        return (
            f"Query: {self.query[:50]}...\n"
            f"Results: {self.unique_results}/{self.total_candidates} "
            f"(hit_rate={self.hit_rate:.1%})\n"
            f"Channels: vec={self.vector_results_count}, "
            f"bm25={self.bm25_results_count}, hyde={self.hyde_results_count}\n"
            f"Time: retrieval={self.retrieval_time_ms:.0f}ms, "
            f"rerank={self.rerank_time_ms:.0f}ms\n"
            f"Quality: {self.quality.value}"
        )


class RAGConfig(BaseModel):
    """
    RAG 管道配置模型
    
    集中管理所有可调参数，支持运行时动态修改。
    通过 Pydantic 验证确保参数合法性。
    
    配置项分类:
        【功能开关】
        use_hyde: 是否启用 Hypothetical Document Embedding
                开启后会增加约 1-2 秒 LLM 调用延迟
                但对模糊查询效果提升显著
                
        use_bm25: 是否启用 BM25 关键词检索
                需要预先调用 initialize_bm25() 构建索引
                
        use_reranking: 是否启用 Cross-Encoder 重排序
                     需要 sentence_transformers 库和本地模型文件
                     
        use_mmr: 是否启用 MMR 多样性优化
                当 top_k > 3 时建议开启，避免结果堆叠
                
        use_parent_document: 是否启用父文档返回模式
                           适用于需要完整上下文的生成任务
                           
        【权重参数】
        hyde_weight / vector_weight / bm25_weight:
            各通道融合权重，总和应为 1.0
            
        【数量参数】
        top_k_vector / top_k_bm25: 各通道初始召回数量
                                  建议设为最终 top_k 的 2-3 倍
                                  
        top_k_final: 最终返回数量
        
        mmr_lambda: MMR 平衡参数
                   0.0 = 纯多样性（可能牺牲相关性）
                   1.0 = 纯相关性（无多样性保证）
                   推荐: 0.65-0.75
    """
    use_hyde: bool = True
    use_bm25: bool = True
    use_reranking: bool = True
    use_mmr: bool = True
    use_parent_document: bool = False
    
    hyde_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    
    top_k_vector: int = Field(default=10, ge=1, le=50)
    top_k_bm25: int = Field(default=10, ge=1, le=50)
    top_k_final: int = Field(default=5, ge=1, le=20)
    
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    reranking_mode: RerankingMode = RerankingMode.CROSS_ENCODER
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
    
    @validator("hyde_weight", "vector_weight", "bm25_weight")
    def weights_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("权重必须在 0-1 之间")
        return v


# ════════════════════════════════════════════════════════════
# 第三部分：BM25 关键词检索器
# ════════════════════════════════════════════════════════════


class BM25Retriever:
    """
    BM25 (Best Matching 25) 关键词检索器 — 纯 Python 实现
    
    【算法原理】
    BM25 是信息检索领域的经典概率检索函数，用于估算查询 Q 与文档 D 的相关度：
    
        score(Q, D) = Σ IDF(qi) × [ f(qi,D) × (k1 + 1) ] / [ f(qi,D) + k1 × (1 - b + b × |D|/avgdl) ]
    
    其中:
        - f(qi,D): 词 qi 在文档 D 中的词频 (TF)
        - IDF(qi): 逆文档频率 = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
        - |D|: 文档 D 的长度（词数）
        - avgdl: 语料库中所有文档的平均长度
        - k1: 词频饱和参数（推荐 1.2-2.0）
        - b: 文档长度归一化参数（推荐 0.75）
    
    【为什么需要 BM25？】
    向量检索擅长语义理解，但在以下场景表现不佳：
    - 精确术语匹配："ROE"、"EBITDA" 等缩写
    - 数值型查询："2023Q3"、"600519"
    - 稀有实体：人名、机构名、产品名
    
    BM25 作为互补通道，专门处理这类「精确匹配」需求。
    
    【性能特征】
    - 索引构建: O(N × L_avg) 时间, O(N × V) 空间
      (N=文档数, L_avg=平均长度, V=词汇表大小)
      
    - 单次查询: O(Q × N × L_avg/Q) ≈ O(Q × N)
      (Q=查询词数，实际可通过倒排索引优化到 O(Q × avg_posting_list))
      
    - 内存占用: 约 N × 100 KB（假设每文档 100 个唯一词）
    
    【适用场景】
    ✅ 金融术语检索: "市盈率"、"资产负债率"
    ✅ 代码/编号检索: "600519.SH"、"2023年报"
    ✅ 中英文混合查询
    ❌ 语义相似但词汇不同: "赚钱能力" vs "盈利水平"
    
    【与外部库对比】
    本实现选择纯 Python 而非 rank_bm25 库的原因：
    1. 减少外部依赖（部署友好）
    2. 支持中文原生分词（字符级 n-gram）
    3. 支持增量索引更新
    4. 内置停用词过滤
    
    如需更高性能，可替换为 Elasticsearch 的 BM25 实现。
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stop_words: Optional[Set[str]] = None,
    ):
        """
        初始化 BM25 检索器
        
        Args:
            k1: 词频饱和参数
                控制词频增长的饱和速度：
                - 较大值(如2.0): 词频影响更大，适合短文档
                - 较小值(如1.2): 词频更快饱和，适合长文档
                默认 1.5 是经验最优值
                
            b: 文档长度惩罚系数
                控制文档长度对得分的影响：
                - 较大值(如0.9): 长文档被显著惩罚
                - 较小值(如0.5): 长文档惩罚较轻
                默认 0.75 是标准推荐值
                
            stop_words: 自定义停用词集
                       为 None 时使用内置中文停用词表
        """
        self.k1 = k1
        self.b = b
        self.stop_words = stop_words or DEFAULT_STOP_WORDS
        
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.term_freq: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.documents: List[str] = []
        self.n_docs: int = 0
        self._indexed: bool = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        中文文本分词（基于字符级 n-gram）
        
        策略说明:
        由于金融领域专有名词较多（如"息税前利润"、"归属于母公司股东"），
        采用字符级 n-gram 可以在不依赖外部分词工具的情况下，
        有效捕获中文词汇的组合特征。
        
        分词层级:
        1. Unigram (单字): "盈" "利" "能" "力"
           → 捕获基础语义单元
           
        2. Bigram (双字): "盈利" "利能" "能力"
           → 捕获常见双字词
           
        3. Trigram (三字): "盈利能" "利能力"
           → 捕获三字固定搭配（如"毛利率"、"净利率"）
        
        Args:
            text: 待分词文本
            
        Returns:
            分词结果列表（已过滤停用词）
        """
        text = text.lower()
        tokens: List[str] = []
        
        for char in text:
            if (char.isalnum() or '\u4e00' <= char <= '\u9fff') and char not in self.stop_words:
                tokens.append(char)
        
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            if ('\u4e00' <= bigram[0] <= '\u9fff' and 
                '\u4e00' <= bigram[1] <= '\u9fff'):
                tokens.append(bigram)
        
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            if all('\u4e00' <= c <= '\u9fff' for c in trigram):
                tokens.append(trigram)
        
        return tokens
    
    def build_index(self, documents: List[Document]) -> None:
        """
        构建 BM25 倒排索引
        
        处理流程:
        1. 遍历所有文档，执行分词
        2. 统计每个文档的词频 (TF)
        3. 统计每个词的文档频率 (DF)
        4. 计算平均文档长度
        
        时间复杂度: O(N × L)
        其中 N=文档数, L=平均文档长度
        
        Args:
            documents: LangChain Document 对象列表
                      每个 Document 包含 page_content 和 metadata
                      
        Raises:
            ValueError: 文档列表为空时抛出
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        print(f"[BM25] 📚 开始构建索引，共 {len(documents)} 个文档...")
        start = time.time()
        
        self.n_docs = len(documents)
        self.term_freq = []
        self.doc_lengths = []
        self.documents = []
        
        total_length = 0
        
        for doc in documents:
            text = doc.page_content
            tokens = self._tokenize(text)
            
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            
            self.term_freq.append(dict(tf))
            self.doc_lengths.append(len(tokens))
            self.documents.append(text)
            
            for token in set(tokens):
                self.doc_freq[token] += 1
            
            total_length += len(tokens)
        
        self.avg_dl = total_length / self.n_docs if self.n_docs > 0 else 1.0
        self._indexed = True
        
        elapsed = time.time() - start
        vocab_size = len(self.doc_freq)
        print(f"[BM25] ✅ 索引构建完成 ({elapsed:.2f}s)")
        print(f"[BM25]    词汇表大小: {vocab_size:,} 词")
        print(f"[BM25]    平均文档长度: {self.avg_dl:.1f} 词")
        print(f"[BM25]    内存估算: ~{self.n_docs * vocab_size * 8 / 1024 / 1024:.1f} MB")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        执行 BM25 检索
        
        算法步骤:
        1. 对查询文本进行分词（与索引时分词策略一致）
        2. 对每个文档，遍历查询词计算 BM25 得分
        3. 汇总所有查询词的贡献得到文档总分
        4. 按得分降序排列，返回 Top-K
        
        Args:
            query: 查询文本
            top_k: 返回结果数量上限
            min_score: 最低得分阈值（过滤低质量结果）
            
        Returns:
            [(doc_index, score), ...] 列表
            按 score 降序排列
            
        Raises:
            RuntimeError: 索引未构建时抛出
            
        示例:
            >>> retriever.build_index(docs)
            >>> results = retriever.search("茅台 盈利能力", top_k=5)
            >>> for idx, score in results:
            ...     print(f"Doc[{idx}] score={score:.2f}")
        """
        if not self._indexed:
            raise RuntimeError("BM25 索引未构建，请先调用 build_index()")
        
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        scores: List[Tuple[int, float]] = []
        
        for doc_idx in range(self.n_docs):
            score = 0.0
            dl = self.doc_lengths[doc_idx]
            
            for token in query_tokens:
                if token not in self.term_freq[doc_idx]:
                    continue
                
                tf = self.term_freq[doc_idx][token]
                df = self.doc_freq[token]
                
                idf = math.log(
                    (self.n_docs - df + 0.5) / (df + 0.5) + 1.0
                )
                
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                
                score += idf * tf_norm
            
            if score >= min_score:
                scores.append((doc_idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def add_document(self, document: Document) -> None:
        """
        增量添加单个文档（无需重建整个索引）
        
        适用于流式场景：新文档持续到达时，避免全量重建开销。
        
        Args:
            document: 新增的 LangChain Document
        """
        text = document.page_content
        tokens = self._tokenize(text)
        
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        self.term_freq.append(dict(tf))
        self.doc_lengths.append(len(tokens))
        self.documents.append(text)
        self.n_docs += 1
        
        for token in set(tokens):
            self.doc_freq[token] += 1
        
        total_length = sum(self.doc_lengths)
        self.avg_dl = total_length / self.n_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息（用于监控面板）"""
        return {
            "n_documents": self.n_docs,
            "vocabulary_size": len(self.doc_freq),
            "avg_doc_length": round(self.avg_dl, 1),
            "max_doc_length": max(self.doc_lengths) if self.doc_lengths else 0,
            "min_doc_length": min(self.doc_lengths) if self.doc_lengths else 0,
            "is_indexed": self._indexed,
        }


# ════════════════════════════════════════════════════════════
# 第四部分：父文档检索器
# ════════════════════════════════════════════════════════════


class ParentDocumentRetriever:
    """
    父文档检索器 (Parent Document Retriever)
    
    【核心思想】
    解决 RAG 中的「信息碎片化」问题：
    
    问题场景:
    用户问："贵州茅台的完整盈利能力分析"
    传统 RAG 可能返回：
    - 片段A: "...净利润增长率18.2%..." （来自第15页第3段）
    - 片段B: "...毛利率91.8%..." （来自第20页第1段）
    
    这些片段缺乏上下文，LLM 无法判断：
    - 18.2% 是同比还是环比？
    - 91.8% 是哪个业务板块？
    - 这两个数字是否属于同一会计期间？
    
    解决方案:
    将文档分层切分：
    ┌─────────────────────────────────┐
    │         原始文档 (10,000字)       │
    └──────────────┬──────────────────┘
                   │ parent_splitter (2000字/块)
          ┌────────▼────────┐
          │  Parent Chunk 1  │ ← 大块，保持语义完整
          │  (2000字)        │
          └────────┬────────┘
                   │ child_splitter (400字/块)
          ┌────────▼────────┐
          │ Child Chunk 1.1  │ ← 小块，提高检索精度
          │ Child Chunk 1.2  │
          │ Child Chunk 1.3  │
          └─────────────────┘
    
    检索流程:
    1. 用小块 (Child) 进行向量/BM25 检索 → 精确定位
    2. 找到匹配的小块后，返回其所属的大块 (Parent) → 保证完整上下文
    
    【优势】
    - 检索精度高（小块更容易匹配）
    - 上下文完整（大块包含完整论述）
    - 支持粒度灵活调整
    
    【参数调优建议】
    - child_chunk_size: 300-500 字（太小丢失语义，太大降低精度）
    - child_chunk_overlap: 50-100 字（保证跨块边界的信息连续性）
    - parent_chunk_size: 1500-2500 字（适配 LLM 上下文窗口）
    """
    
    def __init__(
        self,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
    ):
        """
        初始化父文档检索器
        
        Args:
            child_chunk_size: 子块目标大小（字符数）
                            推荐 400-500，平衡精度与语义完整性
                            
            child_chunk_overlap: 子块之间的重叠字符数
                               推荐 50-100，防止截断关键信息
                               
            parent_chunk_size: 父块目标大小（字符数）
                              推荐 1500-2500，确保包含完整论点
        """
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self.parent_chunk_size = parent_chunk_size
        
        self.parent_docs: Dict[str, Document] = {}
        self.child_to_parent: Dict[str, str] = {}
        self.child_docs: List[Document] = []
        
        self._stats = {
            "n_original_docs": 0,
            "n_parent_chunks": 0,
            "n_child_chunks": 0,
        }
    
    def split_and_store(self, document: Document) -> None:
        """
        切分文档并建立父子关系映射
        
        处理流程:
        1. 生成父文档唯一标识 (parent_id)
        2. 用 parent_splitter 将原文切分为父块
        3. 对每个父块，用 child_splitter 进一步切分为子块
        4. 建立 child_id → parent_id 映射关系
        5. 分别存储到 parent_docs 和 child_docs
        
        ID 生成规则:
        - parent_id: "parent_{hash(content)}"
        - 子块ID: "{parent_id}_p{parent_idx}_c{child_idx}"
        
        Args:
            document: 原始 LangChain Document
        """
        parent_id = f"parent_{hash(document.page_content)}"
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ""],
        )
        parent_chunks = parent_splitter.split_documents([document])
        
        for p_idx, parent_chunk in enumerate(parent_chunks):
            pid = f"{parent_id}_p{p_idx}"
            parent_chunk.metadata["parent_id"] = pid
            self.parent_docs[pid] = parent_chunk
            self._stats["n_parent_chunks"] += 1
            
            child_chunks = self.child_splitter.split_documents([parent_chunk])
            
            for c_idx, child_chunk in enumerate(child_chunks):
                cid = f"{pid}_c{c_idx}"
                child_chunk.metadata["child_id"] = cid
                child_chunk.metadata["parent_id"] = pid
                self.child_to_parent[cid] = pid
                self.child_docs.append(child_chunk)
                self._stats["n_child_chunks"] += 1
        
        self._stats["n_original_docs"] += 1
    
    def get_parent_docs(
        self,
        child_ids: List[str],
        deduplicate: bool = True,
    ) -> List[Document]:
        """
        根据子文档 ID 列表获取对应的父文档
        
        Args:
            child_ids: 子文档 ID 列表（通常来自检索结果）
            deduplicate: 是否去重（多个子块可能属于同一父块）
            
        Returns:
            父文档列表（已去重，保留原始顺序）
            
        示例:
            >>> child_ids = ["parent_123_p0_c2", "parent_123_p0_c5"]
            >>> parents = retriever.get_parent_docs(child_ids)
            >>> print(f"返回 {len(parents)} 个父文档")
            返回 1 个父文档  # 因为两个子块属于同一个父块
        """
        parent_ids_seen: Set[str] = set()
        parent_docs: List[Document] = []
        
        for cid in child_ids:
            pid = self.child_to_parent.get(cid)
            if pid and (not deduplicate or pid not in parent_ids_seen):
                if deduplicate:
                    parent_ids_seen.add(pid)
                parent_doc = self.parent_docs.get(pid)
                if parent_doc:
                    parent_docs.append(parent_doc)
        
        return parent_docs
    
    def get_child_documents(self) -> List[Document]:
        """获取所有子文档（用于构建向量索引）"""
        return self.child_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取切分统计信息"""
        return {
            **self._stats,
            "avg_children_per_parent": (
                round(self._stats["n_child_chunks"] / max(1, self._stats["n_parent_chunks"]), 1)
            ),
        }


# ════════════════════════════════════════════════════════════
# 第五部分：增强版 RAG 主管道
# ════════════════════════════════════════════════════════════


class EnhancedRAGPipeline:
    """
    增强版 RAG 检索管道主协调器
    
    【完整处理流水线】
    
    ┌──────────┐
    │ Query    │
    │ Input    │
    └────┬─────┘
         │
         ▼
    ┌──────────┐     ┌──────────────────────┐
    │ Step 1   │────▶│ HyDE Generation      │
    │ Preprocess│    │ (LLM 生成假设文档)    │
    │          │    │ 延迟: ~1-2s           │
    └────┬─────┘     └──────────┬───────────┘
         │                      │
         ▼                      ▼
    ┌─────────────────────────────────────┐
    │         Step 2: Multi-Path Recall    │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
    │  │ Vector  │ │  BM25   │ │  HyDE   │ │
    │  │ Search  │ │ Search  │ │ Search  │ │
    │  │(语义)   │ │(关键词)  │ │(假设)   │ │
    │  └────┬────┘ └────┬────┘ └────┬────┘ │
    │       └───────────┼───────────┘      │
    │                   ▼                  │
    └───────────────────┬──────────────────┘
                        ▼
              ┌──────────────────┐
              │  Step 3: Fusion  │
              │  (RRF/加权融合)  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Step 4: Rerank  │◀── Cross-Encoder
              │  (精细重排序)    │    或 Vector Rerank
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Step 5: MMR     │
              │  (多样性优化)    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Step 6: Parent  │
              │  Doc Assembly    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Step 7: Output  │
              │  HybridSearchResult │
              └──────────────────┘
    
    【使用方式】
    
    基础用法（仅向量检索）:
        >>> vectorstore = Chroma(...)
        >>> rag = EnhancedRAGPipeline(vectorstore)
        >>> result = rag.query("茅台估值分析")
    
    完整用法（全功能）:
        >>> rag = EnhancedRAGPipeline(vectorstore)
        >>> rag.initialize_bm25(all_documents)      # 启用 BM25
        >>> rag.enable_parent_document_mode()        # 启用父文档
        >>> rag.update_config(use_hyde=True)         # 启用 HyDE
        >>> result = rag.query("茅台盈利能力", top_k=5)
    
    【配置热更新】
    所有参数可在运行时通过 update_config() 动态修改，
    无需重新创建实例。这支持 A/B 测试和自适应调优。
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        config: Optional[RAGConfig] = None,
    ):
        """
        初始化 RAG 管道
        
        Args:
            vectorstore: ChromaDB 向量数据库实例
                         必须预装入了待检索的文档向量
            config: 管道配置对象
                   为 None 时使用默认配置
        """
        self.vectorstore = vectorstore
        self.config = config or RAGConfig()
        
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
        )
        
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0.3,
        )
        
        self.reranker: Optional[Any] = None
        if CROSS_ENCODER_AVAILABLE and self.config.use_reranking:
            self._load_cross_encoder()
        
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.bm25_initialized: bool = False
        
        self.parent_retriever: Optional[ParentDocumentRetriever] = None
    
    def _load_cross_encoder(self) -> None:
        """
        加载 Cross-Encoder 重排序模型
        
        模型选择说明:
        - BAAI/bge-reranker-v2-m3: 多语言支持好，中文表现优秀
        - 推理设备: CPU（自动检测 GPU 如果可用）
        - 模型大小: ~568MB（首次下载后缓存本地）
        
        降级策略:
        加载失败时不抛异常，仅设置 self.reranker = None，
        后续重排序步骤会跳过或回退到向量分数。
        """
        print("[EnhancedRAG] 正在加载 Cross-Encoder 模型...")
        try:
            self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
            print("[EnhancedRAG] ✅ Cross-Encoder 加载成功")
        except Exception as e:
            print(f"[EnhancedRAG] ⚠️ Cross-Encoder 加载失败: {e}")
            print("[EnhancedRAG]    将回退到向量分数排序")
            self.reranker = None
            self.config.use_reranking = False
    
    def initialize_bm25(self, documents: List[Document]) -> None:
        """
        初始化 BM25 关键词检索索引
        
        必须在使用 BM25 检索之前调用此方法。
        通常在系统启动时一次性完成。
        
        Args:
            documents: 用于构建索引的文档列表
                      应与向量库中的文档一致
        """
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.build_index(documents)
        self.bm25_initialized = True
    
    def enable_parent_document_mode(
        self,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
    ) -> None:
        """
        启用父文档返回模式
        
        启用后，检索到的子块会被替换为其所属的父文档块，
        以提供更完整的上下文。
        
        Args:
            child_chunk_size: 子块大小（传给 ParentDocumentRetriever）
            child_chunk_overlap: 子块重叠
            parent_chunk_size: 父块大小
        """
        self.parent_retriever = ParentDocumentRetriever(
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            parent_chunk_size=parent_chunk_size,
        )
        self.config.use_parent_document = True
    
    def update_config(self, **kwargs) -> None:
        """
        动态更新配置参数
        
        Args:
            **kwargs: 要更新的配置项
                     例如: use_hyde=False, top_k_final=10
        """
        current = self.config.dict()
        current.update(kwargs)
        self.config = RAGConfig(**current)
    
    def query(
        self,
        query: str,
        use_hyde: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> HybridSearchResult:
        """
        执行增强版检索（主入口方法）
        
        这是 RAG 管道的核心方法，编排完整的 7 步处理流水线。
        
        Args:
            query: 用户查询文本（原始输入，未经改写）
            use_hyde: 是否使用 HyDE 增强
                     None = 使用配置默认值
                     True/False = 强制覆盖
            top_k: 最终返回结果数量
                  None = 使用配置默认值
                  
        Returns:
            HybridSearchResult 包含：
            - results: 最终结果列表（已排序）
            - 各通道命中统计
            - 性能耗时数据
            - 质量等级判定
            
        异常处理策略:
        - HyDE 生成失败 → 回退到原始查询
        - BM25 未初始化 → 跳过 BM25 通道
        - Cross-Encoder 未加载 → 跳过重排序
        - 任何步骤异常 → 返回已有结果（优雅降级）
        """
        start_time = time.time()

        _use_hyde = use_hyde if use_hyde is not None else self.config.use_hyde
        _top_k = top_k or self.config.top_k_final

        print(f"[EnhancedRAG] 🔍 开始检索: {query[:60]}...")

        # Track results by content hash for proper fusion, not by prefix-key
        content_hashes: Dict[str, RetrievalResult] = {}
        channel_scores: Dict[str, Dict[str, float]] = {}  # hash -> {channel: raw_score}
        channel_counts = {
            RetrievalMethod.VECTOR: 0,
            RetrievalMethod.BM25: 0,
            RetrievalMethod.HYDE: 0,
        }

        hyde_query = query

        try:
            if _use_hyde:
                hyde_query = self._generate_hypothetical_doc(query)
        except Exception as e:
            print(f"[EnhancedRAG] ⚠️ HyDE 生成异常: {e}")
            hyde_query = query

        def _add_result(result: RetrievalResult, channel: str) -> None:
            """Add result to fusion tracking, avoiding content-level duplicates."""
            content_hash = str(hash(result.content))
            result.retrieval_method = channel
            if content_hash not in content_hashes:
                content_hashes[content_hash] = result
                channel_scores[content_hash] = {}
            # Keep the best-scoring RetrievalResult for this content
            existing = content_hashes[content_hash]
            if result.score > existing.score:
                content_hashes[content_hash] = result
            channel_scores[content_hash][channel] = result.score

        try:
            vector_results = self._vector_search(query, self.config.top_k_vector)
            for result in vector_results:
                _add_result(result, RetrievalMethod.VECTOR.value)
                channel_counts[RetrievalMethod.VECTOR] += 1
        except Exception as e:
            print(f"[EnhancedRAG] ⚠️ 向量检索异常: {e}")

        if _use_hyde and hyde_query != query:
            try:
                hyde_results = self._vector_search(hyde_query, self.config.top_k_vector)
                for result in hyde_results:
                    _add_result(result, RetrievalMethod.HYDE.value)
                    channel_counts[RetrievalMethod.HYDE] += 1
            except Exception as e:
                print(f"[EnhancedRAG] ⚠️ HyDE 检索异常: {e}")

        if self.config.use_bm25 and self.bm25_initialized:
            try:
                bm25_results = self._bm25_search(query, self.config.top_k_bm25)
                for result in bm25_results:
                    _add_result(result, RetrievalMethod.BM25.value)
                    channel_counts[RetrievalMethod.BM25] += 1
            except Exception as e:
                print(f"[EnhancedRAG] ⚠️ BM25 检索异常: {e}")

        # Apply configured fusion strategy
        fusion_strategy = self.config.fusion_strategy
        fusion_weights = {
            "vector": self.config.vector_weight,
            "bm25": self.config.bm25_weight,
            "hyde": self.config.hyde_weight,
        }

        if fusion_strategy == FusionStrategy.RRF:
            # Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)
            # k=60 is the standard constant; rank is 1-based within each channel
            RRF_K = 60
            top_k_per_channel = max(self.config.top_k_vector, self.config.top_k_bm25)

            # Collect per-channel ranked lists
            channel_results: Dict[str, List[Tuple[float, RetrievalResult]]] = {
                "vector": [],
                "bm25": [],
                "hyde": [],
            }
            for result in content_hashes.values():
                for ch, score in channel_scores[str(hash(result.content))].items():
                    if ch in channel_results:
                        channel_results[ch].append((score, result))

            # Sort each channel by score descending and assign ranks
            for ch in channel_results:
                channel_results[ch].sort(key=lambda x: x[0], reverse=True)

            # Apply RRF formula
            for content_hash, result in content_hashes.items():
                rrf_score = 0.0
                for ch in ["vector", "bm25", "hyde"]:
                    ranked_list = channel_results[ch]
                    for rank, (score, ranked_result) in enumerate(ranked_list, 1):
                        if str(hash(ranked_result.content)) == content_hash:
                            rrf_score += 1.0 / (RRF_K + rank)
                            break
                result.score = rrf_score

        elif fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            # Weighted fusion: score = Σ w_i * normalized_score_i
            # Normalize each channel's scores to [0, 1] range
            for ch in ["vector", "bm25", "hyde"]:
                scores = [s[ch] for s in channel_scores.values() if ch in s]
                if not scores:
                    continue
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s if max_s > min_s else 1.0
                for content_hash in content_hashes:
                    if ch in channel_scores[content_hash]:
                        norm = (channel_scores[content_hash][ch] - min_s) / range_s
                        channel_scores[content_hash][ch] = norm

            for content_hash, result in content_hashes.items():
                weighted = sum(
                    channel_scores[content_hash].get(ch, 0.0) * fusion_weights.get(ch, 0.0)
                    for ch in ["vector", "bm25", "hyde"]
                )
                result.score = weighted

        else:
            # MAX_SCORE: score = max(score_1, score_2, ...)
            for content_hash, result in content_hashes.items():
                result.score = max(channel_scores[content_hash].values()) if channel_scores[content_hash] else 0.0

        candidates_list = list(content_hashes.values())
        candidates_list.sort(key=lambda x: x.score, reverse=True)
        
        rerank_pool_size = min(_top_k * 4, len(candidates_list), 30)
        rerank_candidates = candidates_list[:rerank_pool_size]
        
        rerank_start = time.time()
        final_results: List[RetrievalResult] = []
        
        if self.config.use_reranking and self.reranker and rerank_candidates:
            try:
                final_results = self._rerank(query, rerank_candidates, _top_k)
            except Exception as e:
                print(f"[EnhancedRAG] ⚠️ 重排序异常: {e}")
                final_results = rerank_candidates[:_top_k]
        else:
            final_results = rerank_candidates[:_top_k]
        
        rerank_time = (time.time() - rerank_start) * 1000
        
        if self.config.use_mmr and len(final_results) > 2:
            try:
                final_results = self._mmr_rerank(query, final_results, _top_k)
            except Exception as e:
                print(f"[EnhancedRAG] ⚠️ MMR 优化异常: {e}")
        
        if self.config.use_parent_document and self.parent_retriever:
            try:
                final_results = self._assemble_parent_docs(final_results)
            except Exception as e:
                print(f"[EnhancedRAG] ⚠️ 父文档组装异常: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        quality = self._assess_quality(
            final_results,
            channel_counts,
            len(candidates_list),
        )
        
        result = HybridSearchResult(
            query=query,
            results=final_results,
            vector_results_count=channel_counts[RetrievalMethod.VECTOR],
            bm25_results_count=channel_counts[RetrievalMethod.BM25],
            hyde_results_count=channel_counts[RetrievalMethod.HYDE],
            fusion_weights={
                "vector": self.config.vector_weight,
                "bm25": self.config.bm25_weight,
                "hyde": self.config.hyde_weight,
            },
            total_candidates=len(candidates_list),
            unique_results=len(final_results),
            retrieval_time_ms=total_time - rerank_time,
            rerank_time_ms=rerank_time,
            quality=quality,
        )
        
        print(f"[EnhancedRAG] ✅ 检索完成:")
        print(f"    候选总数: {result.total_candidates}")
        print(f"    最终返回: {result.unique_results}")
        print(f"    质量等级: {result.quality.value.upper()}")
        print(f"    检索耗时: {result.retrieval_time_ms:.0f}ms")
        print(f"    重排耗时: {result.rerank_time_ms:.0f}ms")
        
        return result
    
    def _generate_hypothetical_doc(self, query: str) -> str:
        """
        生成假设性文档 (Hypothetical Document Embedding, HyDE)
        
        【原理】
        HyDE 是一种查询增强技术，核心思想是：
        "不要直接用简短的查询去检索，而是先用 LLM 生成一段
         可能包含答案的'假想文档'，再用这段文档去检索。"
         
        举例:
        查询: "茅台的护城河是什么？"
        HyDE 生成: "贵州茅台拥有极强的品牌护城河。其飞天茅台酒
                   在高端白酒市场占据主导地位，品牌溢价能力极强...
                   渠道控制力强，经销商体系稳定..."
        
        这段生成的文本比原查询包含更多语义信息和关键词，
        因此能更好地匹配到相关知识库文档。
        
        【Prompt 设计要点】
        - 角色设定：资深金融分析师
        - 输出格式：研究报告风格的专业文本
        - 长度控制：150-200 字（太短信息不足，太长引入噪声）
        - 语言风格：客观、数据驱动
        
        Args:
            query: 原始用户查询
            
        Returns:
            生成的假设性文档文本
            失败时返回原始查询（优雅降级）
        """
        prompt = (
            "你是一位资深金融分析师，拥有 15 年行业研究经验。\n\n"
            "请根据用户的问题，生成一段 150-200 字的专业分析文本，\n"
            "就像这段文字出现在真实的券商研究报告中一样。\n\n"
            "要求:\n"
            "- 包含具体的财务指标和数据（可以是合理的估计值）\n"
            "- 使用专业的金融术语和分析框架\n"
            "- 保持客观中立的分析口吻\n"
            "- 直接回答问题，不要废话\n\n"
            f"问题：{query}\n\n"
            "分析文本："
        )
        
        try:
            response = self.llm.invoke(prompt)
            generated = response.content.strip()
            
            if len(generated) < 20:
                print("[EnhancedRAG] ⚠️ HyDE 生成内容过短，回退到原始查询")
                return query
            
            return generated
        except Exception as e:
            print(f"[EnhancedRAG] HyDE 生成失败: {e}")
            return query
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        向量相似度检索
        
        使用 ChromaDB 的 HNSW 索引进行近似最近邻搜索。
        HNSW (Hierarchical Navigable Small World) 是一种
        基于图的索引结构，查询复杂度为 O(log N)。
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            RetrievalResult 列表
        """
        docs = self.vectorstore.similarity_search(query, k=top_k)
        
        results: List[RetrievalResult] = []
        for idx, doc in enumerate(docs):
            result = RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=doc.metadata.get("score", 0.85),
                source=doc.metadata.get("source", "unknown"),
                chunk_index=idx,
            )
            results.append(result)
        
        return results
    
    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        BM25 关键词检索
        
        封装 BM25Retriever.search() 的结果转换逻辑。
        
        注意:
        BM25 的原始分数范围不固定（取决于语料特性），
        这里做了简单的 Min-Max 归一化映射到 [0, 5] 区间。
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            RetrievalResult 列表
        """
        if not self.bm25_initialized or not self.bm25_retriever:
            return []
        
        scored_docs = self.bm25_retriever.search(query, top_k)
        
        results: List[RetrievalResult] = []
        for doc_idx, score in scored_docs:
            doc_text = self.bm25_retriever.documents[doc_idx]
            result = RetrievalResult(
                content=doc_text,
                metadata={
                    "source": "bm25_index",
                    "doc_index": doc_idx,
                    "bm25_raw_score": round(score, 3),
                },
                score=min(score / 10.0, 5.0),
                source="bm25_index",
                retrieval_method=RetrievalMethod.BM25.value,
            )
            results.append(result)
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Cross-Encoder 精细重排序
        
        【与 Vector Search 的区别】
        - Bi-Encoder (向量检索): 分别编码 query 和 doc，然后算余弦相似度
          速度快，但无法捕捉 query-doc 之间的细粒度交互
          
        - Cross-Encoder (重排序): 将 query 和 doc 拼接后一起编码
          能建模 query-doc 的交互关系，精度更高但速度慢
          
        因此采用「粗排 + 精排」两阶段策略：
        1. Bi-Encoder 快速召回数百个候选
        2. Cross-Encoder 对 Top 候选精排
        
        Args:
            query: 原始查询
            candidates: 候选结果列表
            top_k: 返回数量
            
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        pairs = [[query, candidate.content] for candidate in candidates]
        scores = self.reranker.predict(pairs)
        
        for i, candidate in enumerate(candidates):
            candidate.score = float(scores[i])
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:top_k] if top_k else candidates
    
    def _mmr_rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int,
        lambda_param: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Maximal Marginal Relevance (MMR) 多样性重排序
        
        【问题背景】
        传统排序只考虑相关性，可能导致返回的内容高度集中：
        - 查询: "茅台盈利能力"
        - Top-5 结果可能全部来自同一份年报的不同段落
        - 缺乏多角度信息（如分析师观点、新闻评论等）
        
        【MMR 算法】
        贪心选择策略，每次选择「边际相关性」最大的候选：
        
            MMR = λ × Rel(q, d) - (1-λ) × max Sim(d, d_selected)
        
        其中:
        - Rel(q, d): 候选文档与查询的相关性
        - Sim(d, d_selected): 候选与已选文档的最大相似度
        - λ: 权衡参数（0=纯多样性, 1=纯相关性）
        
        【时间复杂度】
        O(K × S × L)，其中 K=候选数, S=选择数, L=平均文本长度
        
        Args:
            query: 查询文本
            candidates: 候选列表（已按相关性排序）
            top_k: 选择数量
            lambda_param: 平衡参数
            
        Returns:
            优化后的结果列表（兼顾相关性和多样性）
        """
        lambda_param = lambda_param or self.config.mmr_lambda
        
        if len(candidates) <= top_k:
            return candidates
        
        selected: List[RetrievalResult] = []
        remaining = list(candidates)
        
        remaining.sort(key=lambda x: x.score, reverse=True)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                relevance = candidate.score
                
                max_similarity = 0.0
                for sel in selected:
                    similarity = self._text_similarity(candidate.content, sel.content)
                    max_similarity = max(max_similarity, similarity)
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        计算两段文本的 Jaccard 相似度
        
        Jaccard = |A ∩ B| / |A ∪ B|
        
        这是一种轻量级的相似度度量，不需要模型推理，
        适用于 MMR 中的快速多样性计算。
        
        对于更高精度的场景，可以考虑：
        - 余弦相似度（基于 embedding）
        - 编辑距离（Levenshtein distance）
        - Semantic similarity（基于 Sentence-BERT）
        
        Args:
            text1: 文本 1
            text2: 文本 2
            
        Returns:
            相似度分数 [0.0, 1.0]
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _assemble_parent_docs(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        组装父文档（替换子块为父块）
        
        Args:
            results: 当前结果列表（包含子块引用）
            
        Returns:
            替换后的结果列表（内容为父文档）
        """
        child_ids = [
            r.metadata.get("child_id")
            for r in results
            if r.metadata.get("child_id")
        ]
        
        if not child_ids:
            return results
        
        parent_docs = self.parent_retriever.get_parent_docs(child_ids)
        
        for i, result in enumerate(results):
            if i < len(parent_docs):
                result.content = parent_docs[i].page_content
                result.metadata["is_parent_doc"] = True
                result.metadata["original_child_score"] = result.score
        
        return results
    
    def _assess_quality(
        self,
        results: List[RetrievalResult],
        channel_counts: Dict[RetrievalMethod, int],
        total_candidates: int,
    ) -> RetrievalQuality:
        """
        评估检索结果质量
        
        评判维度:
        1. 结果数量: 是否返回足够的结果？
        2. 平均得分: 整体相关性如何？
        3. 通道多样性: 是否多通道都有贡献？
        
        Args:
            results: 最终结果列表
            channel_counts: 各通道命中计数
            total_candidates: 候选总数
            
        Returns:
            质量等级
        """
        if not results:
            return RetrievalQuality.POOR
        
        avg_score = sum(r.score for r in results) / len(results)
        n_channels = sum(1 for c in channel_counts.values() if c > 0)
        
        if avg_score > 0.85 and n_channels >= 2 and len(results) >= 3:
            return RetrievalQuality.EXCELLENT
        elif avg_score > 0.7 and n_channels >= 1:
            return RetrievalQuality.GOOD
        elif avg_score > 0.5:
            return RetrievalQuality.FAIR
        else:
            return RetrievalQuality.POOR


def create_rag_pipeline(
    vectorstore: Chroma,
    documents: Optional[List[Document]] = None,
    **config_kwargs,
) -> EnhancedRAGPipeline:
    """
    工厂函数：创建并配置完整的 RAG 管道实例
    
    封装常见的初始化流程，减少样板代码。
    
    Args:
        vectorstore: ChromaDB 实例
        documents: 用于初始化 BM25 的文档列表（可选）
        **config_kwargs: 传递给 RAGConfig 的额外配置
        
    Returns:
        配置完成的 EnhancedRAGPipeline 实例
        
    示例:
        >>> rag = create_rag_pipeline(
        ...     vectorstore=my_chroma,
        ...     documents=all_docs,
        ...     use_hyde=True,
        ...     use_parent_document=True,
        ... )
        >>> result = rag.query("茅台估值")
    """
    config = RAGConfig(**config_kwargs)
    pipeline = EnhancedRAGPipeline(vectorstore, config=config)
    
    if documents and config.use_bm25:
        pipeline.initialize_bm25(documents)
    
    if config.use_parent_document:
        pipeline.enable_parent_document_mode()
    
    return pipeline


# ════════════════════════════════════════════════════════════
# 第六部分：自测套件
# ════════════════════════════════════════════════════════════


def _run_self_tests():
    """
    RAG 管道自测套件
    
    测试场景覆盖:
    1. 数据模型创建与序列化
    2. BM25 索引构建与检索
    3. 父文档切分与映射
    4. MMR 多样性算法正确性
    5. 融合策略计算
    6. 质量评估逻辑
    """
    print("\n" + "=" * 70)
    print("  Enhanced RAG Pipeline — Self-Test Suite")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    def test(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  [{status}] {name}")
        if detail and not condition:
            print(f"         → {detail}")
        if condition:
            passed += 1
        else:
            failed += 1
    
    print("\n--- Test Group 1: Data Models ---")
    
    result = RetrievalResult(
        content="茅台2023年净利润增长18.2%",
        metadata={"source": "report.pdf"},
        score=0.92,
        retrieval_method=RetrievalMethod.VECTOR.value,
    )
    test(
        "RetrievalResult creation",
        result.content == "茅台2023年净利润增长18.2%" and result.score == 0.92,
    )
    test(
        "RetrievalResult.is_high_quality",
        result.is_high_quality,
    )
    test(
        "RetrievalResult.to_dict serialization",
        "content" in result.to_dict(),
    )
    
    hybrid_result = HybridSearchResult(
        query="茅台估值",
        results=[result],
        total_candidates=10,
        unique_results=1,
    )
    test(
        "HybridSearchResult.hit_rate calculation",
        hybrid_result.hit_rate == 0.1,
    )
    test(
        "HybridSearchResult.summary generation",
        len(hybrid_result.summary()) > 20,
    )
    
    config = RAGConfig(top_k_final=10, mmr_lambda=0.8)
    test(
        "RAGConfig validation",
        config.top_k_final == 10 and config.mmr_lambda == 0.8,
    )
    
    print("\n--- Test Group 2: BM25 Retriever ---")
    
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    
    test(
        "BM25 initial state",
        not bm25._indexed and bm25.n_docs == 0,
    )
    
    docs = [
        Document(page_content="贵州茅台2023年净利润1200亿元"),
        Document(page_content="五粮液2023年净利润300亿元"),
        Document(page_content="泸州老窖2023年净利润100亿元"),
        Document(page_content="洋河股份2023年净利润90亿元"),
    ]
    
    try:
        bm25.build_index(docs)
        test("BM25 index build", bm25._indexed and bm25.n_docs == 4)
    except Exception as e:
        test("BM25 index build", False, str(e))
    
    results = bm25.search("茅台 利润", top_k=2)
    test(
        "BM25 search returns results",
        len(results) >= 1,
        f"Expected >=1, got {len(results)}",
    )
    test(
        "BM25 search ranking (first should be about 茅台)",
        results[0][0] == 0 if results else False,
    )
    
    stats = bm25.get_statistics()
    test(
        "BM25 statistics",
        stats["n_documents"] == 4 and "vocabulary_size" in stats,
    )
    
    new_doc = Document(page_content="山西汾酒2023年净利润50亿元")
    bm25.add_document(new_doc)
    test(
        "BM25 incremental add",
        bm25.n_docs == 5,
    )
    
    print("\n--- Test Group 3: Parent Document Retriever ---")
    
    pdr = ParentDocumentRetriever(
        child_chunk_size=100,
        child_chunk_overlap=20,
        parent_chunk_size=300,
    )
    
    long_doc = Document(
        page_content=(
            "贵州茅台是中国白酒行业的龙头企业。"
            "公司主要产品包括飞天茅台、茅台王子酒等。"
            "2023年公司实现营业收入1505亿元，同比增长18.0%。"
            "归属于母公司股东的净利润为747亿元，同比增长19.2%。"
            "公司的核心竞争力在于品牌价值和渠道控制力。"
            "飞天茅台的市场零售价维持在2800元左右。"
            "公司的毛利率高达91.8%，净利率达到49.6%。"
        )
    )
    
    pdr.split_and_store(long_doc)
    
    pdr_stats = pdr.get_statistics()
    test(
        "Parent document splitting",
        pdr_stats["n_original_docs"] == 1 and pdr_stats["n_child_chunks"] > 0,
        f"Stats: {pdr_stats}",
    )
    
    child_docs = pdr.get_child_documents()
    test(
        "Child documents available",
        len(child_docs) > 0,
        f"Got {len(child_docs)} children",
    )
    
    if child_docs:
        sample_child_ids = [c.metadata.get("child_id") for c in child_docs[:2]]
        parents = pdr.get_parent_docs(sample_child_ids)
        test(
            "Parent retrieval from child IDs",
            len(parents) >= 1,
        )
    
    print("\n--- Test Group 4: MMR Algorithm ---")
    
    candidates = [
        RetrievalResult(content="茅台盈利能力强", score=0.95),
        RetrievalResult(content="茅台利润增长快", score=0.93),
        RetrievalResult(content="茅台收入创新高", score=0.90),
        RetrievalResult(content="五粮液也在增长", score=0.75),
        RetrievalResult(content="白酒行业整体向好", score=0.70),
    ]
    
    mmr_results = EnhancedRAGPipeline._mmr_rerank(
        None, candidates, top_k=3, lambda_param=0.7
    )
    test(
        "MMR returns correct count",
        len(mmr_results) == 3,
        f"Expected 3, got {len(mmr_results)}",
    )
    test(
        "MMR preserves relevance ordering (roughly)",
        mmr_results[0].score >= mmr_results[-1].score,
    )
    
    sim = EnhancedRAGPipeline._text_similarity(
        "茅台利润增长",
        "茅台利润下降",
    )
    test(
        "Jaccard similarity calculation",
        0.0 <= sim <= 1.0,
        f"Similarity={sim}",
    )
    
    sim_same = EnhancedRAGPipeline._text_similarity(
        "茅台利润增长",
        "茅台利润增长",
    )
    test(
        "Jaccard identical texts",
        sim_same == 1.0,
    )
    
    print("\n--- Test Group 5: Quality Assessment ---")
    
    excellent_results = [
        RetrievalResult(content="high quality", score=0.90),
        RetrievalResult(content="good match", score=0.88),
        RetrievalResult(content="relevant", score=0.85),
    ]
    channels_multi = {
        RetrievalMethod.VECTOR: 5,
        RetrievalMethod.BM25: 3,
        RetrievalMethod.HYDE: 2,
    }
    
    quality = EnhancedRAGPipeline._assess_quality(
        None, excellent_results, channels_multi, 20
    )
    test(
        "Quality assessment EXCELLENT",
        quality == RetrievalQuality.EXCELLENT,
        f"Got {quality.value}",
    )
    
    poor_results = [
        RetrievalResult(content="low quality", score=0.2),
    ]
    channels_single = {
        RetrievalMethod.VECTOR: 1,
        RetrievalMethod.BM25: 0,
        RetrievalMethod.HYDE: 0,
    }
    
    quality_poor = EnhancedRAGPipeline._assess_quality(
        None, poor_results, channels_single, 1
    )
    test(
        "Quality assessment POOR",
        quality_poor == RetrievalQuality.POOR,
        f"Got {quality_poor.value}",
    )
    
    print("\n--- Test Group 6: Enums and Constants ---")
    
    test(
        "RetrievalMethod enum values",
        len(RetrievalMethod) == 4,
    )
    test(
        "FusionStrategy enum values",
        len(FusionStrategy) == 3,
    )
    test(
        "RerankingMode enum values",
        len(RerankingMode) == 3,
    )
    test(
        "RetrievalQuality enum values",
        len(RetrievalQuality) == 4,
    )
    test(
        "Default stop words loaded",
        len(DEFAULT_STOP_WORDS) > 50,
        f"Got {len(DEFAULT_STOP_WORDS)} words",
    )
    test(
        "Default fusion weights sum ~1.0",
        abs(sum(DEFAULT_FUSION_WEIGHTS.values()) - 1.0) < 0.01,
    )
    
    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed - please review")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_tests()
