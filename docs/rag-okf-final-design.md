Recommendation: Evidence-first Hybrid RAG + Curated OKF

For this dataset, the best architecture is not pure RAG and not a fully knowledge-graph-driven system.

Use a three-layer architecture:

Transcript evidence layer
Timestamp-preserving transcript chunks for examples, quotations, context, and citations.

Curated OKF concept layer
Consolidated concepts such as “avoidant attachment communication,” “post-breakup no-contact,” or “MBTI compatibility limitations,” including supporting and opposing creator positions.

Query router + conflict-aware synthesis layer
Decides whether the question needs direct evidence, synthesized principles, contrasting viewpoints, or both.

I would call this:

Evidence-first Hybrid RAG

                         ┌──────────────────────┐
                         │     User question    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Query analysis/router │
                         │ intent, topic, filters│
                         │ risk, evidence needs  │
                         └──────┬────────┬──────┘
                                │        │
             concept-oriented ──┘        └── evidence/example-oriented
                                │
             ┌──────────────────▼───┐  ┌─▼─────────────────────────┐
             │ OKF concept retrieval │  │ Transcript hybrid search  │
             │ curated principles    │  │ dense + sparse + metadata │
             │ claims + disagreements│  │ timestamps + video context│
             └──────────┬────────────┘  └─────────────┬─────────────┘
                        │                              │
                        └──────────────┬───────────────┘
                                       │
                           ┌───────────▼───────────┐
                           │ Candidate reranking   │
                           │ diversity + authority │
                           │ conflict detection    │
                           └───────────┬───────────┘
                                       │
                           ┌───────────▼───────────┐
                           │ Grounded generation   │
                           │ advice + uncertainty  │
                           │ timestamped citations │
                           └───────────────────────┘
1. Why this is best for dating advice
Pure RAG’s weakness here

Pure transcript RAG will work for questions such as:

“What are examples of a good first message?”

“What did creator X say about no-contact?”

“Find advice about dating an INFP.”

But it will struggle with questions such as:

“What should I actually do after being ghosted?”

“Do most creators recommend no-contact?”

“Why do some experts say to confront avoidant partners while others say to give them space?”

Dating advice videos repeat the same ideas with slightly different wording. Pure vector retrieval can therefore return ten near-duplicate chunks from three videos instead of ten distinct perspectives. It can also mistake popularity for consensus because one creator may have many videos expressing the same position.

Why the OKF layer helps

The curated layer gives the model a stable map of:

recurring concepts;

recommended actions;

applicable conditions;

exceptions;

risks;

competing schools of thought;

source coverage;

strength of evidence;

supporting transcript passages.

OKF v0.1 is particularly suitable as an interchange and editorial format because it represents knowledge as Markdown files with YAML frontmatter rather than requiring a proprietary graph runtime. Its standard fields include items such as type, title, description, resource, tags, and timestamp. However, it is still an early specification, so your internal schema should be versioned independently rather than tightly coupling the application to every OKF convention. 
Google Cloud

Why you still need raw transcripts

The OKF layer must never replace evidence.

Curated knowledge inevitably loses:

wording and nuance;

situational examples;

speaker qualifications;

surrounding context;

timestamps;

signs that a creator was joking, exaggerating, or discussing an edge case.

Therefore:

OKF = navigation, synthesis, and editorial knowledge
RAG = evidence, provenance, context, and citation
2. Recommended production stack
Component	Recommendation
Language	Python 3.12
API	FastAPI
Data validation	Pydantic v2
Raw metadata	PostgreSQL
Vector database	Qdrant
Embedding	BGE-M3 initially
Sparse retrieval	BGE-M3 sparse vectors or BM25
Reranker	BGE reranker v2 multilingual or another evaluated multilingual cross-encoder
LLM	Provider abstraction supporting structured JSON output
OKF storage	Git-managed Markdown + YAML
Background jobs	Celery/Dramatiq + Redis, or simple CLI jobs initially
Evaluation	pytest + custom retrieval benchmark + Ragas/DeepEval-style metrics
Observability	OpenTelemetry + structured logs
Object storage	Local filesystem initially, S3/R2 in production

BGE-M3 is a strong starting point because it supports multilingual text, dense retrieval, learned sparse retrieval, and multi-vector interaction in one model. Its documentation lists support for more than 100 languages and inputs up to 8,192 tokens. 
BGE Model

For your size, approximately 6.7 MB of transcripts, Qdrant is comfortably sufficient. The reason to choose it is not scale but its clean support for:

named dense and sparse vectors;

payload filtering;

hybrid fusion;

multistage retrieval;

reranking candidates;

custom scoring.

Qdrant recommends ordinary RRF as a safe fusion default when you do not yet have an evaluation set, then weighted RRF once you have enough labeled queries to tune weights. 
Qdrant

3. Data model

Do not treat a chunk as only text plus an embedding. Use explicit provenance.

Transcript chunk
Python
실행됨
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    chunk_id: str
    video_id: str
    channel_id: str
    channel_name: str
    title: str

    text: str
    language: Literal["ko", "en", "mixed"]

    start_seconds: float
    end_seconds: float
    timestamp_url: str

    category: str
    subcategories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    views: int | None = None
    published_at: datetime | None = None

    speaker: str | None = None
    chunk_index: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None

    source_type: Literal["transcript"] = "transcript"
    ingestion_version: str
    transcript_hash: str
Curated claim

A concept is too broad to retrieve directly. Store atomic claims beneath it.

Python
실행됨
class KnowledgeClaim(BaseModel):
    claim_id: str
    concept_id: str

    statement: str
    normalized_action: str | None = None

    stance: Literal[
        "supports",
        "opposes",
        "conditional",
        "neutral",
        "warning",
    ]

    applies_when: list[str] = Field(default_factory=list)
    does_not_apply_when: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    evidence_chunk_ids: list[str]
    creator_ids: list[str]

    evidence_count: int
    creator_count: int

    curation_status: Literal[
        "machine_extracted",
        "machine_reviewed",
        "human_reviewed",
    ]

    confidence: float
    last_reviewed_at: datetime | None = None

The creator_count field is important. Twenty supporting chunks from one channel should not look like twenty independent sources.

4. File structure
dating-advice-rag/
├── pyproject.toml
├── README.md
├── .env.example
├── docker-compose.yml
├── alembic.ini
│
├── config/
│   ├── categories.yaml
│   ├── retrieval.yaml
│   ├── safety.yaml
│   ├── prompts.yaml
│   └── creator_profiles.yaml
│
├── data/
│   ├── raw/
│   │   └── videos/
│   │       └── {video_id}/
│   │           ├── metadata.yaml
│   │           └── transcript.jsonl
│   │
│   ├── normalized/
│   │   ├── videos.jsonl
│   │   └── transcript_segments.jsonl
│   │
│   ├── chunks/
│   │   ├── transcript_chunks.jsonl
│   │   └── chunk_manifest.json
│   │
│   ├── okf/
│   │   ├── bundle.yaml
│   │   ├── concepts/
│   │   │   ├── no-contact-after-breakup.md
│   │   │   ├── avoidant-attachment.md
│   │   │   └── first-message.md
│   │   ├── claims/
│   │   │   ├── claim_no_contact_001.md
│   │   │   └── claim_no_contact_002.md
│   │   ├── creators/
│   │   │   └── {channel_id}.md
│   │   └── taxonomies/
│   │       ├── topics.md
│   │       ├── relationship-stages.md
│   │       └── conflict-types.md
│   │
│   └── eval/
│       ├── retrieval_queries.jsonl
│       ├── answer_cases.jsonl
│       ├── conflict_cases.jsonl
│       └── citation_cases.jsonl
│
├── src/
│   └── dating_rag/
│       ├── api/
│       │   ├── app.py
│       │   ├── dependencies.py
│       │   └── routes/
│       │       ├── chat.py
│       │       ├── search.py
│       │       └── admin.py
│       │
│       ├── domain/
│       │   ├── models.py
│       │   ├── enums.py
│       │   └── exceptions.py
│       │
│       ├── ingestion/
│       │   ├── loader.py
│       │   ├── normalizer.py
│       │   ├── language.py
│       │   ├── chunker.py
│       │   ├── deduplicator.py
│       │   └── indexer.py
│       │
│       ├── embeddings/
│       │   ├── base.py
│       │   ├── bge_m3.py
│       │   └── batching.py
│       │
│       ├── retrieval/
│       │   ├── query_analyzer.py
│       │   ├── filters.py
│       │   ├── transcript_retriever.py
│       │   ├── concept_retriever.py
│       │   ├── hybrid_fusion.py
│       │   ├── reranker.py
│       │   ├── diversification.py
│       │   └── context_expander.py
│       │
│       ├── knowledge/
│       │   ├── okf_loader.py
│       │   ├── concept_extractor.py
│       │   ├── claim_clusterer.py
│       │   ├── conflict_detector.py
│       │   ├── curator.py
│       │   └── validator.py
│       │
│       ├── generation/
│       │   ├── prompt_builder.py
│       │   ├── context_builder.py
│       │   ├── answer_generator.py
│       │   ├── citation_validator.py
│       │   └── safety.py
│       │
│       ├── storage/
│       │   ├── postgres.py
│       │   ├── qdrant.py
│       │   └── repositories.py
│       │
│       ├── evaluation/
│       │   ├── retrieval.py
│       │   ├── generation.py
│       │   ├── citations.py
│       │   ├── conflicts.py
│       │   └── reports.py
│       │
│       └── settings.py
│
├── scripts/
│   ├── ingest.py
│   ├── build_chunks.py
│   ├── index_transcripts.py
│   ├── extract_claims.py
│   ├── build_okf.py
│   ├── validate_okf.py
│   └── run_eval.py
│
└── tests/
    ├── unit/
    ├── integration/
    └── golden/
5. Chunking pipeline
Recommended chunking

Do semantic, timestamp-aware windowing, not fixed character slicing.

Target:

250–450 tokens per chunk

hard maximum around 600 tokens

40–80 token overlap

usually 30–90 seconds of speech

preserve sentence and topic boundaries

retain exact first and last timestamps

split aggressively on clear topic changes

keep short examples with the setup that explains them

For Korean transcripts, token counts vary considerably by tokenizer, so store both:

model-token count;

character count.

Chunking algorithm

Parse timestamped transcript rows.

Clean ASR artifacts without rewriting meaning.

Group sentences into micro-segments.

Detect topic transitions using:

pauses;

discourse markers;

embedding similarity drops;

explicit section headings;

maximum token size.

Build chunks.

Add a small overlap from complete previous sentences.

Store adjacent chunk IDs.

Generate a short retrieval-only summary, but preserve raw text separately.

Python
실행됨
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    token_count: int


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    text: str
    segments: tuple[Segment, ...]


def chunk_segments(
    segments: Iterable[Segment],
    target_tokens: int = 360,
    max_tokens: int = 560,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    source = list(segments)
    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens

        if not current:
            return

        chunks.append(
            Chunk(
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(item.text for item in current),
                segments=tuple(current),
            )
        )

        overlap: list[Segment] = []
        count = 0

        for item in reversed(current):
            if count + item.token_count > overlap_tokens and overlap:
                break
            overlap.append(item)
            count += item.token_count

        current = list(reversed(overlap))
        current_tokens = sum(item.token_count for item in current)

    for segment in source:
        projected = current_tokens + segment.token_count

        if current and projected > max_tokens:
            flush()

        current.append(segment)
        current_tokens += segment.token_count

        # Replace this with semantic boundary detection.
        ends_topic = (
            segment.text.endswith(("정리해 보면", "결론적으로", "다음으로"))
            or segment.end - segment.start > 45
        )

        if current_tokens >= target_tokens and ends_topic:
            flush()

    flush()
    return chunks
Parent-child retrieval

Store two representations:

child chunks: 250–450 tokens, optimized for retrieval;

parent windows: approximately 800–1,500 tokens or adjacent chunk windows, used for generation.

Search children, then expand the winning results to include:

the preceding chunk;

the matching chunk;

the following chunk.

This prevents a source from being cited after retrieving only the punchline and losing its qualification.

6. Indexing and hybrid search

Use two Qdrant collections initially:

transcript_chunks
knowledge_claims

Avoid placing chunks and claims into one collection because they have different retrieval semantics and scoring.

BGE-M3 embedding adapter
Python
실행됨
from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel


@dataclass
class EmbeddedBatch:
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
        )

    def encode(self, texts: list[str]) -> EmbeddedBatch:
        output = self.model.encode(
            texts,
            batch_size=16,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        sparse_vectors: list[dict[int, float]] = []
        for lexical_weights in output["lexical_weights"]:
            sparse_vectors.append(
                {int(key): float(value) for key, value in lexical_weights.items()}
            )

        return EmbeddedBatch(
            dense=output["dense_vecs"].tolist(),
            sparse=sparse_vectors,
        )

You probably do not need ColBERT-style vectors in version 1. Add them only if evaluation shows cross-encoder reranking is too slow or insufficient.

Qdrant collection
Python
실행됨
from qdrant_client import QdrantClient, models


DENSE_SIZE = 1024


def create_transcript_collection(
    client: QdrantClient,
    collection_name: str = "transcript_chunks",
) -> None:
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )

    for field in [
        "video_id",
        "channel_id",
        "category",
        "language",
        "source_type",
    ]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="views",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

Metadata constraints belong in payload filters rather than being encoded into the text. Qdrant explicitly supports applying conditions to payload fields during retrieval, which is the right mechanism for category, channel, language, and view-count constraints. 
Qdrant

Hybrid retrieval
Python
실행됨
from qdrant_client import QdrantClient, models


def to_sparse_vector(weights: dict[int, float]) -> models.SparseVector:
    return models.SparseVector(
        indices=list(weights.keys()),
        values=list(weights.values()),
    )


def build_filter(
    category: str | None = None,
    channel_ids: list[str] | None = None,
    language: str | None = None,
    minimum_views: int | None = None,
) -> models.Filter | None:
    must: list[models.FieldCondition] = []

    if category:
        must.append(
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category),
            )
        )

    if channel_ids:
        must.append(
            models.FieldCondition(
                key="channel_id",
                match=models.MatchAny(any=channel_ids),
            )
        )

    if language:
        must.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=language),
            )
        )

    if minimum_views is not None:
        must.append(
            models.FieldCondition(
                key="views",
                range=models.Range(gte=minimum_views),
            )
        )

    return models.Filter(must=must) if must else None


def hybrid_search(
    client: QdrantClient,
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    query_filter: models.Filter | None,
    limit: int = 20,
):
    return client.query_points(
        collection_name="transcript_chunks",
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                filter=query_filter,
                limit=60,
            ),
            models.Prefetch(
                query=to_sparse_vector(sparse_vector),
                using="sparse",
                filter=query_filter,
                limit=60,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=limit,
        with_payload=True,
    )
7. Retrieval flow
User query
   ↓
Language detection
   ↓
Intent classification
   ├─ direct-source question
   ├─ practical advice
   ├─ comparison/conflict
   ├─ creator-specific
   ├─ example request
   └─ high-risk emotional/safety question
   ↓
Query expansion
   ├─ original Korean
   ├─ optional English translation
   ├─ synonyms
   └─ normalized dating concepts
   ↓
Parallel retrieval
   ├─ transcript dense search
   ├─ transcript sparse search
   └─ OKF claim/concept search
   ↓
RRF fusion
   ↓
Cross-encoder reranking
   ↓
Source diversification
   ↓
Conflict grouping
   ↓
Context expansion
   ↓
Grounded answer generation
   ↓
Citation verification
Korean and English query expansion

Do not translate every Korean query to English and search only the translation. Search both:

Python
실행됨
class SearchQuery(BaseModel):
    original: str
    translated: str | None = None
    normalized_topics: list[str] = []
    keywords: list[str] = []

For a query such as:

잠수 이별 후에 다시 연락해야 할까?

Generate variants such as:

잠수 이별 후 다시 연락
고스팅 후 연락
ghosting reconnect
contact after being ghosted

Fuse the results, but deduplicate by chunk_id.

8. Diversification and duplicate control

This is essential for your corpus.

After retrieval:

Limit each video to at most two chunks.

Limit each creator to three or four chunks before final reranking.

Cluster near-duplicate chunks.

Prefer one representative per semantic cluster.

Preserve opposing clusters even when their raw scores are lower.

A practical final context might contain:

3–5 concept claims;

6–10 transcript chunks;

at least 3 creators where available;

no more than 2 chunks from one video;

explicit opposing evidence when disagreement exists.

Python
실행됨
def diversify(
    candidates: list[dict],
    final_limit: int = 10,
    max_per_video: int = 2,
    max_per_channel: int = 3,
) -> list[dict]:
    selected: list[dict] = []
    video_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}

    for item in candidates:
        payload = item["payload"]
        video_id = payload["video_id"]
        channel_id = payload["channel_id"]

        if video_counts.get(video_id, 0) >= max_per_video:
            continue
        if channel_counts.get(channel_id, 0) >= max_per_channel:
            continue

        selected.append(item)
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        channel_counts[channel_id] = channel_counts.get(channel_id, 0) + 1

        if len(selected) >= final_limit:
            break

    return selected
9. Handling conflicting advice

Do not ask the LLM to silently select a winner.

Research on conflicting-source RAG indicates that models frequently handle disagreement poorly, while explicitly prompting them to identify and reason about conflicts improves their behavior. 
arXiv

Use a structured conflict pipeline.

Conflict types
1. Direct contradiction
   "Contact them immediately" vs "Never contact them"

2. Conditional difference
   Contact them if the breakup was ambiguous,
   but not if boundaries were clearly stated.

3. Goal difference
   Advice for reconciliation vs advice for emotional recovery.

4. Population difference
   Advice for avoidant partners vs generally secure partners.

5. Time-horizon difference
   What helps this week vs what supports a healthy relationship long term.

6. Value difference
   Strategic dating advice vs authenticity-oriented advice.

7. Evidence-quality difference
   Personal anecdote vs therapist explanation vs survey/research reference.
Conflict representation
YAML
conflict_id: conflict_no_contact_001
topic: no-contact-after-breakup
type: conditional_difference

positions:
  - position_id: immediate_contact
    summary: Clarify the relationship once before withdrawing.
    conditions:
      - breakup was ambiguous
      - no explicit request for no contact
    supporting_claims:
      - claim_102
      - claim_184

  - position_id: strict_no_contact
    summary: Stop contact to regain emotional stability.
    conditions:
      - repeated rejection
      - obsessive checking
      - explicit boundary from former partner
    supporting_claims:
      - claim_221
      - claim_302

resolution:
  type: condition_matrix
  summary: The appropriate action depends on boundary clarity and the user's goal.
Answer behavior

The generated answer should say:

Several creators agree on X.

They differ on Y:
- Position A recommends ...
- Position B recommends ...

The difference appears to depend on ...
Given your described situation, A is more applicable because ...

This is advice, not a certainty about the other person's psychology.

Do not use views as a truth score. Views can be used for:

tie-breaking;

source discovery;

optional popularity filters;

displaying “widely viewed perspective.”

Views should not determine which advice is safest or most accurate.

10. Building the OKF layer

Use OKF as a Git-reviewable knowledge product, not as a dump of LLM summaries.

Concept file
Markdown
---
type: concept
title: No-contact after a breakup
description: Temporary or indefinite cessation of contact after a breakup.
resource: dating://concept/no-contact-after-breakup
tags:
  - breakup
  - emotional-recovery
  - reconciliation
timestamp: 2026-07-12T00:00:00Z

concept_id: no-contact-after-breakup
aliases:
  ko:
    - 노컨택
    - 연락 끊기
    - 이별 후 연락 금지
  en:
    - no contact rule
relationship_stage:
  - post_breakup
risk_level: medium
schema_version: "1.0"
---

# Definition

No-contact means intentionally avoiding direct and indirect contact with a
former partner for a defined or indefinite period.

# Common goals

- Emotional stabilization
- Boundary protection
- Reducing compulsive checking
- Creating space before reassessment

# Important distinctions

No-contact used for emotional recovery is different from silence used to
manipulate a former partner into returning.

# Claims

- [No-contact can reduce repeated emotional triggering](../claims/claim-001.md)
- [One clarification message may be appropriate after an ambiguous breakup](../claims/claim-002.md)
- [No-contact should not be framed as a guaranteed reconciliation tactic](../claims/claim-003.md)

# Known disagreements

See: [Whether to send a final message](../conflicts/no-contact-final-message.md)
Claim file
Markdown
---
type: claim
title: No-contact may reduce repeated emotional triggering
description: A conditional claim about using no-contact for emotional recovery.
resource: dating://claim/no-contact-reduce-triggering
tags:
  - breakup
  - no-contact
timestamp: 2026-07-12T00:00:00Z

claim_id: claim-001
concept_id: no-contact-after-breakup
stance: supports
confidence: 0.82
curation_status: human_reviewed
evidence_count: 7
creator_count: 4
schema_version: "1.0"
---

# Statement

Reducing exposure to messages, social profiles, and repeated conversations may
help some people interrupt cycles of emotional triggering after a breakup.

# Applies when

- The user repeatedly checks the former partner's profile.
- Contact repeatedly restarts distress.
- The former partner has asked for space.

# Does not imply

- The former partner will return.
- No-contact is appropriate in every shared-parenting or work situation.

# Evidence

- `video_123:chunk_08`, 04:12–05:06
- `video_331:chunk_14`, 10:21–11:03
- `video_422:chunk_04`, 02:09–03:01
Build process
Phase A: automated extraction

For every chunk, extract candidate records:

JSON
{
  "concepts": ["no-contact-after-breakup"],
  "claims": [
    {
      "statement": "No-contact can help stop emotional re-triggering.",
      "stance": "supports",
      "applies_when": ["contact repeatedly causes distress"],
      "evidence_span": "..."
    }
  ]
}
Phase B: canonicalization

Cluster paraphrases such as:

“연락을 끊어야 감정이 가라앉는다”

“거리를 둬야 감정 회복이 시작된다”

“stop checking their messages to regulate emotions”

into one canonical claim.

Phase C: contradiction discovery

For each concept:

embed claims;

retrieve similar claims;

run natural-language-inference or structured LLM classification;

label pairs as:

agreement;

contradiction;

conditional;

unrelated;

send uncertain cases for review.

Phase D: human review

Human review should prioritize:

top 30–50 concepts by query demand;

safety-sensitive topics;

concepts with high disagreement;

concepts used in many answers.

Do not manually curate the entire 481-video corpus before launch. That would turn the project into a library catalog with a chatbot attached.

11. Query router
Python
실행됨
from typing import Literal
from pydantic import BaseModel


class QueryPlan(BaseModel):
    intent: Literal[
        "specific_example",
        "general_advice",
        "creator_lookup",
        "compare_viewpoints",
        "definition",
        "high_risk",
    ]
    topics: list[str]
    use_transcripts: bool
    use_okf: bool
    require_conflict_search: bool
    require_source_diversity: bool
    category_filter: str | None = None
    channel_filters: list[str] = []
    minimum_views: int | None = None


def default_plan(intent: str) -> QueryPlan:
    if intent == "specific_example":
        return QueryPlan(
            intent="specific_example",
            topics=[],
            use_transcripts=True,
            use_okf=False,
            require_conflict_search=False,
            require_source_diversity=True,
        )

    if intent == "compare_viewpoints":
        return QueryPlan(
            intent="compare_viewpoints",
            topics=[],
            use_transcripts=True,
            use_okf=True,
            require_conflict_search=True,
            require_source_diversity=True,
        )

    return QueryPlan(
        intent="general_advice",
        topics=[],
        use_transcripts=True,
        use_okf=True,
        require_conflict_search=True,
        require_source_diversity=True,
    )

Router output should be validated JSON. When classification fails, default to retrieving both layers.

12. Generation prompt
SYSTEM

You are a grounded dating-advice assistant.

Your job is to help the user reason about their situation, not to predict
another person's hidden motives.

Use only the supplied knowledge claims and transcript evidence for source-based
claims.

Rules:

1. Distinguish:
   - source-supported information,
   - your synthesis,
   - uncertainty,
   - questions that cannot be answered from the evidence.

2. Do not diagnose attachment style, personality disorder, narcissism, or
   another mental-health condition based on limited behavior.

3. When sources conflict:
   - identify the disagreement,
   - explain the conditions behind each position,
   - do not silently merge contradictory advice,
   - do not treat majority count as proof.

4. Give practical next steps appropriate to the user's goal and constraints.

5. Cite transcript evidence as:
   [Channel, Video title, MM:SS–MM:SS]

6. Every citation must correspond to a supplied source.
   Never invent videos, channels, quotes, or timestamps.

7. Prefer paraphrase. Use short quotations only when wording is important.

8. If the situation involves threats, coercion, stalking, violence,
   self-harm, or immediate danger, prioritize safety rather than dating
   strategy.

OUTPUT STRUCTURE

- Brief interpretation
- What the sources broadly agree on
- Where advice differs
- Recommended next steps
- Caveats
- Sources

Pass claims and transcript evidence separately:

<curated_knowledge>
...
</curated_knowledge>

<transcript_evidence>
[S1]
channel: ...
video: ...
time: 04:12–05:06
text: ...
</transcript_evidence>
13. Citation validation

Never trust the first generated citation string.

After generation:

parse all [S1], [S2] references;

confirm each source ID exists;

confirm cited timestamps match that chunk;

reject citations not used in the answer;

optionally run an entailment check between the sentence and evidence;

regenerate unsupported sentences or remove them.

Python
실행됨
import re


SOURCE_PATTERN = re.compile(r"\[S(\d+)\]")


def validate_source_ids(answer: str, allowed_ids: set[str]) -> list[str]:
    referenced = {
        f"S{match}"
        for match in SOURCE_PATTERN.findall(answer)
    }
    return sorted(referenced - allowed_ids)

For clickable YouTube citations:

Python
실행됨
def timestamp_url(video_id: str, start_seconds: float) -> str:
    seconds = max(0, int(start_seconds))
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
14. End-to-end service skeleton
Python
실행됨
class DatingAdviceService:
    def __init__(
        self,
        query_analyzer,
        transcript_retriever,
        concept_retriever,
        reranker,
        conflict_detector,
        generator,
        citation_validator,
    ) -> None:
        self.query_analyzer = query_analyzer
        self.transcript_retriever = transcript_retriever
        self.concept_retriever = concept_retriever
        self.reranker = reranker
        self.conflict_detector = conflict_detector
        self.generator = generator
        self.citation_validator = citation_validator

    async def answer(self, question: str, user_filters: dict) -> dict:
        plan = await self.query_analyzer.analyze(
            question=question,
            filters=user_filters,
        )

        transcript_candidates = []
        concept_candidates = []

        if plan.use_transcripts:
            transcript_candidates = await self.transcript_retriever.search(
                question=question,
                plan=plan,
            )

        if plan.use_okf:
            concept_candidates = await self.concept_retriever.search(
                question=question,
                topics=plan.topics,
            )

        transcript_evidence = await self.reranker.rerank_and_diversify(
            query=question,
            candidates=transcript_candidates,
        )

        conflicts = []
        if plan.require_conflict_search:
            conflicts = await self.conflict_detector.detect(
                claims=concept_candidates,
                transcript_chunks=transcript_evidence,
            )

        answer = await self.generator.generate(
            question=question,
            plan=plan,
            claims=concept_candidates,
            evidence=transcript_evidence,
            conflicts=conflicts,
        )

        validation = await self.citation_validator.validate(
            answer=answer,
            evidence=transcript_evidence,
        )

        if not validation.valid:
            answer = await self.generator.repair(
                answer=answer,
                validation_errors=validation.errors,
                evidence=transcript_evidence,
            )

        return {
            "answer": answer,
            "sources": transcript_evidence,
            "conflicts": conflicts,
            "plan": plan,
        }
15. Evaluation

Create evaluation sets before tuning chunk size or fusion weights.

Retrieval metrics
Metric	Purpose
Recall@5 / @10 / @20	Was at least one useful passage found?
MRR	How early was the first useful passage?
nDCG@10	Were highly relevant passages ranked first?
Creator diversity@10	Did results represent independent creators?
Duplicate rate@10	How many results repeat the same point?
Conflict coverage	Were opposing positions retrieved?
Timestamp precision	Does the cited interval contain the claim?
Generation metrics
Metric	Purpose
Citation correctness	Does the citation support the sentence?
Citation completeness	Are major source-based claims cited?
Groundedness	Is the answer entailed by retrieved context?
Conflict transparency	Does the answer disclose disagreement?
Conditionality	Does it explain when each recommendation applies?
Actionability	Are next steps concrete?
Safety compliance	Does it avoid diagnosis/manipulation/dangerous advice?
Source balance	Is one creator dominating synthesis?
Minimum benchmark

Start with approximately:

100 normal questions;

30 creator-specific questions;

30 Korean-English cross-lingual questions;

40 conflict questions;

20 citation/timestamp questions;

20 safety-sensitive questions.

That gives roughly 240 evaluation cases. Even a first manually reviewed set of 80–100 cases is much better than tuning based on a few attractive demos.

16. Implementation plan

Assuming one experienced Python engineer working full-time:

Stage	Deliverable	Estimate
1	Normalize metadata and timestamp transcripts	1–2 days
2	Implement semantic timestamp chunking	2–3 days
3	BGE-M3 embeddings and Qdrant indexing	1–2 days
4	Dense+sparse hybrid retrieval and filters	2–3 days
5	Reranking, deduplication, creator diversity	2–3 days
6	Grounded answer generation and citations	2–3 days
7	Query router and context expansion	2 days
8	Initial evaluation set and retrieval tuning	3–5 days
9	Automated claim extraction	3–4 days
10	OKF schema, loader, validator, Git workflow	2–3 days
11	Conflict detection and comparison answers	3–5 days
12	Safety, tracing, caching, deployment	3–5 days
Practical total

Transcript-RAG MVP: approximately 10–15 engineering days

Hybrid RAG + initial OKF: approximately 20–30 engineering days

Production hardening and meaningful evaluation: approximately 30–40 engineering days

The schedule depends more on transcript quality and evaluation work than on vector indexing. At 6.7 MB, embedding and storage are the tiny gears. Knowledge quality is the heavy flywheel.

17. Recommended rollout order
Version 1: evidence RAG

Build first:

timestamp-aware chunks;

BGE-M3 dense+sparse retrieval;

metadata filters;

reranking;

channel/video diversity;

adjacent context expansion;

citation validation.

Do not begin by curating hundreds of OKF files.

Version 2: lightweight OKF

Curate the top 30–50 recurring concepts:

ghosting;

no-contact;

first messages;

rejection;

ambiguous relationships;

avoidant behavior;

jealousy;

boundaries;

breakup recovery;

reconciliation;

conversation pacing;

MBTI limitations.

Version 3: conflict-aware advice

Add:

atomic claims;

stance labeling;

conditional applicability;

creator-independent support counts;

conflict records;

answer templates for disagreement.

Version 4: feedback-driven expansion

Use anonymized query analytics to find:

unanswered topics;

poor retrieval terms;

missing concept aliases;

frequently contested claims;

citation failures.

Expand OKF based on actual demand, not merely the order in which videos were ingested.

Final architecture decision

Choose RAG + OKF, but make RAG the evidentiary source of truth.

The production pattern should be:

Hybrid transcript retrieval
+ curated claim/concept retrieval
+ creator-level diversification
+ conflict classification
+ timestamp-grounded generation
+ citation verification

The most important design rule is:

Never allow a curated summary to become detached from its timestamped transcript evidence.

That gives users both sides of the product they need: a compass made from repeated insights, and a trail of breadcrumbs back to the exact videos.
