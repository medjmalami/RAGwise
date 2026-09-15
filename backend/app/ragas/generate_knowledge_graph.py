import os
import random

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import Testset, TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms

from app.config import settings

# --- CONFIGURATION ---
os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None
COLLECTION_NAME = "ragwise_arxiv"
FETCH_POOL = 1000
SAMPLE_SIZE = 200
TESTSET_SIZE = 10
OUTPUT_CSV = "ragas_test_set.csv"
KG_CACHE_PATH = "knowledge_graph.json"
GEMINI_MODEL = "gemma-4-31b-it"

RUN_CONFIG = RunConfig(max_workers=3, max_wait=90, max_retries=8)


def rough_token_estimate(text: str) -> int:
    """Very rough ~4-chars-per-token estimate, only used for the sanity
    check below -- NOT what ragas uses internally for its own bucketing."""
    return max(1, len(text) // 4)


# --- 1. FETCH RANDOM BATCH FROM QDRANT ---
print(f"Fetching a pool of {FETCH_POOL} chunks from Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

records, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=FETCH_POOL,
    with_payload=True,
    with_vectors=False,
)

if len(records) > SAMPLE_SIZE:
    records = random.sample(records, SAMPLE_SIZE)
print(f"Selected a random batch of {len(records)} chunks.")

docs = []
for record in records:
    if record.payload is None:
        continue
    payload = record.payload
    text_content = payload.get("text")
    if not text_content:
        continue
    metadata = {
        "filename": payload.get("paper_id"),
        "chunk_id": payload.get("chunk_id"),
        "source": payload.get("source_json"),
    }
    docs.append(Document(page_content=text_content, metadata=metadata))

print(f"Built {len(docs)} LangChain documents.")
if not docs:
    raise SystemExit(
        "No documents with a 'text' payload field were found -- check COLLECTION_NAME/payload schema."
    )

short_ratio = sum(rough_token_estimate(d.page_content) <= 100 for d in docs) / len(docs)
if short_ratio > 0.75:
    print(
        "WARNING: most sampled chunks look very short (~<=100 tokens estimated). "
        "ragas' default_transforms may raise a 'Documents appears to be too short' "
        "error, or generate shallow single-hop questions. See the note above the "
        "document-building loop."
    )

# --- 3. BUILD KNOWLEDGE GRAPH ---
print("Building Knowledge Graph from documents...")
kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

# --- 4. APPLY TRANSFORMS ---
print("Initializing Google LLM and BGE-M3 embedder...")
generator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=GEMINI_MODEL))
generator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
    )
)

print("Applying transforms to Knowledge Graph (this may take a while)...")
transforms = default_transforms(
    documents=docs,
    llm=generator_llm,
    embedding_model=generator_embeddings,
)
apply_transforms(kg, transforms, run_config=RUN_CONFIG)

kg.save(KG_CACHE_PATH)
print(
    f"Saved enriched knowledge graph to {KG_CACHE_PATH} (reload with KnowledgeGraph.load(...) if generation fails partway)."
)

# --- 5. GENERATE TEST SET ---
print("Generating test set...")
generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg
)
dataset = generator.generate(
    testset_size=TESTSET_SIZE,
    run_config=RUN_CONFIG,
    raise_exceptions=False,
)
assert isinstance(dataset, Testset)

# --- 6. SAVE OUTPUT ---
df = dataset.to_pandas()
df.to_csv(OUTPUT_CSV, index=False)
print(f"Test set generated and saved to {OUTPUT_CSV}")
