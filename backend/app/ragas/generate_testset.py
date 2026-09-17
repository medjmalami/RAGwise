import argparse
import logging
import os
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import Testset, TestsetGenerator
from ragas.testset.graph import KnowledgeGraph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAGAS testset from a saved knowledge graph."
    )
    parser.add_argument(
        "--kg-path",
        default="knowledge_graph.json",
        help="Path to the knowledge graph JSON file.",
    )
    parser.add_argument(
        "--output-csv",
        default="testset.csv",
        help="Path to write the generated testset CSV.",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=10,
        help="Number of test samples to generate.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:31b-cloud",
        help="Ollama model tag used as the generator LLM (e.g. a '*-cloud' tag "
        "to run on Ollama's cloud instead of locally).",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL of the local Ollama daemon (it forwards '-cloud' tags "
        "to Ollama's cloud service automatically).",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="HuggingFace embedding model name.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, help="RunConfig max_workers."
    )
    parser.add_argument(
        "--max-wait", type=int, default=90, help="RunConfig max_wait (seconds)."
    )
    parser.add_argument(
        "--max-retries", type=int, default=8, help="RunConfig max_retries."
    )
    return parser.parse_args()


def load_knowledge_graph(kg_path: str) -> KnowledgeGraph:
    if not os.path.exists(kg_path):
        logger.error("Knowledge graph file not found: %s", kg_path)
        sys.exit(1)
    logger.info("Loading knowledge graph from %s ...", kg_path)
    kg = KnowledgeGraph.load(kg_path)
    logger.info(
        "Loaded knowledge graph with %d nodes and %d relationships.",
        len(kg.nodes),
        len(kg.relationships),
    )
    return kg


def main() -> None:
    args = parse_args()

    kg = load_knowledge_graph(args.kg_path)

    logger.info(
        "Setting up generator LLM (%s via %s) and embeddings (%s) ...",
        args.model,
        args.ollama_base_url,
        args.embedding_model,
    )
    generator_llm = LangchainLLMWrapper(
        ChatOllama(model=args.model, base_url=args.ollama_base_url)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    run_config = RunConfig(
        max_workers=args.max_workers,
        max_wait=args.max_wait,
        max_retries=args.max_retries,
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )

    logger.info("Generating test set (size=%d) ...", args.testset_size)
    dataset = generator.generate(
        testset_size=args.testset_size,
        run_config=run_config,
        raise_exceptions=False,
    )
    assert isinstance(dataset, Testset)

    df = dataset.to_pandas()
    df.to_csv(args.output_csv, index=False)
    logger.info(
        "Test set generated and saved to %s (%d rows).", args.output_csv, len(df)
    )


if __name__ == "__main__":
    main()
