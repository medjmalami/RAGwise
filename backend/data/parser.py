"""
Stage 1: Parse arXiv HTML papers into Docling Documents (JSON) and Markdown,
extracting images and generating VLM-based picture descriptions via the
Gemini API (Gemma 4).
"""

import argparse
import inspect
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def build_picture_description_options(api_key: str, model: str):
    """Configure remote picture-description enrichment against Gemini's
    OpenAI-compatible chat-completions endpoint, serving a Gemma 4 model.
    """
    from docling.datamodel.pipeline_options import PictureDescriptionApiOptions

    return PictureDescriptionApiOptions(
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        params=dict(
            model=model,
            max_tokens=1024,  # Increased from 400 so the model can finish thinking and output the final answer
            temperature=0.1,
        ),
        prompt=(
            "You are an expert AI analyzing figures from scientific papers. "
            "Describe this figure in 2-3 sentences. "
            "If it is a plot or chart, name the axes/variables and the key trend. "
            "If it is a diagram, describe what it depicts. "
            "Be concise and factual. "
            "IMPORTANT: Output ONLY the final description. "
            "DO NOT include any reasoning, chain-of-thought, or <thought> tags."
        ),
        timeout=90,
    )


def build_convert_pipeline_options(api_key: str, model: str):
    """ConvertPipelineOptions is the enrichment-capable options class used by
    SimplePipeline (and its descendants), which is what the HTML backend runs
    on. This is distinct from PdfPipelineOptions.
    """
    from docling.datamodel.pipeline_options import ConvertPipelineOptions

    return ConvertPipelineOptions(
        do_picture_description=True,
        picture_description_options=build_picture_description_options(api_key, model),
        enable_remote_services=True,  # required whenever the target is a remote API
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parse HTML files to Docling Document and Markdown with "
        "extracted images and Gemini/Gemma picture descriptions."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing HTML files.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for parsed files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of HTML files to process.",
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default="gemma-4-26b-a4b-it",
        choices=["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
        help="Gemma 4 model to use for picture description, as exposed by "
        "the Gemini API (default: gemma-4-26b-a4b-it, the cheaper/faster MoE "
        "model; gemma-4-31b-it is the stronger dense model).",
    )
    parser.add_argument(
        "--no-picture-description",
        action="store_true",
        help="Disable picture-description enrichment (parse only, no API calls).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(input_dir.glob("*.html"))
    if args.limit is not None:
        html_files = html_files[: args.limit]

    print(f"Found {len(html_files)} HTML files to process.")

    gemini_api_key = None
    if not args.no_picture_description:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print(
                "Error: GEMINI_API_KEY is not set. Export it, or pass "
                "--no-picture-description to skip image descriptions."
            )
            sys.exit(1)

    try:
        from docling.backend.html_backend import HTMLDocumentBackend
        from docling.datamodel.backend_options import HTMLBackendOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.pipeline.simple_pipeline import SimplePipeline
        from docling_core.types.doc import ImageRefMode
    except ImportError as e:
        print(f"Error importing Docling components: {e}")
        sys.exit(1)

    # 1. Setup HTML backend options
    html_options_kwargs = {"fetch_images": True}
    if (
        "max_image_data_base64_bytes"
        in inspect.signature(HTMLBackendOptions).parameters
    ):
        html_options_kwargs["max_image_data_base64_bytes"] = (
            100 * 1024 * 1024
        )  # 100MB limit
    html_options = HTMLBackendOptions(**html_options_kwargs)

    # 2. Setup enrichment (picture description) pipeline options, if enabled.
    convert_pipeline_options = None
    if gemini_api_key:
        convert_pipeline_options = build_convert_pipeline_options(
            api_key=gemini_api_key, model=args.gemma_model
        )
        print(f"Picture description enabled via Gemini API ({args.gemma_model}).")
    else:
        print("Picture description disabled (--no-picture-description).")

    # 3. Setup DocumentConverter
    format_options = {
        InputFormat.HTML: FormatOption(
            pipeline_cls=SimplePipeline,
            backend=HTMLDocumentBackend,
            backend_options=html_options,
            pipeline_options=convert_pipeline_options,
        )
    }
    doc_converter = DocumentConverter(format_options=format_options)
    print(
        f"Configured DocumentConverter with HTMLBackendOptions ({html_options_kwargs})"
    )

    failures = []

    for html_path in html_files:
        paper_id = html_path.stem
        paper_output_dir = output_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        artifacts_dir = paper_output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {paper_id}...")

        try:
            conv_res = doc_converter.convert(html_path)
            doc = conv_res.document

            md_path = paper_output_dir / f"{paper_id}.md"
            json_path = paper_output_dir / f"{paper_id}.docling.json"

            # 1. SAVE MARKDOWN
            save_md_kwargs = {"image_mode": ImageRefMode.REFERENCED}
            sig_md_params = inspect.signature(doc.save_as_markdown).parameters
            for p in ["image_dir", "artifacts_dir", "artifacts_path"]:
                if p in sig_md_params:
                    save_md_kwargs[p] = artifacts_dir
                    break
            doc.save_as_markdown(md_path, **save_md_kwargs)

            # 2. SAVE JSON
            save_json_kwargs = {"image_mode": ImageRefMode.REFERENCED}
            sig_json_params = inspect.signature(doc.save_as_json).parameters
            for p in ["image_dir", "artifacts_dir", "artifacts_path"]:
                if p in sig_json_params:
                    save_json_kwargs[p] = artifacts_dir
                    break
            doc.save_as_json(json_path, **save_json_kwargs)

            # 3. CLEANUP ARTIFACTS FOLDERS
            default_artifacts_dir = paper_output_dir / f"{paper_id}_artifacts"
            if default_artifacts_dir.exists() and default_artifacts_dir.is_dir():
                for img_file in default_artifacts_dir.iterdir():
                    if img_file.is_file():
                        shutil.move(str(img_file), str(artifacts_dir / img_file.name))
                shutil.rmtree(default_artifacts_dir)
                print(f"  Reorganized images into {artifacts_dir}")

            # 4. FIX JSON PATHS & SANITIZE DESCRIPTIONS
            if json_path.exists():
                json_content = json_path.read_text(encoding="utf-8")
                json_content = json_content.replace(
                    f"{paper_id}_artifacts/", "artifacts/"
                )
                # Clear any remaining base64 payloads
                json_content = re.sub(
                    r'"data":\s*"[A-Za-z0-9+/=]+",', '"data": null,', json_content
                )
                # Strip out <thought> blocks (both closed and truncated/unclosed)
                # in case the VLM ignored the prompt constraints
                json_content = re.sub(
                    r"<thought>.*?(?:</thought>|$)", "", json_content, flags=re.DOTALL
                )
                # Clean up any stray newline artifacts left behind by the regex
                json_content = re.sub(r"\n\s*\n\s*", "\n", json_content)

                json_path.write_text(json_content, encoding="utf-8")
                print(
                    f"  Cleaned up JSON paths, stripped base64, and sanitized descriptions."
                )

            # 5. FIX MARKDOWN PATHS & SANITIZE DESCRIPTIONS
            if md_path.exists():
                md_content = md_path.read_text(encoding="utf-8")
                md_content = re.sub(
                    r"!\[(.*?)\]\([^)]*?artifacts/([^)]+)\)",
                    r"![\1](artifacts/\2)",
                    md_content,
                )
                md_content = md_content.replace(f"{paper_id}_artifacts/", "artifacts/")
                # Strip out <thought> blocks from markdown as well
                md_content = re.sub(
                    r"<thought>.*?(?:</thought>|$)", "", md_content, flags=re.DOTALL
                )
                md_path.write_text(md_content, encoding="utf-8")
                print(f"  Cleaned up Markdown image paths and sanitized descriptions.")

        except Exception as e:
            print(f"  Error processing {paper_id}: {e}")
            import traceback

            traceback.print_exc()
            failures.append(paper_id)

    succeeded = len(html_files) - len(failures)
    print(f"\nDone. {succeeded}/{len(html_files)} succeeded.")
    if failures:
        print(f"Failed ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
