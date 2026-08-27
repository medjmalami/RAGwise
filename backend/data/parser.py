import argparse
import inspect
import re
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Parse HTML files to Docling Document and Markdown with extracted images."
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

    try:
        from docling.datamodel.backend_options import HTMLBackendOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, FormatOption
        from docling_core.types.doc import ImageRefMode
    except ImportError as e:
        print(f"Error importing Docling components: {e}")
        sys.exit(1)

    # 1. Setup HTML options
    html_options_kwargs = {"fetch_images": True}
    if (
        "max_image_data_base64_bytes"
        in inspect.signature(HTMLBackendOptions).parameters
    ):
        html_options_kwargs["max_image_data_base64_bytes"] = (
            100 * 1024 * 1024
        )  # 100MB limit

    # 2. Setup DocumentConverter
    doc_converter = None
    try:
        html_options = HTMLBackendOptions(**html_options_kwargs)
        try:
            from docling.backend.html_backend import HTMLDocumentBackend
            from docling.pipeline.simple_pipeline import SimplePipeline

            format_options = {
                InputFormat.HTML: FormatOption(
                    pipeline_cls=SimplePipeline,
                    backend=HTMLDocumentBackend,
                    backend_options=html_options,
                )
            }
            doc_converter = DocumentConverter(format_options=format_options)
            print(
                f"Configured DocumentConverter with HTMLBackendOptions ({html_options_kwargs})"
            )
        except Exception:
            try:
                format_options = {
                    InputFormat.HTML: FormatOption(backend_options=html_options)
                }
                doc_converter = DocumentConverter(format_options=format_options)
            except Exception:
                doc_converter = DocumentConverter()
    except Exception:
        doc_converter = DocumentConverter()

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
            # This extracts images to disk and assigns URIs
            save_md_kwargs = {"image_mode": ImageRefMode.REFERENCED}
            sig_md_params = inspect.signature(doc.save_as_markdown).parameters
            for p in ["image_dir", "artifacts_dir", "artifacts_path"]:
                if p in sig_md_params:
                    save_md_kwargs[p] = artifacts_dir
                    break
            doc.save_as_markdown(md_path, **save_md_kwargs)

            # 2. SAVE JSON
            # We MUST pass image_mode=ImageRefMode.REFERENCED so it extracts the base64 to disk instead of crashing
            save_json_kwargs = {"image_mode": ImageRefMode.REFERENCED}
            sig_json_params = inspect.signature(doc.save_as_json).parameters
            for p in ["image_dir", "artifacts_dir", "artifacts_path"]:
                if p in sig_json_params:
                    save_json_kwargs[p] = artifacts_dir
                    break
            doc.save_as_json(json_path, **save_json_kwargs)

            # 3. CLEANUP ARTIFACTS FOLDERS
            # Docling often ignores the path and creates a default <paper_id>_artifacts folder anyway
            default_artifacts_dir = paper_output_dir / f"{paper_id}_artifacts"
            if default_artifacts_dir.exists() and default_artifacts_dir.is_dir():
                for img_file in default_artifacts_dir.iterdir():
                    if img_file.is_file():
                        # Move to our clean artifacts dir
                        shutil.move(str(img_file), str(artifacts_dir / img_file.name))
                shutil.rmtree(default_artifacts_dir)
                print(f"  Reorganized images into {artifacts_dir}")

            # 4. FIX JSON PATHS
            # Read the JSON, replace the default folder name with 'artifacts/', and save
            if json_path.exists():
                json_content = json_path.read_text()
                json_content = json_content.replace(
                    f"{paper_id}_artifacts/", "artifacts/"
                )
                # Clear any remaining base64 payloads to keep the JSON lightweight for chunking
                json_content = re.sub(
                    r'"data":\s*"[A-Za-z0-9+/=]+",', '"data": null,', json_content
                )
                json_path.write_text(json_content)
                print(f"  Cleaned up JSON image paths and stripped base64 data.")

            # 5. FIX MARKDOWN PATHS
            if md_path.exists():
                md_content = md_path.read_text()
                md_content = re.sub(
                    r"!\[(.*?)\]\([^)]*?artifacts/([^)]+)\)",
                    r"![\1](artifacts/\2)",
                    md_content,
                )
                md_content = md_content.replace(f"{paper_id}_artifacts/", "artifacts/")
                md_path.write_text(md_content)
                print(f"  Cleaned up Markdown image paths.")

        except Exception as e:
            print(f"  Error processing {paper_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
