import base64

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings


def create_tracer_provider() -> TracerProvider:
    auth = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()

    provider = TracerProvider()

    exporter = OTLPSpanExporter(
        endpoint=f"{settings.langfuse_base_url}/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {auth}",
            "x-langfuse-ingestion-version": "4",
        },
    )

    provider.add_span_processor(BatchSpanProcessor(exporter))

    return provider
