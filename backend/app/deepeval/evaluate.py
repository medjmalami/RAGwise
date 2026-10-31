"""
Multi-turn RAG evaluation with DeepEval + Ollama judge.

Usage:
    python eval_multiturn.py --limit 3     # smoke test on 3 conversations
    python eval_multiturn.py               # full run (resumes if interrupted)
"""

import argparse
import json
import os
from typing import Optional

import pandas as pd
from deepeval.metrics import (
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from deepeval.models import OllamaModel
from deepeval.models.llms.ollama_model import retry_ollama
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from pydantic import BaseModel

CONVS_CSV = "simulated_conversations_graph.csv"
GOLDENS_CSV = "conversational_goldens.csv"
RESULTS_CSV = "multiturn_eval_results.csv"
JUDGE_MODEL = "gemma4:31b-cloud"  # <- set to the Ollama model you want as judge
OLLAMA_URL = "http://localhost:11434"


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that some Ollama (cloud) models add."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class FenceTolerantOllama(OllamaModel):
    """OllamaModel that survives judges which wrap JSON in markdown fences.

    The stock class calls schema.model_validate_json() on the raw reply, which
    raises on '```json ...```'. Here we strip the fences first; if validation
    still fails we return the raw text so DeepEval's own JSON extraction
    (trimAndLoadJson) gets a second chance instead of the whole call failing.
    """

    def _parse(self, content: str, schema: Optional[BaseModel]) -> str | BaseModel:
        if not schema:
            return content
        cleaned = _strip_fences(content)
        try:
            return schema.model_validate_json(cleaned)
        except Exception:
            return cleaned

    @retry_ollama
    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[list[Audio]] = None,
    ) -> tuple[str | BaseModel, float]:
        client = self.load_model()
        resp = client.chat(
            model=self.name,
            messages=self._prompt_messages(prompt, audio),
            format=schema.model_json_schema() if schema else None,
            options={"temperature": self.temperature, **self.generation_kwargs},
        )
        return self._parse(resp.message.content, schema), 0

    @retry_ollama
    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[list[Audio]] = None,
    ) -> tuple[str | BaseModel, float]:
        client = self.load_model(async_mode=True)
        resp = await client.chat(
            model=self.name,
            messages=self._prompt_messages(prompt, audio),
            format=schema.model_json_schema() if schema else None,
            options={"temperature": self.temperature, **self.generation_kwargs},
        )
        return self._parse(resp.message.content, schema), 0


def build_cases(limit: int | None = None) -> dict[object, ConversationalTestCase]:
    convs = pd.read_csv(CONVS_CSV)
    goldens = pd.read_csv(GOLDENS_CSV).set_index("golden_id")
    cases: dict[object, ConversationalTestCase] = {}
    for gid, g in convs.sort_values(["golden_id", "turn_index"]).groupby("golden_id"):
        turns: list[Turn] = [
            Turn(
                role="assistant",
                content=row["content"],
                retrieval_context=json.loads(row["retrieved_contexts"]),
            )
            if row["role"] == "assistant"
            else Turn(role="user", content=row["content"])
            for row in g.to_dict(orient="records")
        ]
        cases[gid] = ConversationalTestCase(
            turns=turns,
            scenario=goldens.loc[gid, "scenario"],
            expected_outcome=goldens.loc[gid, "expected_outcome"],
        )
        if limit and len(cases) >= limit:
            break
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    judge = FenceTolerantOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL, temperature=0)
    # async_mode=False: avoid parallel requests against a single Ollama instance
    metrics = [
        TurnRelevancyMetric(model=judge, include_reason=True),
        TurnFaithfulnessMetric(model=judge, include_reason=True),
        TurnContextualPrecisionMetric(model=judge, include_reason=True),
        TurnContextualRecallMetric(model=judge, include_reason=True),
    ]

    cases = build_cases(args.limit)

    # Resume support: skip (golden_id, metric) pairs already scored.
    done = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        ok = prev[prev.error.isna()]  # only successful rows count as done
        done = set(zip(ok.golden_id, ok.metric))
    write_header = not os.path.exists(RESULTS_CSV)

    for i, (gid, case) in enumerate(cases.items(), 1):
        for m in metrics:
            name = type(m).__name__
            if (gid, name) in done:
                continue
            try:
                m.measure(case)
                row = dict(
                    golden_id=gid,
                    metric=name,
                    score=m.score,
                    success=m.success,
                    reason=m.reason,
                    error="",
                )
            except Exception as e:  # keep going; inspect errors afterwards
                row = dict(
                    golden_id=gid,
                    metric=name,
                    score=None,
                    success=None,
                    reason="",
                    error=repr(e)[:500],
                )
            pd.DataFrame([row]).to_csv(
                RESULTS_CSV, mode="a", header=write_header, index=False
            )
            write_header = False
            print(f"[{i}/{len(cases)}] {gid} {name}: {row['score']} {row['error']}")

    res = pd.read_csv(RESULTS_CSV)
    print("\nMean score per metric:")
    print(res.groupby("metric").score.mean().round(3))
    print("\nErrors per metric:")
    print(res.assign(err=res.error.notna()).groupby("metric").err.sum())


if __name__ == "__main__":
    main()
