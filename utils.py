"""Utility helpers for the longevity agent PoC."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("longevity_agent")


# ================= CONFIGURATION =================


@dataclass
class SystemConfig:
    """Lightweight configuration container for the agent."""

    world_model_path: str = "ontology.yaml"
    log_dir: str = "logs"
    output_dir: str = "outputs"

    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    max_budget_usd: int = 200_000
    max_duration_months: int = 12

    log_level: str = "INFO"
    log_to_file: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(self.__dict__, handle, default_flow_style=False)


DEFAULT_CONFIG = SystemConfig()


# ================= LOGGING =================


def setup_logging(config: SystemConfig = DEFAULT_CONFIG) -> None:
    """Configure a minimal logging sink for the demo."""

    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)

    handlers = [logging.StreamHandler()]
    if config.log_to_file:
        file_handler = logging.FileHandler(log_dir / "agent.log")
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        handlers=handlers,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ================= RESULT STRUCTURES =================


@dataclass
class AnalysisResult:
    """Structured result produced by the agent pipeline."""

    query: str
    timestamp: datetime
    intent: str
    interventions: List[str]
    translation_analyses: List[Dict[str, Any]] = field(default_factory=list)
    experiment_plans: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    full_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "interventions": self.interventions,
            "translation_analyses": self.translation_analyses,
            "experiment_plans": self.experiment_plans,
            "processing_time": self.processing_time,
            "errors": self.errors,
            "warnings": self.warnings,
            "full_report": self.full_report,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "AnalysisResult":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def get_summary(self) -> str:
        summary = [
            f"Query: {self.query}",
            f"Timestamp: {self.timestamp:%Y-%m-%d %H:%M:%S}",
            f"Intent: {self.intent}",
            f"Interventions: {', '.join(self.interventions)}",
            f"Processing time: {self.processing_time:.2f}s",
        ]

        if self.translation_analyses:
            summary.append("\nTranslation analyses:")
            for analysis in self.translation_analyses:
                summary.append(
                    f"  - {analysis['intervention_id']}: HRS={analysis['human_relevance_score']:.1f}, "
                    f"Confidence={analysis['confidence']}"
                )

        if self.errors:
            summary.append(f"\nErrors: {len(self.errors)}")

        return "\n".join(summary)


# ================= RESULT HELPERS =================


class ResultManager:
    """Persist and retrieve agent outputs."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_result(self, result: AnalysisResult, prefix: str = "analysis") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{prefix}_{timestamp}.json"
        result.save(str(filepath))
        logger.info("Result saved to %s", filepath)
        return filepath

    def list_results(self, pattern: str = "*.json") -> List[Path]:
        return sorted(self.output_dir.glob(pattern), reverse=True)

    def load_latest(self) -> Optional[AnalysisResult]:
        results = self.list_results()
        if results:
            return AnalysisResult.load(str(results[0]))
        return None

    def export_to_csv(self, output_file: str = "results_summary.csv") -> None:
        results = self.list_results()
        with open(output_file, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "Timestamp",
                "Query",
                "Intent",
                "Interventions",
                "Avg HRS",
                "Processing Time",
                "Errors",
            ])

            for result_path in results:
                result = AnalysisResult.load(str(result_path))
                avg_hrs = 0.0
                if result.translation_analyses:
                    avg_hrs = sum(
                        analysis["human_relevance_score"] for analysis in result.translation_analyses
                    ) / len(result.translation_analyses)

                writer.writerow(
                    [
                        result.timestamp.isoformat(),
                        result.query,
                        result.intent,
                        "; ".join(result.interventions),
                        f"{avg_hrs:.1f}",
                        f"{result.processing_time:.2f}",
                        len(result.errors),
                    ]
                )

        logger.info("Exported %s results to %s", len(results), output_file)


# ================= BATCH PROCESSING =================


class BatchProcessor:
    """Simple batch runner around the agent callable."""

    def __init__(self, agent_runner, result_manager: ResultManager):
        self.agent_runner = agent_runner
        self.result_manager = result_manager

    def process_batch(self, queries: List[str], save_results: bool = True) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        for query in queries:
            start = datetime.now()
            try:
                raw = self.agent_runner(query)
                duration = (datetime.now() - start).total_seconds()
                result = AnalysisResult(
                    query=query,
                    timestamp=start,
                    intent=raw.get("intent", "unknown"),
                    interventions=raw.get("entities", {}).get("intervention_names", []),
                    translation_analyses=raw.get("analyses", []),
                    experiment_plans=raw.get("experiment_plans", []),
                    processing_time=duration,
                    errors=raw.get("errors", []),
                    full_report=raw.get("report", ""),
                )
                results.append(result)
                if save_results:
                    self.result_manager.save_result(result)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to process query: %s", query)
                results.append(
                    AnalysisResult(
                        query=query,
                        timestamp=start,
                        intent="error",
                        interventions=[],
                        errors=[str(exc)],
                    )
                )
        return results

    def process_from_file(self, queries_file: str, save_results: bool = True) -> List[AnalysisResult]:
        if queries_file.endswith(".json"):
            with open(queries_file, "r", encoding="utf-8") as handle:
                queries = json.load(handle)
        else:
            with open(queries_file, "r", encoding="utf-8") as handle:
                queries = [line.strip() for line in handle if line.strip()]
        return self.process_batch(queries, save_results)


# ================= COMPARISON HELPERS =================


class ResultComparator:
    """Utility to sort and render analyses side by side."""

    @staticmethod
    def compare_interventions(analyses: List[Dict[str, Any]], metric: str = "human_relevance_score") -> List[Dict[str, Any]]:
        return sorted(analyses, key=lambda item: item.get(metric, 0), reverse=True)

    @staticmethod
    def generate_comparison_table(analyses: List[Dict[str, Any]]) -> str:
        if not analyses:
            return "No analyses to compare"

        table = ["| Intervention | HRS | Confidence | Strengths | Concerns |", "|-------------|-----|-----------|----------|---------|"]
        for analysis in analyses:
            row = [
                analysis.get("intervention_id", "N/A"),
                f"{analysis.get('human_relevance_score', 0):.1f}",
                analysis.get("confidence", "N/A"),
                str(len(analysis.get("key_strengths", []))),
                str(len(analysis.get("key_concerns", []))),
            ]
            table.append("| " + " | ".join(row) + " |")
        return "\n".join(table)


# ================= QUERY CACHE =================


class QueryCache:
    """Very small disk-backed cache keyed by query text."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return None

    def set(self, query: str, result: Dict[str, Any]) -> None:
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
