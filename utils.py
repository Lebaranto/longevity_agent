"""
Конфигурация и утилиты для агентной системы
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
from loguru import logger


# ============ КОНФИГУРАЦИЯ ============

@dataclass
class SystemConfig:
    """Конфигурация агентной системы"""
    
    # Paths
    world_model_path: str = "ontology.yaml"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    # LLM settings
    llm_provider: str = "openai"  # openai, anthropic
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4000
    llm_timeout: int = 60  # seconds
    
    # Agent settings
    max_iterations: int = 5
    enable_memory: bool = False
    enable_cache: bool = True
    
    # Scoring thresholds
    high_hrs_threshold: float = 70.0
    medium_hrs_threshold: float = 50.0
    
    # Experiment planning constraints
    max_budget_usd: int = 200000
    max_duration_months: int = 12
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Загрузка конфигурации из YAML"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Сохранение конфигурации в YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


# Конфигурация по умолчанию
DEFAULT_CONFIG = SystemConfig()


# ============ ЛОГИРОВАНИЕ ============

def setup_logging(config: SystemConfig):
    """Настройка системы логирования"""
    
    # Создание директории для логов
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Удаление стандартного handler
    logger.remove()
    
    # Console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    # File handler
    if config.log_to_file:
        log_file = log_dir / f"agent_system_{datetime.now():%Y%m%d_%H%M%S}.log"
        logger.add(
            log_file,
            level=config.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
    
    logger.info("Logging system initialized")


# ============ РЕЗУЛЬТАТЫ ============

@dataclass
class AnalysisResult:
    """Структурированный результат анализа"""
    
    query: str
    timestamp: datetime
    intent: str
    interventions: List[str]
    
    # Analyses
    translation_analyses: List[Dict[str, Any]] = field(default_factory=list)
    experiment_plans: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Report
    full_report: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
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
            "full_report": self.full_report
        }
    
    def save(self, path: str):
        """Сохранение результата"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "AnalysisResult":
        """Загрузка результата"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def get_summary(self) -> str:
        """Краткая сводка"""
        summary = []
        summary.append(f"Query: {self.query}")
        summary.append(f"Timestamp: {self.timestamp:%Y-%m-%d %H:%M:%S}")
        summary.append(f"Intent: {self.intent}")
        summary.append(f"Interventions: {', '.join(self.interventions)}")
        summary.append(f"Processing time: {self.processing_time:.2f}s")
        
        if self.translation_analyses:
            summary.append("\nTranslation Analyses:")
            for analysis in self.translation_analyses:
                summary.append(f"  - {analysis['intervention_id']}: HRS={analysis['human_relevance_score']:.1f}, Confidence={analysis['confidence']}")
        
        if self.errors:
            summary.append(f"\nErrors: {len(self.errors)}")
        
        return "\n".join(summary)


# ============ УТИЛИТЫ ДЛЯ РАБОТЫ С РЕЗУЛЬТАТАМИ ============

class ResultManager:
    """Менеджер для работы с результатами анализа"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_result(self, result: AnalysisResult, prefix: str = "analysis") -> Path:
        """Сохранение результата с автоматическим именованием"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        result.save(str(filepath))
        logger.info(f"Result saved to {filepath}")
        
        return filepath
    
    def list_results(self, pattern: str = "*.json") -> List[Path]:
        """Список всех сохраненных результатов"""
        return sorted(self.output_dir.glob(pattern), reverse=True)
    
    def load_latest(self) -> Optional[AnalysisResult]:
        """Загрузка последнего результата"""
        results = self.list_results()
        if results:
            return AnalysisResult.load(str(results[0]))
        return None
    
    def export_to_csv(self, output_file: str = "results_summary.csv"):
        """Экспорт всех результатов в CSV"""
        import csv
        
        results = self.list_results()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Query", "Intent", "Interventions",
                "Avg HRS", "Processing Time", "Errors"
            ])
            
            for result_path in results:
                result = AnalysisResult.load(str(result_path))
                
                avg_hrs = 0.0
                if result.translation_analyses:
                    avg_hrs = sum(a['human_relevance_score'] for a in result.translation_analyses) / len(result.translation_analyses)
                
                writer.writerow([
                    result.timestamp.isoformat(),
                    result.query,
                    result.intent,
                    "; ".join(result.interventions),
                    f"{avg_hrs:.1f}",
                    f"{result.processing_time:.2f}",
                    len(result.errors)
                ])
        
        logger.info(f"Exported {len(results)} results to {output_file}")


# ============ BATCH PROCESSING ============

class BatchProcessor:
    """Пакетная обработка запросов"""
    
    def __init__(self, agent_system, result_manager: ResultManager):
        self.agent_system = agent_system
        self.result_manager = result_manager
    
    def process_batch(
        self, 
        queries: List[str],
        save_results: bool = True
    ) -> List[AnalysisResult]:
        """Обработка пакета запросов"""
        
        results = []
        
        logger.info(f"Starting batch processing of {len(queries)} queries")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query}")
            
            try:
                import time
                start_time = time.time()
                
                raw_result = self.agent_system.process_query(query)
                
                processing_time = time.time() - start_time
                
                # Конвертация в AnalysisResult
                result = AnalysisResult(
                    query=query,
                    timestamp=datetime.now(),
                    intent=raw_result.get("intent", "unknown"),
                    interventions=raw_result.get("entities", {}).get("intervention_names", []),
                    translation_analyses=raw_result.get("analyses", []),
                    experiment_plans=raw_result.get("experiment_plans", []),
                    processing_time=processing_time,
                    errors=raw_result.get("errors", []),
                    full_report=raw_result.get("report", "")
                )
                
                results.append(result)
                
                if save_results:
                    self.result_manager.save_result(result, prefix=f"batch_{i:03d}")
                
                logger.success(f"Query {i} processed successfully in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                
                # Создание результата с ошибкой
                error_result = AnalysisResult(
                    query=query,
                    timestamp=datetime.now(),
                    intent="error",
                    interventions=[],
                    errors=[str(e)]
                )
                results.append(error_result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        
        return results
    
    def process_from_file(self, queries_file: str, save_results: bool = True) -> List[AnalysisResult]:
        """Обработка запросов из файла"""
        
        # Поддержка JSON или TXT
        if queries_file.endswith('.json'):
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
        else:
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        
        return self.process_batch(queries, save_results)


# ============ СРАВНЕНИЕ РЕЗУЛЬТАТОВ ============

class ResultComparator:
    """Сравнение результатов анализа"""
    
    @staticmethod
    def compare_interventions(
        analyses: List[Dict[str, Any]],
        metric: str = "human_relevance_score"
    ) -> List[Dict[str, Any]]:
        """Сравнение интервенций по метрике"""
        
        sorted_analyses = sorted(
            analyses,
            key=lambda x: x.get(metric, 0),
            reverse=True
        )
        
        return sorted_analyses
    
    @staticmethod
    def generate_comparison_table(analyses: List[Dict[str, Any]]) -> str:
        """Генерация таблицы сравнения"""
        
        if not analyses:
            return "No analyses to compare"
        
        # Header
        table = []
        table.append("| Intervention | HRS | Confidence | Strengths | Concerns |")
        table.append("|-------------|-----|-----------|----------|---------|")
        
        # Rows
        for a in analyses:
            row = [
                a.get('intervention_id', 'N/A'),
                f"{a.get('human_relevance_score', 0):.1f}",
                a.get('confidence', 'N/A'),
                str(len(a.get('key_strengths', []))),
                str(len(a.get('key_concerns', [])))
            ]
            table.append("| " + " | ".join(row) + " |")
        
        return "\n".join(table)


# ============ КЭШИРОВАНИЕ ============

class QueryCache:
    """Простой кэш для запросов"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, query: str) -> str:
        """Генерация ключа кэша"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Получение из кэша"""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            return data
        
        return None
    
    def set(self, query: str, result: Dict[str, Any]):
        """Сохранение в кэш"""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cached result for query: {query[:50]}...")
    
    def clear(self):
        """Очистка кэша"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        logger.info("Cache cleared")


# ============ ПРИМЕР ИСПОЛЬЗОВАНИЯ ============

def example_usage():
    """Пример использования утилит"""
    
    # Настройка конфигурации
    config = SystemConfig(
        llm_model="gpt-4o-mini",
        log_level="DEBUG",
        max_budget_usd=100000
    )
    
    setup_logging(config)
    
    # Менеджер результатов
    result_manager = ResultManager(output_dir="outputs")
    
    # Загрузка последнего результата
    latest_result = result_manager.load_latest()
    if latest_result:
        print(latest_result.get_summary())
    
    # Экспорт в CSV
    result_manager.export_to_csv("summary.csv")
    
    # Сравнение интервенций
    if latest_result and latest_result.translation_analyses:
        comparator = ResultComparator()
        table = comparator.generate_comparison_table(latest_result.translation_analyses)
        print("\nComparison Table:")
        print(table)


if __name__ == "__main__":
    # Создание конфигурации по умолчанию
    default_config = SystemConfig()
    default_config.to_yaml("config.yaml")
    print("Created default config.yaml")
    
    # Пример использования
    example_usage()