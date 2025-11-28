from __future__ import annotations

from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal
from typing_extensions import NotRequired
import operator
import json

from langgraph.graph import StateGraph, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from world_model import WorldModel


# ============ STATE SCHEMA ============

class AgentState(MessagesState):
    """Состояние агентной системы"""
    # Пользовательский ввод
    user_query: str

    # Распознанное намерение
    intent: NotRequired[str]  # "analyze_intervention", "compare", "search", "general_question"
    extracted_entities: NotRequired[Dict[str, Any]]  # intervention_names, species, etc

    # Данные из World Model
    world_model_data: NotRequired[Dict[str, Any]]

    # Результаты анализа
    translation_analysis: NotRequired[Dict[str, Any]]
    experiment_plan: NotRequired[List[Dict[str, Any]]]

    # Финальный отчет
    final_report: NotRequired[str]

    # Служебная информация
    errors: NotRequired[List[str]]
    next_step: NotRequired[str]


# ============ STRUCTURED OUTPUT SCHEMAS ============

class ExtractedEntities(BaseModel):
    """Извлеченные из запроса сущности"""
    intervention_names: List[str] = Field(description="Названия интервенций")
    species_mentioned: List[str] = Field(default_factory=list, description="Упомянутые виды")
    comparison_request: bool = Field(default=False, description="Запрос на сравнение")
    question_type: str = Field(description="Тип вопроса: analyze, compare, search, general")


class TranslationAnalysis(BaseModel):
    """Результат анализа трансляционного потенциала"""
    intervention_id: str
    human_relevance_score: float = Field(ge=0, le=100)
    confidence: str = Field(description="high/medium/low")
    key_strengths: List[str]
    key_concerns: List[str]
    mechanistic_reasoning: str
    bottom_line: str


class ExperimentProposal(BaseModel):
    """Предложение эксперимента"""
    priority: int = Field(ge=1, le=3, description="1=highest, 3=lowest")
    experiment_type: str
    design_summary: str
    addresses_gap: str
    expected_uncertainty_reduction: float = Field(ge=0, le=100)
    estimated_cost: str
    estimated_duration: str


class PlannerResponse(BaseModel):
    """Ответ планировщика: список экспериментов"""
    experiments: List[ExperimentProposal]


# ============ ПРОМПТЫ ============

INTENT_EXTRACTION_PROMPT = """Ты — эксперт по обработке запросов о longevity-интервенциях.

Пользовательский запрос:
{user_query}

Проанализируй запрос и извлеки:
1. Названия упомянутых интервенций (rapamycin, metformin, caloric restriction и т.д.)
2. Упомянутые виды (mouse, human, fly, worm)
3. Является ли это запросом на сравнение нескольких интервенций
4. Тип вопроса:
   - "analyze" - детальный анализ одной/нескольких интервенций
   - "compare" - сравнение интервенций
   - "search" - поиск информации
   - "general" - общий вопрос о longevity

Ответь в формате JSON со следующими полями:
- intervention_names: список строк
- species_mentioned: список строк
- comparison_request: bool
- question_type: str

Если интервенция упомянута в общей форме (например, "сенолитики"), укажи конкретные примеры из известных тебе."""

TRANSLATOR_AGENT_PROMPT = """Ты — эксперт по кросс-видовой трансляции результатов longevity-исследований.

# КОНТЕКСТ
Интервенция: {intervention_name}
Базовый score из формальных правил: {formal_score}

# ДАННЫЕ ИЗ WORLD MODEL
Таргеты: {targets}
Эффекты по видам: {effects}
Задействованные пути: {pathways}
Human longevity evidence: {human_evidence}

# ШАГИ ФОРМАЛЬНОГО ВЫВОДА
{reasoning_steps}

# ПРЕДУПРЕЖДЕНИЯ
{warnings}

# ТВОЯ ЗАДАЧА
Проведи глубокий анализ трансляционного потенциала этой интервенции для человека.

1. **Human Relevance Score (0-100)**: Оцени вероятность успешной трансляции с учетом:
   - Филогенетическое расстояние протестированных видов
   - Консервативность молекулярных механизмов
   - Наличие human longevity evidence
   - Качество экспериментальных данных
   - Формальный score: {formal_score}

2. **Confidence** (high/medium/low): Уверенность в оценке

3. **Key Strengths** (2-4 пункта): Главные сильные стороны интервенции

4. **Key Concerns** (2-4 пункта): Основные риски и ограничения

5. **Mechanistic Reasoning**: Почему механизм действия может/не может работать у человека (2-3 предложения)

6. **Bottom Line**: Краткая рекомендация (1-2 предложения)

Ответь в формате JSON с полями:
- intervention_id: str
- human_relevance_score: float (0-100)
- confidence: str
- key_strengths: List[str]
- key_concerns: List[str]
- mechanistic_reasoning: str
- bottom_line: str"""

PLANNER_AGENT_PROMPT = """Ты — планировщик экспериментов для валидации longevity-интервенций.

# КОНТЕКСТ
Интервенция: {intervention_name}
Human Relevance Score: {hrs}
Confidence: {confidence}

# ТЕКУЩИЕ ДАННЫЕ
{current_data}

# ВЫЯВЛЕННЫЕ ПРОБЕЛЫ
{identified_gaps}

# ДОСТУПНЫЕ ТИПЫ ЭКСПЕРИМЕНТОВ
1. **In vitro (человеческие клетки)** - стоимость: $10-50k, время: 2-4 мес
   Подходит для: проверки механизма на человеческих клеточных линиях

2. **Ex vivo (органоиды)** - стоимость: $50-150k, время: 4-6 мес
   Подходит для: тестирования на сложных тканевых моделях

3. **Дополнительные модельные организмы** - стоимость: $30-100k, время: 6-12 мес
   Подходит для: заполнения филогенетических пробелов

4. **Non-human primates** - стоимость: $500k-2M, время: 2-5 лет
   Подходит для: финальной валидации перед клиникой

5. **Эпидемиологический анализ** - стоимость: $20-80k, время: 3-6 мес
   Подходит для: поиска корреляций в человеческих популяциях

6. **Ретроспективный анализ клинических данных** - стоимость: $15-50k, время: 2-4 мес
   Подходит для: одобренных препаратов с историей применения

# ТВОЯ ЗАДАЧА
Предложи 2-3 эксперимента, которые максимально уменьшат неопределенность и риски.

ОГРАНИЧЕНИЯ:
- Бюджет: $200k
- Время: 12 месяцев

Для каждого эксперимента укажи:
- priority: 1 (highest), 2 (medium), 3 (lowest)
- experiment_type: тип из списка выше
- design_summary: краткое описание дизайна (2-3 предложения)
- addresses_gap: какой пробел в знаниях закрывает
- expected_uncertainty_reduction: снижение неопределенности в % (0-100)
- estimated_cost: диапазон ($)
- estimated_duration: время в месяцах

Ответь в формате JSON с ОБЪЕКТОМ следующей структуры (СТРОГО В JSON, НИКАК ИНАЧЕ):
{
  "experiments": [
    {
      "priority": ...,
      "experiment_type": "...",
      "design_summary": "...",
      "addresses_gap": "...",
      "expected_uncertainty_reduction": ...,
      "estimated_cost": "...",
      "estimated_duration": "..."
    },
    ...
  ]
}

ВАЖНО: Учитывай текущий уровень confidence и HRS при планировании:
- Если confidence=low → начни с базовых экспериментов (in vitro)
- Если confidence=high и HRS>70 → можно сразу к приматам/клинике
- Если есть одобренный препарат → приоритет на эпидемиологию"""


# ============ АГЕНТНАЯ СИСТЕМА ============

class LongevityAgentSystem:
    def __init__(
        self,
        world_model_path: str,
        llm_model: str = "gpt-5-mini",
        temperature: float = 0.0
    ):
        self.world_model = WorldModel(world_model_path)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """Строит LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Узлы
        workflow.add_node("extract_intent", self._extract_intent_node)
        workflow.add_node("query_world_model", self._query_world_model_node)
        workflow.add_node("translator_agent", self._translator_agent_node)
        workflow.add_node("planner_agent", self._planner_agent_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Граф выполнения
        workflow.set_entry_point("extract_intent")

        # Условная маршрутизация после извлечения намерений
        workflow.add_conditional_edges(
            "extract_intent",
            self._route_after_intent,
            {
                "query_wm": "query_world_model",
                "end": END,
            }
        )

        workflow.add_edge("query_world_model", "translator_agent")
        workflow.add_edge("translator_agent", "planner_agent")
        workflow.add_edge("planner_agent", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow

    # ============ УЗЛЫ ГРАФА ============

    def _extract_intent_node(self, state: AgentState) -> AgentState:
        """Узел 1: Извлечение намерений и сущностей"""
        query = state["user_query"]

        prompt = INTENT_EXTRACTION_PROMPT.format(user_query=query)

        try:
            llm_with_structure = self.llm.with_structured_output(ExtractedEntities)
            result = llm_with_structure.invoke(prompt)

            state["intent"] = result.question_type
            state["extracted_entities"] = {
                "intervention_names": result.intervention_names,
                "species_mentioned": result.species_mentioned,
                "comparison_request": result.comparison_request,
            }

            state["messages"] = [
                f"✓ Intent: {result.question_type}",
                f"✓ Извлечено интервенций: {len(result.intervention_names)}",
            ]

        except Exception as e:
            state["errors"] = [f"Ошибка извлечения намерений: {str(e)}"]
            state["next_step"] = "end"

        return state

    def _route_after_intent(self, state: AgentState) -> str:
        """Маршрутизация после извлечения намерений"""
        if state.get("errors"):
            return "end"

        entities = state.get("extracted_entities", {})
        if not entities.get("intervention_names"):
            state["final_report"] = (
                "Не удалось определить конкретные интервенции для анализа. "
                "Пожалуйста, уточните запрос, упомянув название интервенции "
                "(например: rapamycin, metformin, caloric restriction)."
            )
            return "end"

        return "query_wm"

    def _query_world_model_node(self, state: AgentState) -> AgentState:
        """Узел 2: Запрос к World Model"""
        entities = state["extracted_entities"]
        intervention_names = entities["intervention_names"]

        wm_data = []

        for name in intervention_names:
            iv = self.world_model.find_intervention_by_name(name)

            if iv is None:
                # возвращаем только новое сообщение — остальное склеит MessagesState
                state["messages"] = [
                    f"⚠ Интервенция '{name}' не найдена в базе знаний"
                ]
                continue

            score_result = self.world_model.compute_translation_score(iv.id)

            pathways_info = []
            for target_id in iv.targets:
                gene = self.world_model.genes.get(target_id)
                if gene:
                    for pw_id in gene.pathways:
                        pw = self.world_model.pathways.get(pw_id)
                        if pw:
                            pathways_info.append({
                                "id": pw.id,
                                "name": pw.name,
                                "conserved": pw.conserved_human_mouse,
                                "hallmarks": pw.hallmarks
                            })

            human_evidence = []
            for target_id in iv.targets:
                gene = self.world_model.genes.get(target_id)
                if gene:
                    ev = gene.longevity_evidence.get("human", "none")
                    human_evidence.append(f"{target_id}: {ev}")

            wm_data.append({
                "intervention": {
                    "id": iv.id,
                    "name": iv.name,
                    "type": iv.type,
                    "targets": iv.targets,
                    "effects": [e.model_dump() for e in iv.effects],
                },
                "formal_score": score_result["score"],
                "reasoning_steps": score_result["steps"],
                "warnings": score_result["warnings"],
                "pathways": pathways_info,
                "human_evidence": human_evidence,
            })

        state["world_model_data"] = {"interventions": wm_data}
        state["messages"] = [
            f"✓ Загружено данных для {len(wm_data)} интервенций"
        ]

        return state

    def _translator_agent_node(self, state: AgentState) -> AgentState:
        """Узел 3: Анализ трансляционного потенциала"""
        wm_data = state["world_model_data"]["interventions"]

        analyses = []

        for iv_data in wm_data:
            iv = iv_data["intervention"]

            effects_str = "\n".join([
                f"  - {e['species']}: {e['lifespan_change_pct']}% ({e['evidence_level']})"
                for e in iv["effects"]
            ])

            pathways_str = "\n".join([
                f"  - {p['name']} (консервативен: {p['conserved']})"
                for p in iv_data["pathways"]
            ])

            steps_str = "\n".join([
                f"  - {s['rule_name']}: {s['explanation']} (Δ={s['delta']})"
                for s in iv_data["reasoning_steps"]
            ])

            warnings_str = "\n".join([f"  - {w}" for w in iv_data["warnings"]]) or "Нет предупреждений"

            prompt = TRANSLATOR_AGENT_PROMPT.format(
                intervention_name=iv["name"],
                formal_score=iv_data["formal_score"],
                targets=", ".join(iv["targets"]),
                effects=effects_str,
                pathways=pathways_str,
                human_evidence=", ".join(iv_data["human_evidence"]),
                reasoning_steps=steps_str,
                warnings=warnings_str,
            )

            try:
                llm_with_structure = self.llm.with_structured_output(TranslationAnalysis)
                analysis = llm_with_structure.invoke(prompt)
                if not analysis.intervention_id:
                    analysis.intervention_id = iv["id"]
                analyses.append(analysis.model_dump())
            except Exception as e:
                state["errors"] = state.get("errors", []) + [
                    f"Ошибка анализа {iv['name']}: {str(e)}"
                ]

        state["translation_analysis"] = {"analyses": analyses}
        state["messages"] = [
            f"✓ Проведен анализ для {len(analyses)} интервенций"
        ]

        return state

    def _planner_agent_node(self, state: AgentState) -> AgentState:
        """Узел 4: Планирование экспериментов"""
        wm_data = state["world_model_data"]["interventions"]
        analyses = state["translation_analysis"]["analyses"]

        all_plans = []

        for iv_data, analysis in zip(wm_data, analyses):
            iv = iv_data["intervention"]

            gaps = []

            species_tested = {e["species"] for e in iv["effects"]}
            if "mouse" not in species_tested:
                gaps.append("Отсутствуют данные по мышам (ближайший к человеку протестированный вид)")

            if species_tested.issubset({"fly", "worm"}):
                gaps.append("Данные только по беспозвоночным - нужна валидация на млекопитающих")

            weak_evidence = any(e["evidence_level"] in {"weak", "mixed"} for e in iv["effects"])
            if weak_evidence:
                gaps.append("Гетерогенный или слабый evidence - требуется репликация")

            if not any("strong" in ev or "candidate" in ev for ev in iv_data["human_evidence"]):
                gaps.append("Отсутствие прямых данных по человеческим генам долголетия")

            current_data_str = f"Виды: {', '.join(species_tested)}\nЭффекты: {iv['effects']}"
            gaps_str = "\n".join([f"  - {g}" for g in gaps]) or "Пробелов не выявлено"

            prompt = PLANNER_AGENT_PROMPT.format(
                intervention_name=iv["name"],
                hrs=analysis["human_relevance_score"],
                confidence=analysis["confidence"],
                current_data=current_data_str,
                identified_gaps=gaps_str,
            )

            try:
                llm_with_structure = self.llm.with_structured_output(PlannerResponse)
                resp: PlannerResponse = llm_with_structure.invoke(prompt)

                all_plans.append({
                    "intervention_id": iv["id"],
                    "intervention_name": iv["name"],
                    "experiments": [e.model_dump() for e in resp.experiments]
                })
            except Exception as e:
                state["errors"] = state.get("errors", []) + [
                    f"Ошибка планирования для {iv['name']}: {str(e)}"
                ]

        state["experiment_plan"] = all_plans
        state["messages"] = [
            "✓ Сгенерированы планы экспериментов"
        ]

        return state

    def _generate_report_node(self, state: AgentState) -> AgentState:
        """Узел 5: Генерация финального отчета"""
        wm_data = state["world_model_data"]["interventions"]
        analyses = state["translation_analysis"]["analyses"]
        plans = state.get("experiment_plan", [])

        report_sections: List[str] = []

        report_sections.append("# ОТЧЕТ: АНАЛИЗ LONGEVITY-ИНТЕРВЕНЦИЙ\n")
        report_sections.append(f"Запрос: {state['user_query']}\n")
        report_sections.append("=" * 80 + "\n")

        if not wm_data or not analyses:
            report_sections.append("\nНе удалось построить детальный отчет — нет данных по интервенциям.\n")
        else:
            for iv_data, analysis, plan in zip(wm_data, analyses, plans):
                iv = iv_data["intervention"]

                report_sections.append(f"\n## ИНТЕРВЕНЦИЯ: {iv['name'].upper()}\n")
                report_sections.append(f"Тип: {iv['type']}\n")
                report_sections.append(f"Таргеты: {', '.join(iv['targets'])}\n")

                report_sections.append(
                    f"\n### Human Relevance Score: {analysis['human_relevance_score']:.1f}/100"
                )
                report_sections.append(f"Confidence: {analysis['confidence']}\n")

                report_sections.append(f"\n**Bottom Line:** {analysis['bottom_line']}\n")

                report_sections.append("\n**Сильные стороны:**")
                for strength in analysis['key_strengths']:
                    report_sections.append(f"  ✓ {strength}")

                report_sections.append("\n**Риски и ограничения:**")
                for concern in analysis['key_concerns']:
                    report_sections.append(f"  ⚠ {concern}")

                report_sections.append(f"\n**Механистическое обоснование:**")
                report_sections.append(f"{analysis['mechanistic_reasoning']}\n")

                report_sections.append("\n### Предлагаемые эксперименты:\n")
                for exp in plan.get("experiments", []):
                    report_sections.append(
                        f"\n**Приоритет {exp['priority']}: {exp['experiment_type']}**"
                    )
                    report_sections.append(f"Дизайн: {exp['design_summary']}")
                    report_sections.append(f"Закрывает пробел: {exp['addresses_gap']}")
                    report_sections.append(
                        f"Снижение неопределенности: {exp['expected_uncertainty_reduction']}%"
                    )
                    report_sections.append(
                        f"Стоимость: {exp['estimated_cost']}, Срок: {exp['estimated_duration']}\n"
                    )

                report_sections.append("\n" + "-" * 80 + "\n")

        state["final_report"] = "\n".join(report_sections)
        state["messages"] = [
            "✓ Отчет сгенерирован"
        ]

        return state

    # ============ PUBLIC API ============

    def process_query(self, query: str) -> Dict[str, Any]:
        """Обрабатывает пользовательский запрос"""
        initial_state: AgentState = {
            "user_query": query,
            "messages": []
        }

        result = self.app.invoke(initial_state)

        return {
            "report": result.get("final_report", "Отчет не сгенерирован"),
            "intent": result.get("intent"),
            "entities": result.get("extracted_entities"),
            "analyses": result.get("translation_analysis", {}).get("analyses", []),
            "experiment_plans": result.get("experiment_plan", []),
            "logs": result.get("messages", []),
            "errors": result.get("errors", []),
        }


# ============ ПРИМЕР ИСПОЛЬЗОВАНИЯ ============

if __name__ == "__main__":
    agent_system = LongevityAgentSystem(
        world_model_path="onthology.yaml",
        llm_model="gpt-4o-mini",  # или "gpt-4o-mini" для тестирования
        temperature=0.0
    )

    queries = [
        "Проанализируй рапамицин как потенциальную интервенцию для продления жизни человека",
        "Сравни метформин и калорийную рестрикцию по эффективности",
        "Какие эксперименты нужны для валидации сенолитиков?",
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"ЗАПРОС: {query}")
        print(f"{'=' * 80}\n")

        result = agent_system.process_query(query)

        print("ЛОГИ:")
        for log in result["logs"]:
            print(f"  {log}")

        if result["errors"]:
            print("\nОШИБКИ:")
            for error in result["errors"]:
                print(f"  ❌ {error}")

        print("\n" + result["report"])
        print("\n")
