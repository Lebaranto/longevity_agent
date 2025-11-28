from __future__ import annotations

from functools import partial
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from dotenv import load_dotenv

from world_model import WorldModel

load_dotenv()


# ================= STATE DEFINITION =================


class AgentState(MessagesState):
    """State passed between LangGraph nodes."""

    user_query: str
    intent: NotRequired[str]
    extracted_entities: NotRequired[Dict[str, Any]]
    world_model_data: NotRequired[Dict[str, Any]]
    micro_reasoning: NotRequired[List[str]]
    translation_analysis: NotRequired[Dict[str, Any]]
    experiment_plan: NotRequired[List[Dict[str, Any]]]
    final_report: NotRequired[str]
    errors: NotRequired[List[str]]
    next_step: NotRequired[str]


# ================= STRUCTURED OUTPUT SCHEMAS =================


class ExtractedEntities(BaseModel):
    """Entities extracted from the incoming query."""

    intervention_names: List[str] = Field(description="Explicitly named interventions")
    species_mentioned: List[str] = Field(default_factory=list, description="Species mentioned in the prompt")
    comparison_request: bool = Field(default=False, description="Whether the user is comparing interventions")
    question_type: str = Field(description="analyze, compare, search, general")
    is_relevant: bool = Field(description="True if the query relates to longevity or aging science")


class TranslationAnalysis(BaseModel):
    """Result of the translation potential analysis."""

    intervention_id: str
    human_relevance_score: float = Field(ge=0, le=100)
    confidence: str = Field(description="high/medium/low")
    key_strengths: List[str]
    key_concerns: List[str]
    mechanistic_reasoning: str
    bottom_line: str


class ExperimentProposal(BaseModel):
    """Experiment planning suggestion."""

    priority: int = Field(ge=1, le=3, description="1=highest, 3=lowest")
    experiment_type: str
    design_summary: str
    addresses_gap: str
    expected_uncertainty_reduction: float = Field(ge=0, le=100)
    estimated_cost: str
    estimated_duration: str


class PlannerResponse(BaseModel):
    """List of experiments to run next."""

    experiments: List[ExperimentProposal]


# ================= PROMPTS =================


INTENT_EXTRACTION_PROMPT = """You are a guardrail and intent extractor for a longevity research agent.

User query:
{user_query}

Decide whether the query is relevant to longevity, aging biology, translational research, or experiment design. If it is off-topic (travel, math homework, personal advice, etc.), mark is_relevant as false.

Extract:
1. intervention_names: concrete interventions (rapamycin, metformin, caloric restriction, etc.)
2. species_mentioned: explicit species names
3. comparison_request: true/false
4. question_type: one of analyze, compare, search, general
5. is_relevant: true if longevity/aging related, else false

Respond strictly as JSON with keys: intervention_names, species_mentioned, comparison_request, question_type, is_relevant."""


MICRO_REASONING_PROMPT = """You are a concise micro-reasoning layer for a longevity translation agent.
Intervention: {name}
World model formal score: {score}
World model warnings: {warnings}
Targets: {targets}
Cross-species effects: {effects}

Produce 3 short bullet points (<=40 words each) that link the formal score to practical intuition and identify one potential hidden risk or confounder. Keep it compact."""


TRANSLATOR_AGENT_PROMPT = """You are an expert in cross-species translation for longevity interventions.

# CONTEXT
Intervention: {intervention_name}
Formal score from knowledge graph rules: {formal_score}
Targets: {targets}
Effects by species:
{effects}
Pathways involved:
{pathways}
Human longevity evidence: {human_evidence}
Reasoning trace from scoring rules:
{reasoning_steps}
Micro reasoning hints:
{micro_reasoning}
Warnings:
{warnings}

# TASK
Provide a JSON object with:
- intervention_id
- human_relevance_score (0-100)
- confidence (high/medium/low)
- key_strengths (2-4 items)
- key_concerns (2-4 items)
- mechanistic_reasoning (2-3 sentences)
- bottom_line (1-2 sentences)
"""


PLANNER_AGENT_PROMPT = """You are an experiment planner for validating longevity interventions.

# CONTEXT
Intervention: {intervention_name}
Human Relevance Score: {hrs}
Confidence: {confidence}

# CURRENT DATA
{current_data}

# IDENTIFIED GAPS
{identified_gaps}

# AVAILABLE EXPERIMENT TYPES
1. In vitro (human cells) — $10-50k, 2-4 months
2. Ex vivo (organoids) — $50-150k, 4-6 months
3. Additional model organisms — $30-100k, 6-12 months
4. Non-human primates — $500k-2M, 24-60 months
5. Epidemiology — $20-80k, 3-6 months
6. Retrospective clinical data — $15-50k, 2-4 months

Budget limit: $200k. Time limit: 12 months.

Return JSON with structure:
{{
  "experiments": [
    {{
      "priority": ...,
      "experiment_type": "...",
      "design_summary": "...",
      "addresses_gap": "...",
      "expected_uncertainty_reduction": ...,
      "estimated_cost": "...",
      "estimated_duration": "..."
    }}
  ]
}}
"""


# ================= NODE IMPLEMENTATIONS =================


def _extract_intent_node(state: AgentState, *, llm: ChatOpenAI) -> AgentState:
    query = state["user_query"]
    prompt = INTENT_EXTRACTION_PROMPT.format(user_query=query)

    try:
        llm_with_structure = llm.with_structured_output(ExtractedEntities)
        result = llm_with_structure.invoke(prompt)

        state["intent"] = result.question_type
        state["extracted_entities"] = {
            "intervention_names": result.intervention_names,
            "species_mentioned": result.species_mentioned,
            "comparison_request": result.comparison_request,
            "is_relevant": result.is_relevant,
        }

        state["messages"] = [f"Intent: {result.question_type}"]
    except Exception as exc:  # pragma: no cover - defensive
        state["errors"] = [f"Intent extraction failed: {exc}"]
        state["next_step"] = "end"

    return state


def _route_after_intent(state: AgentState) -> str:
    if state.get("errors"):
        return "end"

    entities = state.get("extracted_entities", {})

    if not entities.get("is_relevant", True):
        state["final_report"] = (
            "The request does not look related to longevity or aging. "
            "Please rephrase with a concrete intervention or aging topic, or ask to terminate."
        )
        return "end"

    if not entities.get("intervention_names"):
        state["final_report"] = (
            "I could not find specific interventions to analyze. "
            "Please name at least one intervention (e.g., rapamycin, metformin, caloric restriction)."
        )
        return "end"

    return "query_wm"


def _query_world_model_node(state: AgentState, *, world_model: WorldModel) -> AgentState:
    entities = state["extracted_entities"]
    intervention_names = entities["intervention_names"]

    wm_data = []

    for name in intervention_names:
        iv = world_model.find_intervention_by_name(name)

        if iv is None:
            state["messages"] = [f"Intervention '{name}' is not in the knowledge base."]
            continue

        score_result = world_model.compute_translation_score(iv.id)

        pathways_info = []
        for target_id in iv.targets:
            gene = world_model.genes.get(target_id)
            if gene:
                for pw_id in gene.pathways:
                    pw = world_model.pathways.get(pw_id)
                    if pw:
                        pathways_info.append(
                            {
                                "id": pw.id,
                                "name": pw.name,
                                "conserved": pw.conserved_human_mouse,
                                "hallmarks": pw.hallmarks,
                            }
                        )

        human_evidence = []
        for target_id in iv.targets:
            gene = world_model.genes.get(target_id)
            if gene:
                ev = gene.longevity_evidence.get("human", "none")
                human_evidence.append(f"{target_id}: {ev}")

        wm_data.append(
            {
                "intervention": {
                    "id": iv.id,
                    "name": iv.name,
                    "type": iv.type,
                    "targets": iv.targets,
                    "effects": [effect.model_dump() for effect in iv.effects],
                },
                "formal_score": score_result["score"],
                "reasoning_steps": score_result["steps"],
                "warnings": score_result["warnings"],
                "pathways": pathways_info,
                "human_evidence": human_evidence,
            }
        )

    state["world_model_data"] = {"interventions": wm_data}
    state["messages"] = [f"Loaded {len(wm_data)} interventions from the knowledge graph."]
    return state


def _micro_reasoning_node(state: AgentState, *, llm: ChatOpenAI) -> AgentState:
    wm_interventions = state.get("world_model_data", {}).get("interventions", [])
    mini_insights: List[str] = []

    for iv_data in wm_interventions:
        iv = iv_data["intervention"]
        prompt = MICRO_REASONING_PROMPT.format(
            name=iv["name"],
            score=iv_data.get("formal_score", 0),
            warnings=", ".join(iv_data.get("warnings", [])) or "none",
            targets=", ".join(iv.get("targets", [])),
            effects=", ".join(
                f"{eff['species']}: {eff['lifespan_change_pct']}% ({eff['evidence_level']})"
                for eff in iv.get("effects", [])
            ),
        )

        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            mini_insights.append(response.content.strip())
        except Exception as exc:  # pragma: no cover - defensive
            mini_insights.append(f"Micro reasoning failed for {iv['name']}: {exc}")

    state["micro_reasoning"] = mini_insights
    return state


def _translator_agent_node(state: AgentState, *, llm: ChatOpenAI) -> AgentState:
    wm_data = state["world_model_data"]["interventions"]
    micro_reasoning = state.get("micro_reasoning", [])

    analyses = []

    for idx, iv_data in enumerate(wm_data):
        iv = iv_data["intervention"]

        effects_str = "\n".join(
            f"  - {effect['species']}: {effect['lifespan_change_pct']}% ({effect['evidence_level']})"
            for effect in iv["effects"]
        )

        pathways_str = "\n".join(
            f"  - {path['name']} (conserved: {path['conserved']})"
            for path in iv_data["pathways"]
        )

        steps_str = "\n".join(
            f"  - {step['rule_name']}: {step['explanation']} (Δ={step['delta']})"
            for step in iv_data["reasoning_steps"]
        )

        micro_hint = micro_reasoning[idx] if idx < len(micro_reasoning) else ""
        warnings_str = "\n".join(f"  - {warn}" for warn in iv_data["warnings"]) or "None"

        prompt = TRANSLATOR_AGENT_PROMPT.format(
            intervention_name=iv["name"],
            formal_score=iv_data["formal_score"],
            targets=", ".join(iv["targets"]),
            effects=effects_str,
            pathways=pathways_str,
            human_evidence=", ".join(iv_data["human_evidence"]),
            reasoning_steps=steps_str,
            micro_reasoning=micro_hint,
            warnings=warnings_str,
        )

        try:
            llm_with_structure = llm.with_structured_output(TranslationAnalysis)
            analysis = llm_with_structure.invoke(prompt)
            if not analysis.intervention_id:
                analysis.intervention_id = iv["id"]
            analyses.append(analysis.model_dump())
        except Exception as exc:  # pragma: no cover - defensive
            state["errors"] = state.get("errors", []) + [f"Translation analysis failed for {iv['name']}: {exc}"]

    state["translation_analysis"] = {"analyses": analyses}
    state["messages"] = [f"Generated translation analysis for {len(analyses)} interventions."]
    return state


def _planner_agent_node(state: AgentState, *, llm: ChatOpenAI) -> AgentState:
    wm_data = state["world_model_data"]["interventions"]
    analyses = state["translation_analysis"]["analyses"]

    all_plans = []

    for iv_data, analysis in zip(wm_data, analyses):
        iv = iv_data["intervention"]

        gaps = []

        species_tested = {effect["species"] for effect in iv["effects"]}
        if "mouse" not in species_tested:
            gaps.append("Missing mouse data as the closest tested mammal.")

        if species_tested.issubset({"fly", "worm"}):
            gaps.append("Only invertebrate evidence is available; mammalian validation is needed.")

        weak_evidence = any(effect["evidence_level"] in {"weak", "mixed"} for effect in iv["effects"])
        if weak_evidence:
            gaps.append("Evidence quality is weak or mixed; replication is required.")

        has_human = any("strong" in ev.lower() or "candidate" in ev.lower() for ev in iv_data["human_evidence"])
        if not has_human:
            gaps.append("No direct human longevity evidence for the targets.")

        current_data_str = f"Species tested: {', '.join(species_tested)}\nEffects: {iv['effects']}"
        gaps_str = "\n".join(f"  - {gap}" for gap in gaps) or "No critical gaps detected."

        prompt = PLANNER_AGENT_PROMPT.format(
            intervention_name=iv["name"],
            hrs=analysis["human_relevance_score"],
            confidence=analysis["confidence"],
            current_data=current_data_str,
            identified_gaps=gaps_str,
        )

        try:
            llm_with_structure = llm.with_structured_output(PlannerResponse)
            resp: PlannerResponse = llm_with_structure.invoke(prompt)

            all_plans.append(
                {
                    "intervention_id": iv["id"],
                    "intervention_name": iv["name"],
                    "experiments": [exp.model_dump() for exp in resp.experiments],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            state["errors"] = state.get("errors", []) + [f"Planning failed for {iv['name']}: {exc}"]

    state["experiment_plan"] = all_plans
    state["messages"] = ["Generated experiment plans."]
    return state


def _generate_report_node(state: AgentState) -> AgentState:
    wm_data = state["world_model_data"].get("interventions", [])
    analyses = state["translation_analysis"].get("analyses", [])
    plans = state.get("experiment_plan", [])

    report_sections: List[str] = []

    report_sections.append("# Longevity Intervention Report\n")
    report_sections.append(f"Query: {state['user_query']}\n")
    report_sections.append("=" * 80 + "\n")

    if not wm_data or not analyses:
        report_sections.append("No detailed report available — missing intervention data.\n")
    else:
        for iv_data, analysis, plan in zip(wm_data, analyses, plans):
            iv = iv_data["intervention"]

            report_sections.append(f"\n## Intervention: {iv['name']}\n")
            report_sections.append(f"Type: {iv['type']}\n")
            report_sections.append(f"Targets: {', '.join(iv['targets'])}\n")

            report_sections.append(
                f"\n### Human Relevance Score: {analysis['human_relevance_score']:.1f}/100"
            )
            report_sections.append(f"Confidence: {analysis['confidence']}\n")

            report_sections.append(f"\n**Bottom Line:** {analysis['bottom_line']}\n")

            report_sections.append("\n**Key strengths:**")
            for strength in analysis["key_strengths"]:
                report_sections.append(f"  ✓ {strength}")

            report_sections.append("\n**Key concerns:**")
            for concern in analysis["key_concerns"]:
                report_sections.append(f"  ⚠ {concern}")

            report_sections.append("\n**Mechanistic reasoning:**")
            report_sections.append(f"{analysis['mechanistic_reasoning']}\n")

            report_sections.append("\n### Suggested experiments:\n")
            for exp in plan.get("experiments", []):
                report_sections.append(f"\n**Priority {exp['priority']}: {exp['experiment_type']}**")
                report_sections.append(f"Design: {exp['design_summary']}")
                report_sections.append(f"Addresses gap: {exp['addresses_gap']}")
                report_sections.append(
                    f"Expected uncertainty reduction: {exp['expected_uncertainty_reduction']}%"
                )
                report_sections.append(
                    f"Cost: {exp['estimated_cost']}, Duration: {exp['estimated_duration']}\n"
                )

            report_sections.append("\n" + "-" * 80 + "\n")

    state["final_report"] = "\n".join(report_sections)
    state["messages"] = ["Report generated."]
    return state


# ================= PUBLIC BUILDER =================


def build_agent_workflow(world_model_path: str, llm_model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Create the LangGraph workflow and supporting context."""

    world_model = WorldModel(world_model_path)
    llm = ChatOpenAI(model=llm_model, temperature=temperature)

    workflow = StateGraph(AgentState)

    workflow.add_node("extract_intent", partial(_extract_intent_node, llm=llm))
    workflow.add_node("query_world_model", partial(_query_world_model_node, world_model=world_model))
    workflow.add_node("micro_reasoning_step", partial(_micro_reasoning_node, llm=llm))
    workflow.add_node("translator_agent", partial(_translator_agent_node, llm=llm))
    workflow.add_node("planner_agent", partial(_planner_agent_node, llm=llm))
    workflow.add_node("generate_report", _generate_report_node)

    workflow.set_entry_point("extract_intent")

    workflow.add_conditional_edges(
        "extract_intent", _route_after_intent, {"query_wm": "query_world_model", "end": END}
    )

    workflow.add_edge("query_world_model", "micro_reasoning_step")
    workflow.add_edge("micro_reasoning_step", "translator_agent")
    workflow.add_edge("translator_agent", "planner_agent")
    workflow.add_edge("planner_agent", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def create_agent_runner(world_model_path: str = "ontology.yaml", llm_model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Return a callable that runs the full pipeline for a single query."""

    app = build_agent_workflow(world_model_path, llm_model=llm_model, temperature=temperature)

    def run(query: str) -> Dict[str, Any]:
        initial_state: AgentState = {"user_query": query, "messages": []}
        result = app.invoke(initial_state)

        return {
            "report": result.get("final_report", "Report was not generated."),
            "intent": result.get("intent"),
            "entities": result.get("extracted_entities"),
            "analyses": result.get("translation_analysis", {}).get("analyses", []),
            "experiment_plans": result.get("experiment_plan", []),
            "logs": result.get("messages", []),
            "errors": result.get("errors", []),
        }

    return run


if __name__ == "__main__":
    run_agent = create_agent_runner(world_model_path="ontology.yaml", llm_model="gpt-4o-mini", temperature=0.0)

    sample_queries = [
        "Analyze rapamycin as a candidate to extend human healthspan.",
        "Compare metformin and caloric restriction for translational potential.",
        "What experiments would validate senolytics for humans?",
    ]

    for query in sample_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80 + "\n")

        result = run_agent(query)

        if result["errors"]:
            print("Errors:")
            for error in result["errors"]:
                print(f"  - {error}")

        print(result["report"])
        print("\n")
