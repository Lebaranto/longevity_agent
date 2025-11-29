import os
from typing import Any, Dict, List

import streamlit as st

from agent_system import create_agent_runner
from world_model import WorldModel

st.set_page_config(
    page_title="Longevity Agent PoC",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def init_agent() -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Please configure it in your environment variables.")
        st.stop()

    runner = create_agent_runner(world_model_path="ontology.yaml", llm_model="gpt-4o-mini", temperature=0.0)
    wm = WorldModel("ontology.yaml")
    return {"runner": runner, "world_model": wm}


def _render_analysis_block(analysis: Dict[str, Any]):
    st.subheader(analysis.get("intervention_id", "Intervention"))

    cols = st.columns(3)
    cols[0].metric("Human Relevance Score", f"{analysis['human_relevance_score']:.1f}/100")
    cols[1].metric("Confidence", analysis.get("confidence", "n/a"))
    cols[2].metric("Strengths", len(analysis.get("key_strengths", [])))

    st.markdown("**Mechanistic reasoning**")
    st.info(analysis.get("mechanistic_reasoning", "No reasoning available."))

    st.markdown("**Key strengths**")
    for strength in analysis.get("key_strengths", []):
        st.success(strength)

    st.markdown("**Key concerns**")
    for concern in analysis.get("key_concerns", []):
        st.warning(concern)

    st.markdown("**Bottom line**")
    st.write(analysis.get("bottom_line", ""))


def _render_experiments(plan: Dict[str, Any]):
    st.subheader(plan.get("intervention_name", "Intervention"))
    experiments = plan.get("experiments", [])
    if not experiments:
        st.info("No experiments suggested.")
        return

    for exp in sorted(experiments, key=lambda x: x.get("priority", 3)):
        with st.expander(f"Priority {exp['priority']}: {exp['experiment_type']}", expanded=False):
            st.markdown(f"**Design:** {exp['design_summary']}")
            st.markdown(f"**Addresses gap:** {exp['addresses_gap']}")
            metrics = st.columns(3)
            metrics[0].metric("Uncertainty reduction", f"{exp['expected_uncertainty_reduction']}%")
            metrics[1].metric("Cost", exp['estimated_cost'])
            metrics[2].metric("Duration", exp['estimated_duration'])


def _render_world_model_snapshot(world_model: WorldModel, interventions: List[str]):
    st.subheader("World Model snapshot")
    if not interventions:
        st.info("Run an analysis to see the knowledge graph slice.")
        return

    for name in interventions:
        iv = world_model.find_intervention_by_name(name)
        if not iv:
            continue
        with st.expander(iv.name, expanded=False):
            st.markdown(f"**Type:** {iv.type}")
            st.markdown(f"**Targets:** {', '.join(iv.targets)}")
            st.markdown("**Cross-species effects:**")
            for eff in iv.effects:
                st.write(f"- {eff.species}: {eff.lifespan_change_pct}% ({eff.evidence_level})")


def main():
    st.title("🧬 Longevity Agent PoC")
    st.caption("Minimal interactive demo")

    resources = init_agent()
    runner = resources["runner"]
    world_model = resources["world_model"]

    with st.sidebar:
        st.header("Available interventions")
        st.write(", ".join(sorted(iv.name for iv in world_model.interventions.values())))

        st.markdown("---")
        st.header("Example prompts")
        for example in [
            "Analyze rapamycin for human longevity translation.",
            "Compare metformin and caloric restriction for humans.",
            "Design experiments for senolytics in humans.",
        ]:
            if st.button(example, use_container_width=True):
                st.session_state["query"] = example

    st.subheader("Ask the agent")
    query = st.text_area(
        "Enter a longevity question or intervention to analyze:",
        value=st.session_state.get("query", ""),
        height=120,
        placeholder="e.g., Evaluate the translational potential of rapamycin for human healthspan",
    )

    run_col, clear_col = st.columns([1, 1])
    run_clicked = run_col.button("Run analysis", type="primary", use_container_width=True)
    if clear_col.button("Clear", use_container_width=True):
        st.session_state.clear()
        st.experimental_rerun()

    if run_clicked and query:
        with st.spinner("Running the agent pipeline..."):
            st.session_state["result"] = runner(query)

    if "result" in st.session_state:
        result = st.session_state["result"]

        if result.get("errors"):
            st.error("Errors were reported during processing:")
            for error in result["errors"]:
                st.error(error)

        tabs = st.tabs(["Report", "Analyses", "Experiments", "World Model"])

        with tabs[0]:
            st.markdown(result.get("report", "No report generated."))
            st.download_button(
                label="Download report (Markdown)",
                data=result.get("report", ""),
                file_name="longevity_report.md",
                mime="text/markdown",
            )

        with tabs[1]:
            analyses = result.get("analyses", [])
            if not analyses:
                st.info("No analyses available.")
            else:
                for analysis in analyses:
                    _render_analysis_block(analysis)
                    st.markdown("---")

        with tabs[2]:
            plans = result.get("experiment_plans", [])
            if not plans:
                st.info("No experiment plans available.")
            else:
                for plan in plans:
                    _render_experiments(plan)
                    st.markdown("---")

        with tabs[3]:
            entities = result.get("entities", {})
            _render_world_model_snapshot(world_model, entities.get("intervention_names", []))

        with st.expander("Execution logs"):
            for log in result.get("logs", []):
                st.text(log)


if __name__ == "__main__":
    main()
