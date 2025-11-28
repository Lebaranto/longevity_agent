import argparse

from agent_system import create_agent_runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the longevity agent against a query.")
    parser.add_argument("query", type=str, nargs="?", help="Question or intervention to analyze")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model to use")
    parser.add_argument("--ontology", default="ontology.yaml", help="Path to the ontology YAML file")
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    run_agent = create_agent_runner(world_model_path=args.ontology, llm_model=args.model, temperature=0.0)
    result = run_agent(args.query)

    if result.get("errors"):
        print("Errors:")
        for error in result["errors"]:
            print(f"- {error}")

    print(result.get("report", "No report generated."))


if __name__ == "__main__":
    main()