.PHONY: validate diagrams

validate:
	python scripts/validate_exogenous_inputs.py

diagrams:
	python scripts/generate_mermaid_workflow.py
	@echo "Mermaid source updated: docs/diagrams/workflow.mmd"
