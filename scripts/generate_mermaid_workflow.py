"""Generate Mermaid workflow diagram from configs/coupling.yml.

Writes: docs/diagrams/workflow.mmd
"""

from pathlib import Path
import yaml
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
OUT = ROOT / "docs" / "diagrams" / "workflow.mmd"

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    c = load_yaml(CFG / "coupling.yml")
    ex = c["exchange"]

    n_sd_dmfa = len(ex.get("sd_to_dmfa", []))
    n_dmfa_sd = len(ex.get("dmfa_to_sd", []))
    n_trade_out = len(ex.get("trade_outputs", []))

    # Keep syntax conservative for older Mermaid renderers.
    mermaid = textwrap.dedent(
        f"""\
        %% Auto-generated: do not edit by hand.
        %% Source: configs/coupling.yml
        %% sd_to_dmfa vars: {n_sd_dmfa}
        %% dmfa_to_sd vars: {n_dmfa_sd}
        %% trade_outputs vars: {n_trade_out}

        graph LR
          SD[SD]
          DMFA[dMFA]
          TRADE[OD Trade]
          IND[Indicators]

          SD -->|sd_to_dmfa| DMFA
          DMFA -->|dmfa_to_sd| SD
          TRADE -->|trade_outputs| SD
          SD --> IND
          DMFA --> IND
          TRADE --> IND
        """
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(mermaid, encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
