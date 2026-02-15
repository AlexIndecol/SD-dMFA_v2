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
    c = load_yaml(CFG/"coupling.yml")
    ex = c["exchange"]

    def fmt_vars(items):
        vs = [x["var"] for x in items]
        # keep diagram readable: show up to ~8 vars then +n
        if len(vs) > 8:
            return "\n".join(vs[:8] + [f"+{len(vs)-8} more"])
        return "\n".join(vs)

    sd_to_dmfa = fmt_vars(ex.get("sd_to_dmfa", []))
    dmfa_to_sd = fmt_vars(ex.get("dmfa_to_sd", []))
    trade_out = fmt_vars(ex.get("trade_outputs", []))

    mermaid = f"""%% Auto-generated: do not edit by hand.
%% Source: configs/coupling.yml

flowchart LR
  sd[SD (BPTK-Py)]
  dmfa[dMFA (flodym)]
  trade[OD Trade]
  ind[Indicators]

  sd -->|"{sd_to_dmfa}"| dmfa
  dmfa -->|"{dmfa_to_sd}"| sd
  trade -->|"{trade_out}"| sd
  sd --> ind
  dmfa --> ind
  trade --> ind
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(mermaid, encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
