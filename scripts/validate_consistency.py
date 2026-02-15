from __future__ import annotations
from pathlib import Path
import re
import sys
import yaml


def scan_yaml_markers(rootp: Path) -> list[str]:
    issues = []
    for yml in list(rootp.rglob("*.yml")) + list(rootp.rglob("*.yaml")):
        try:
            txt = yml.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            if line.strip() in {"...","---"}:
                issues.append(f"{yml.relative_to(rootp)}:{i}: contains YAML document marker '{line.strip()}' (remove for clarity).")
    return issues

ALLOWED_TOKENS = {"eps"}
WEIGHT_RE = re.compile(r"^w\d+$")

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(root: str = ".") -> int:
    rootp = Path(root)
    reg = load_yaml(rootp / "registry" / "variable_registry.yml")
    var_names = {v["name"] for v in reg.get("variables", []) if v.get("name")}
    ind = load_yaml(rootp / "configs" / "indicators.yml")
    ind_names = {i["name"] for i in ind.get("indicators", []) if i.get("name")}
    indicators = ind.get("indicators", [])

    unknown = []
    for it in indicators:
        for r in (it.get("requires") or []):
            if r in var_names or r in ind_names:
                continue
            if r in ALLOWED_TOKENS or WEIGHT_RE.match(str(r)):
                continue
            unknown.append((it.get("name"), r))

    # Scenario index file checks
    sc_idx = load_yaml(rootp / "configs" / "scenarios" / "index.yml")
    scen = sc_idx.get("scenarios", {})
    sc_issues = []
    for sid, f in scen.items():
        fp = rootp / f
        if not fp.exists():
            sc_issues.append(f"[missing] {sid}: {f}")
            continue
        y = load_yaml(fp)
        if y.get("id") and y["id"] != sid:
            sc_issues.append(f"[id_mismatch] {sid}: yaml id={y.get('id')} file={f}")

    yaml_marker_issues = scan_yaml_markers(rootp)

    print("== Consistency check ==")
    print(f"Variables in registry: {len(var_names)}")
    print(f"Indicators defined:     {len(ind_names)}")
    if unknown:
        print("\nUnknown indicator requirements (not in registry or indicators):")
        for n, r in unknown[:200]:
            print(f"- {n} requires {r}")
        if len(unknown) > 200:
            print(f"... ({len(unknown)-200} more)")
    else:
        print("\nAll indicator requirements resolve (registry or indicator names).")

    if yaml_marker_issues:
        print("\nYAML document marker issues:")
        for s in yaml_marker_issues[:200]:
            print("-", s)
        if len(yaml_marker_issues) > 200:
            print(f"... ({len(yaml_marker_issues)-200} more)")

    if sc_issues:
        print("\nScenario registry issues:")
        for s in sc_issues:
            print("-", s)
    else:
        print("\nScenario registry OK.")

    # Fail if unknown requirements or missing scenario files
    if unknown or yaml_marker_issues or any(i.startswith("[missing]") for i in sc_issues):
        return 2
    return 0

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    raise SystemExit(main(root))
