#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import csv, json, re, subprocess, math

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "main.tex"
PDF = ROOT / "main.pdf"
SUMMARY = ROOT / "tables" / "run_summary.csv"
FRONTIER = ROOT / "tables" / "table_frontier_thresholds.csv"
SIMVAL = ROOT / "tables" / "table_simulation_validation.csv"
MATLAB = ROOT / "code" / "generate_all_outputs_matlab.m"


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def maybe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    matlab = MATLAB.read_text(encoding="utf-8")
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("main_tex_exists", TEX.exists(), str(TEX))
    add("main_pdf_exists", PDF.exists(), str(PDF))
    add("matlab_exists", MATLAB.exists(), str(MATLAB))
    add("no_visible_v7_in_manuscript", "v7" not in tex.lower(), "literal `v7` absent from manuscript source")
    add("bibliography_points_to", "\\bibliography{refs}" in tex, "refs.bib declared")

    figure_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    figure_paths = []
    for item in figure_names:
        p = Path(item)
        if not p.suffix:
            p = p.with_suffix('.pdf')
        if not p.is_absolute():
            if not str(p).startswith('figures/'):
                p = Path('figures') / p
            p = ROOT / p
        figure_paths.append(p)
    missing_figs = [str(p.relative_to(ROOT)) for p in figure_paths if not p.exists()]
    add("all_manuscript_figures_exist", not missing_figs, f"referenced={len(figure_paths)}, missing={missing_figs}")

    table_inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    input_paths = [ROOT / p for p in table_inputs]
    missing_tabs = [str(p.relative_to(ROOT)) for p in input_paths if not p.exists()]
    add("all_manuscript_table_inputs_exist", not missing_tabs, f"referenced={len(input_paths)}, missing={missing_tabs}")

    summary_rows = parse_csv(SUMMARY)
    add("run_summary_has_single_row", len(summary_rows) == 1, f"rows={len(summary_rows)}")
    if summary_rows:
        row = summary_rows[0]
        resid_public = maybe_float(row["residual_public"])
        resid_h = maybe_float(row["residual_VH"])
        resid_l = maybe_float(row["residual_VL"])
        figs = int(float(row["figures_generated"]))
        add("public_residual_below_1e-7", resid_public < 1e-7, f"{resid_public:.12g}")
        add("high_type_residual_below_1e-7", resid_h < 1e-7, f"{resid_h:.12g}")
        add("low_type_residual_below_1e-7", resid_l < 1e-7, f"{resid_l:.12g}")
        add("figure_panel_count_equals_45", figs == 45, f"figures_generated={figs}")

    sim_rows = parse_csv(SIMVAL)
    add("simulation_validation_has_two_types", len(sim_rows) == 2, f"rows={len(sim_rows)}")
    if len(sim_rows) == 2:
        types = {r["type"] for r in sim_rows}
        add("simulation_validation_type_labels", types == {"High type", "Low type"}, f"types={sorted(types)}")

    frontier_rows = parse_csv(FRONTIER)
    add("frontier_threshold_table_has_six_rows", len(frontier_rows) == 6, f"rows={len(frontier_rows)}")
    add("frontier_table_uses_exact_test_language", any("exact" in r["frontier"].lower() or "exact" in r["interpretation"].lower() for r in frontier_rows), "diagnostic wording present")

    add("matlab_targets_tables", "table_simulation_validation_matlab.csv" in matlab and "table_frontier_thresholds_matlab.csv" in matlab, "extended validation and frontier outputs declared")
    add("matlab_uses_same_baseline_primitives", all(token in matlab for token in ["par.thetaH = 0.92", "par.thetaL = 0.62", "par.kappa = 1.35", "par.gamma = 0.82", "par.L = 0.42", "par.delta = 0.92"]), "baseline primitives found")

    pages = None
    try:
        out = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
        m = re.search(r"^Pages:\s+(\d+)", out, flags=re.MULTILINE)
        if m:
            pages = int(m.group(1))
    except Exception:
        pass
    add("pdf_page_count_at_least_20", pages is not None and pages >= 20, f"pages={pages}")

    status = all(bool(c["passed"]) for c in checks)
    report = {
        "status": "PASS" if status else "FAIL",
        "root": str(ROOT),
        "checks": checks,
    }
    (ROOT / "tables" / "consistency_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# POM_agentic_ai consistency audit",
        "",
        f"**Overall status:** {'PASS' if status else 'FAIL'}",
        "",
        "This audit treats the equation -> code -> table/figure -> manuscript chain as a hard submission constraint. It checks the final manuscript references, numerical summary thresholds, exact policy-frontier files, and MATLAB cross-check declarations.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for c in checks:
        md.append(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {str(c['detail']).replace('|','/')} |")
    md.extend([
        "",
        "## Interpretation",
        "",
        "- A PASS means the final package is internally wired: all manuscript-facing figures and tables exist, solver residuals meet the audit threshold, and the MATLAB script declares the same extended validation/frontier families as the manuscript package.",
        "- MATLAB numerical execution is not claimed inside this sandbox because a MATLAB runtime is unavailable here; the audit verifies script structure, filenames, primitives, and the Python-generated submission outputs that the manuscript actually includes.",
    ])
    (ROOT / "review" / "consistency_audit_CN.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status else 1

if __name__ == "__main__":
    raise SystemExit(main())
