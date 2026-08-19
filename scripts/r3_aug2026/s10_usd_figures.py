#!/usr/bin/env python3
"""STEP 10 -- regenerate Figure 8 (feature importance) and Figure 9 (entity-level
leave-one-out impact) on the PRIMARY USD-notional volume specification.

Background
----------
The manuscript's primary volume statistic is USD-notional turnover
(sum over bars of volume x close, aggregated to daily; `common.py`
volume_basis='usd_close').  The prose, headline tables and Table
`tab:entity_impact` were migrated to that basis; two figures and the
feature-importance table were still rendered from the base-asset-unit build.
This script rebuilds those figure inputs on the USD basis and re-renders both
figures through the *unmodified* published plotters, so the visual style is
byte-for-byte the same code path as the originals.

What is reused, verbatim and unmodified
---------------------------------------
  code/src/market  ................ via common.build_stats_matrix('usd_close')
  code/src/analysis/robustness.py ......... RobustnessChecker.feature_importance
  code/src/analysis/cross_sectional.py .... CrossSectionalAnalyzer
                                            .compute_entity_alignment
  code/scripts/run_expansion_figures.py ... fig8_feature_importance,
                                            fig9_entity_impact
Nothing under code/src/ or code/scripts/run_expansion_figures.py is written to
or altered.  New JSON figure inputs are written to ./figdata_usd/ and the two
PDFs/PNGs to paper/fi-resubmission/figures/.

Which panel of Figure 9 moves to USD, and why not both
------------------------------------------------------
The RIGHT panel (content-verified, n = 43) is the manuscript's primary
estimate -- it is what Table `tab:entity_impact` and the surrounding prose
report -- so it is rebuilt on USD notional.

The LEFT panel (earlier contaminated corpus, n = 37) is deliberately KEPT on
the base-asset build.  It documents a superseded corpus snapshot, and the
manuscript already declares that basis for it in the footnote to the
contamination decomposition ("the figures quoted here are computed on the
base-asset build in which the contaminated corpus was originally analysed").
Recomputing it on USD is a counterfactual nobody claims, and it would put the
figure in direct conflict with three passages of body text:

    base-asset (as published)        USD notional (counterfactual)
    XMR  +0.0200  helps              XMR  +0.0195  helps
    CRV  +0.0133  helps              CRV  +0.0072  neutral
    YFI  +0.0122  helps              YFI  -0.0050  neutral   <- flips sign
    SOL  +0.0104  helps              SOL  +0.0059  neutral
    helpers = XMR, CRV, YFI, SOL     helpers = XMR only

i.e. the cautionary tale the panel exists to illustrate would vanish, and
    - the caption's "several tokens appear to help ... including YFI",
    - the Results' "eliminates all four apparent 'helpers'", and
    - the Discussion's "three of the four apparent helpers (XMR, CRV, SOL)"
would all become false.  The USD recomputation is still performed and stored in
results_10_usd_figures.json under `contaminated_usd_counterfactual` so the
finding is on the record; it is simply not what gets rendered.

Usage:  python code/scripts/r3_aug2026/s10_usd_figures.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import common as C
from analysis.cross_sectional import CrossSectionalAnalyzer
from analysis.robustness import RobustnessChecker

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]                       # .../whitepaper-claims/code
REPO = CODE.parent                           # .../whitepaper-claims
FIGDATA = HERE / "figdata_usd"
FIGOUT = REPO / "paper" / "fi-resubmission" / "figures"
RESULTS = HERE / "results_10_usd_figures.json"

VOLUME_BASIS = "usd_close"          # primary specification (right panel, fig 8)
CONTAM_BASIS = "base"               # left panel: superseded build, see docstring

# Targets established by the Aug-2026 re-run (R3_RERUN_AUG2026.md ss.2.4, s.8).
# The script refuses to write figures that disagree with the manuscript text.
TARGET_FEATURE_IMPORTANCE = {
    "interoperability": 0.015, "privacy": 0.011, "scalability": 0.006,
    "store_of_value": 0.005, "governance": 0.004, "oracle": 0.002,
    "smart_contracts": 0.001, "defi": 0.000, "medium_of_exchange": -0.003,
    "data_storage": -0.007,
}
TARGET_ENTITY_TOP = {"SOL": 0.012, "TRB": 0.011, "SUI": 0.009, "RENDER": 0.009}
TARGET_ENTITY_BOTTOM = {"SC": -0.012, "POL": -0.017, "STORJ": -0.017,
                        "LINK": -0.018}
TARGET_BTC_LOO = 0.005
TARGET_MAX_ABS = 0.018
TARGET_N_BELOW_001 = 37
TARGET_N = 43


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_inputs() -> dict:
    """Recompute both figures' JSON inputs on the USD-notional stats matrix."""
    z, ssym, _ = C.build_stats_matrix(VOLUME_BASIS)
    fac, fsym = C.load_factors()
    cats = C.load_meta(C.OUT / "nlp" / "claims_matrix_meta.json")["categories"]

    cl_clean, sym_clean = C.load_claims(clean=True)
    cl_dirty, sym_dirty = C.load_claims(clean=False)

    # ---- Figure 8: per-category ablation on the verified corpus -------------
    Cl, St, _Fa, common = C.align_three(cl_clean, sym_clean, z, ssym, fac, fsym)
    assert len(common) == TARGET_N, f"clean corpus n={len(common)}, expected {TARGET_N}"

    rc = RobustnessChecker(C.OUT / "analysis")
    fi = rc.feature_importance(Cl, St, cats, C.STATISTICS)
    fig8 = {
        "importance": fi["claims_importance"],
        "phi_full": fi["phi_full"],
        "most_important": fi["most_important"],
        "n": len(common),
        "volume_basis": VOLUME_BASIS,
        "note": ("Per-category ablation on the primary USD-notional turnover "
                 "specification (sum of volume x close per bar, aggregated to "
                 "daily). Supersedes the base-asset-unit build."),
    }

    # ---- Figure 9 right panel: verified corpus, n = 43 ----------------------
    csa = CrossSectionalAnalyzer(C.OUT / "analysis")
    clean_res = csa.compute_entity_alignment(Cl, St, common)
    fig9_clean = {
        "entities": clean_res["entity_analysis"],
        "phi_full": clean_res["phi_full"],
        "best_aligned": clean_res["best_aligned"],
        "worst_aligned": clean_res["worst_aligned"],
        "n": len(common),
        "corpus": "content-verified-43",
        "volume_basis": VOLUME_BASIS,
    }

    # ---- Figure 9 left panel: earlier contaminated corpus, n = 37 -----------
    # Rendered on the base-asset build in which that corpus was originally
    # analysed (see module docstring); the USD recomputation is kept alongside
    # as a documented counterfactual but is NOT what gets plotted.
    zc, ssym_c, _ = C.build_stats_matrix(CONTAM_BASIS)
    Cd, Sd, _Fd, common_d = C.align_three(cl_dirty, sym_dirty, zc, ssym_c, fac, fsym)
    dirty_res = csa.compute_entity_alignment(Cd, Sd, common_d)
    fig9_dirty = {
        "entities": dirty_res["entity_analysis"],
        "phi_full": dirty_res["phi_full"],
        "best_aligned": dirty_res["best_aligned"],
        "worst_aligned": dirty_res["worst_aligned"],
        "n": len(common_d),
        "corpus": "contaminated-37",
        "volume_basis": CONTAM_BASIS,
        "note": ("Superseded corpus snapshot, retained on the base-asset build "
                 "in which it was originally analysed. See "
                 "results_10_usd_figures.json :: contaminated_usd_counterfactual "
                 "for the USD-notional recomputation."),
    }

    Cu, Su, _Fu, common_u = C.align_three(cl_dirty, sym_dirty, z, ssym, fac, fsym)
    dirty_usd = csa.compute_entity_alignment(Cu, Su, common_u)

    FIGDATA.mkdir(parents=True, exist_ok=True)
    (FIGDATA / "feature_importance_plot.json").write_text(
        json.dumps(fig8, indent=2) + "\n")
    (FIGDATA / "entity_impact_plot.json").write_text(
        json.dumps(fig9_clean, indent=2) + "\n")
    (FIGDATA / "entity_impact_plot_contaminated_n37.json").write_text(
        json.dumps(fig9_dirty, indent=2) + "\n")

    return {"fig8": fig8, "fig9_clean": fig9_clean, "fig9_dirty": fig9_dirty,
            "fig9_dirty_usd": {
                "entities": dirty_usd["entity_analysis"],
                "phi_full": dirty_usd["phi_full"],
                "best_aligned": dirty_usd["best_aligned"],
                "worst_aligned": dirty_usd["worst_aligned"],
                "n": len(common_u), "volume_basis": VOLUME_BASIS,
            }}


def verify(built: dict) -> dict:
    """Hard check against the re-run's published target values."""
    fails, report = [], {}

    got_fi = {d["feature"]: d["importance"] for d in built["fig8"]["importance"]}
    for feat, tgt in TARGET_FEATURE_IMPORTANCE.items():
        got = got_fi.get(feat)
        if got is None or abs(round(got, 3) - tgt) > 5e-4:
            fails.append(f"feature importance {feat}: got {got!r}, target {tgt:+.3f}")
    report["feature_importance_3dp"] = {k: round(v, 3) for k, v in got_fi.items()}
    report["feature_importance_max_abs"] = max(abs(v) for v in got_fi.values())
    if report["feature_importance_max_abs"] >= 0.02:
        fails.append("a category moves phi by >= 0.02, contradicting the text")

    ents = built["fig9_clean"]["entities"]
    imp = {e["symbol"]: e["impact"] for e in ents}
    order = sorted(ents, key=lambda e: e["impact"], reverse=True)
    for sym, tgt in {**TARGET_ENTITY_TOP, **TARGET_ENTITY_BOTTOM}.items():
        got = imp.get(sym)
        if got is None or abs(round(got, 3) - tgt) > 5e-4:
            fails.append(f"entity LOO {sym}: got {got!r}, target {tgt:+.3f}")
    if abs(round(imp.get("BTC", 99), 3) - TARGET_BTC_LOO) > 5e-4:
        fails.append(f"BTC LOO: got {imp.get('BTC')!r}, target {TARGET_BTC_LOO:+.3f}")

    top4 = [e["symbol"] for e in order[:4]]
    bot4 = [e["symbol"] for e in order[-4:]]
    if top4 != ["SOL", "TRB", "SUI", "RENDER"]:
        fails.append(f"top-4 by impact is {top4}, expected [SOL, TRB, SUI, RENDER]")
    if bot4 != ["SC", "POL", "STORJ", "LINK"]:
        fails.append(f"bottom-4 by impact is {bot4}, expected [SC, POL, STORJ, LINK]")

    max_abs = max(abs(v) for v in imp.values())
    n_below = sum(1 for v in imp.values() if abs(v) < 0.01)
    if abs(round(max_abs, 3) - TARGET_MAX_ABS) > 5e-4:
        fails.append(f"max |dphi| = {max_abs:.4f}, target {TARGET_MAX_ABS}")
    if n_below != TARGET_N_BELOW_001:
        fails.append(f"{n_below} of {len(imp)} below 0.01, target {TARGET_N_BELOW_001}")

    report.update({
        "n_clean": len(ents),
        "entity_top4": [(e["symbol"], round(e["impact"], 3)) for e in order[:4]],
        "entity_bottom4": [(e["symbol"], round(e["impact"], 3)) for e in order[-4:]],
        "entity_max_abs": max_abs,
        "entity_n_below_0.01": n_below,
        "btc_loo": imp.get("BTC"),
        "clean_helpers_gt_0.01": built["fig9_clean"]["best_aligned"],
        "clean_hurters_lt_-0.01": built["fig9_clean"]["worst_aligned"],
        "phi_full_clean": built["fig9_clean"]["phi_full"],
        "n_contaminated": built["fig9_dirty"]["n"],
        "contaminated_volume_basis": built["fig9_dirty"]["volume_basis"],
        "phi_full_contaminated": built["fig9_dirty"]["phi_full"],
        "contaminated_helpers": built["fig9_dirty"]["best_aligned"],
        "contaminated_hurters": built["fig9_dirty"]["worst_aligned"],
        "contaminated_top5": [(e["symbol"], round(e["impact"], 4))
                              for e in built["fig9_dirty"]["entities"][:5]],
        "failures": fails,
    })

    # The left panel must still reproduce the published base-asset build.
    got = {e["symbol"]: e["impact"] for e in built["fig9_dirty"]["entities"]}
    if built["fig9_dirty"]["best_aligned"] != ["XMR", "CRV", "YFI", "SOL"]:
        fails.append("contaminated panel helpers changed from [XMR, CRV, YFI, SOL]")
    if abs(built["fig9_dirty"]["phi_full"] - 0.24573765481201018) > 1e-9:
        fails.append("contaminated panel phi_full no longer reproduces the "
                     "published base-asset value 0.245738")

    cf = built["fig9_dirty_usd"]
    cf_imp = {e["symbol"]: e["impact"] for e in cf["entities"]}
    report["contaminated_usd_counterfactual"] = {
        "why_not_rendered": (
            "Recomputing the superseded contaminated corpus on USD notional "
            "collapses the helper set from four to one and flips YFI's sign, "
            "contradicting the figure caption, the Results paragraph on the "
            "four apparent helpers, and the Discussion. The panel documents "
            "the earlier build and is retained on that build's basis."),
        "phi_full": cf["phi_full"],
        "helpers": cf["best_aligned"],
        "hurters": cf["worst_aligned"],
        "key_symbols_base_vs_usd": {
            s: {"base": round(got[s], 4), "usd": round(cf_imp[s], 4)}
            for s in ["XMR", "CRV", "YFI", "SOL", "SUI", "BTC", "RPL", "HBAR"]
        },
    }
    return report


def main():
    print(f"volume basis: {VOLUME_BASIS}\n")
    built = build_inputs()
    rep = verify(built)

    print("Figure 8 -- feature importance (USD notional), impact-descending:")
    for d in built["fig8"]["importance"]:
        print(f"  {d['feature']:20s} {d['importance']:+.4f}")
    print(f"  max |impact| = {rep['feature_importance_max_abs']:.4f}\n")

    print(f"Figure 9 right panel -- verified corpus n = {rep['n_clean']}, "
          f"phi_full = {rep['phi_full_clean']:.5f}")
    print(f"  top-4    : {rep['entity_top4']}")
    print(f"  bottom-4 : {rep['entity_bottom4']}")
    print(f"  max |dphi| = {rep['entity_max_abs']:.4f}; "
          f"{rep['entity_n_below_0.01']} of {rep['n_clean']} below 0.01; "
          f"BTC = {rep['btc_loo']:+.4f}")
    print(f"  > +0.01 : {rep['clean_helpers_gt_0.01']}")
    print(f"  < -0.01 : {rep['clean_hurters_lt_-0.01']}\n")

    print(f"Figure 9 left panel -- contaminated corpus n = {rep['n_contaminated']}, "
          f"phi_full = {rep['phi_full_contaminated']:.5f} "
          f"[basis: {rep['contaminated_volume_basis']}, superseded build]")
    print(f"  top-5   : {rep['contaminated_top5']}")
    print(f"  > +0.01 : {rep['contaminated_helpers']}")
    print(f"  < -0.01 : {rep['contaminated_hurters']}")
    cf = rep["contaminated_usd_counterfactual"]
    print(f"  USD counterfactual (recorded, NOT rendered): "
          f"helpers would be {cf['helpers']}, YFI "
          f"{cf['key_symbols_base_vs_usd']['YFI']['base']:+.4f} -> "
          f"{cf['key_symbols_base_vs_usd']['YFI']['usd']:+.4f}\n")

    if rep["failures"]:
        print("TARGET CHECK FAILED -- figures NOT written:")
        for f in rep["failures"]:
            print("  - " + f)
        RESULTS.write_text(json.dumps(rep, indent=2) + "\n")
        raise SystemExit(1)
    print("target check: PASS (all re-run values reproduced)\n")

    figs = _load("run_expansion_figures", CODE / "scripts" / "run_expansion_figures.py")
    FIGOUT.mkdir(parents=True, exist_ok=True)
    figs.fig8_feature_importance(FIGDATA, FIGOUT)
    figs.fig9_entity_impact(FIGDATA, FIGOUT)

    RESULTS.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"\nwrote {FIGOUT / 'fig8_feature_importance.pdf'}")
    print(f"wrote {FIGOUT / 'fig9_entity_impact.pdf'}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
