"""
CRAWL Phase: Decision Gate Summary

Evaluates all 4 CRAWL test results and makes final GO/NO-GO decision
for proceeding to WALK phase.

Usage:
    python src/crawl/crawl_decision_gate.py
"""

import json
from datetime import datetime
from pathlib import Path

from src.utils import create_decision_gate_summary, get_config, save_figure


def load_test_results(config):
    """
    Load results from all 4 CRAWL tests.

    Returns:
        Dict with test results
    """
    results_dir = config.get_path("paths.results_dir") / "experiments"

    test_files = {
        "Test 1: Separability": results_dir / "crawl_test_1_results.json",
        "Test 2: Temporal Signal": results_dir / "crawl_test_2_results.json",
        "Test 3: Generalization": results_dir / "crawl_test_3_results.json",
        "Test 4: Minimal Model": results_dir / "crawl_test_4_results.json",
    }

    results = {}

    for name, filepath in test_files.items():
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
                results[name] = data
        else:
            print(f"  WARNING: {name} results not found at {filepath}")
            results[name] = None

    return results


def evaluate_decision_gate(test_results, config):
    """
    Evaluate all test results and make GO/NO-GO decision.

    Args:
        test_results: Dict with results from all 4 tests
        config: Config instance

    Returns:
        dict with decision gate evaluation
    """
    print("=" * 80)
    print("CRAWL PHASE: DECISION GATE")
    print("=" * 80)
    print("\nEvaluating all test results...\n")

    thresholds = config.crawl_thresholds

    # Test 1: Separability
    test1 = test_results.get("Test 1: Separability")
    if test1:
        # Use ROC-AUC as primary metric
        roc_auc = test1["decision"]["roc_auc"]
        threshold = test1["decision"]["roc_auc_threshold"]
        passed1 = test1["decision"]["passed"]
        print(f"✓ Test 1: Separability")
        print(f"    ROC-AUC: {roc_auc:.3f} (required ≥{threshold:.3f})")
        print(f"    Status: {'PASS' if passed1 else 'FAIL'}")
    else:
        passed1 = False
        print(f"✗ Test 1: Separability - MISSING")

    # Test 2: Temporal Signal
    test2 = test_results.get("Test 2: Temporal Signal")
    if test2:
        p_value = test2["decision"]["p_value"]
        threshold = test2["decision"]["threshold"]
        passed2 = test2["decision"]["passed"]
        print(f"\n✓ Test 2: Temporal Signal")
        print(f"    p-value: {p_value:.6f} (required <{threshold:.3f})")
        print(f"    Status: {'PASS' if passed2 else 'FAIL'}")
    else:
        passed2 = False
        print(f"\n✗ Test 2: Temporal Signal - MISSING")

    # Test 3: Generalization
    test3 = test_results.get("Test 3: Generalization")
    if test3:
        cv = test3["decision"]["cv"]
        threshold = test3["decision"]["threshold"]
        passed3 = test3["decision"]["passed"]
        print(f"\n✓ Test 3: Generalization")
        print(f"    CV: {cv:.3f} (required <{threshold:.3f})")
        print(f"    Status: {'PASS' if passed3 else 'FAIL'}")
    else:
        passed3 = False
        print(f"\n✗ Test 3: Generalization - MISSING")

    # Test 4: Minimal Model
    test4 = test_results.get("Test 4: Minimal Model")
    if test4:
        auc = test4["decision"]["auc"]
        threshold = test4["decision"]["threshold"]
        passed4 = test4["decision"]["passed"]
        status = test4["decision"]["status"]
        print(f"\n✓ Test 4: Minimal Model")
        print(f"    AUC: {auc:.3f} (required ≥{threshold:.3f})")
        print(f"    Status: {status}")
    else:
        passed4 = False
        print(f"\n✗ Test 4: Minimal Model - MISSING")

    # Overall decision
    all_passed = passed1 and passed2 and passed3 and passed4

    print("\n" + "=" * 80)
    print("FINAL DECISION")
    print("=" * 80)

    if all_passed:
        decision = "GO TO WALK PHASE"
        color = "green"
        print(f"\n✓✓✓ {decision} ✓✓✓")
        print(f"\nAll 4 CRAWL tests passed!")
        print(f"\nKey Findings:")
        if test1:
            print(f"  • AlphaEarth embeddings distinguish cleared from intact forest (ROC-AUC: {roc_auc:.3f})")
        if test2:
            print(f"  • Embeddings show temporal signal before clearing (p<0.000001)")
        if test3:
            print(f"  • Signal is consistent across regions (CV: {cv:.3f})")
        if test4:
            print(f"  • Strong predictive signal with just 2 features (AUC: {auc:.3f})")
        print(f"\nConclusion:")
        print(f"  The approach is fundamentally sound. Proceed with confidence to")
        print(f"  WALK phase to build a robust, validated model.")
    else:
        decision = "STOP OR PIVOT"
        color = "red"
        print(f"\n✗✗✗ {decision} ✗✗✗")
        print(f"\nOne or more CRAWL tests failed.")
        print(f"\nFailed tests:")
        if not passed1:
            print(f"  • Test 1: Separability")
        if not passed2:
            print(f"  • Test 2: Temporal Signal")
        if not passed3:
            print(f"  • Test 3: Generalization")
        if not passed4:
            print(f"  • Test 4: Minimal Model")
        print(f"\nRecommendation: Review failed tests before proceeding.")

    print("=" * 80 + "\n")

    # Prepare summary for visualization
    viz_results = {}

    if test1:
        viz_results["Test 1: Separability"] = {
            "passed": passed1,
            "value": roc_auc,
            "threshold": threshold,
        }

    if test2:
        viz_results["Test 2: Temporal Signal"] = {
            "passed": passed2,
            "value": p_value,
            "threshold": threshold,
        }

    if test3:
        viz_results["Test 3: Generalization"] = {
            "passed": passed3,
            "value": cv,
            "threshold": threshold,
        }

    if test4:
        viz_results["Test 4: Minimal Model"] = {
            "passed": passed4,
            "value": auc,
            "threshold": threshold,
        }

    return {
        "decision": decision,
        "all_passed": all_passed,
        "individual_results": {
            "test_1": {"passed": passed1, "status": test1["decision"]["status"] if test1 else "MISSING"},
            "test_2": {"passed": passed2, "status": test2["decision"]["status"] if test2 else "MISSING"},
            "test_3": {"passed": passed3, "status": test3["decision"]["status"] if test3 else "MISSING"},
            "test_4": {"passed": passed4, "status": test4["decision"]["status"] if test4 else "MISSING"},
        },
        "viz_results": viz_results,
    }


def run_decision_gate(save_results=True):
    """
    Run CRAWL decision gate evaluation.

    Args:
        save_results: Whether to save results to disk

    Returns:
        dict with decision gate results
    """
    print("\n" + "=" * 80)
    print("CRAWL PHASE: DECISION GATE EVALUATION")
    print("=" * 80)
    print()

    # Load config
    config = get_config()

    # Load test results
    print("Loading test results...")
    test_results = load_test_results(config)
    print(f"  Loaded {len([r for r in test_results.values() if r])} / 4 test results\n")

    # Evaluate decision gate
    decision_results = evaluate_decision_gate(test_results, config)

    # Create visualization
    if len(decision_results["viz_results"]) > 0:
        print("Generating decision gate summary visualization...")
        fig = create_decision_gate_summary(
            test_results=decision_results["viz_results"],
        )

        # Save figure
        save_figure(fig, "crawl_decision_gate.png", subdir="crawl")

    # Save results
    if save_results:
        results_dir = config.get_path("paths.results_dir") / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "crawl_decision_gate.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision_results["decision"],
            "all_passed": decision_results["all_passed"],
            "individual_results": decision_results["individual_results"],
        }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nDecision gate results saved to: {results_file}")

    return decision_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRAWL Decision Gate Evaluation")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )

    args = parser.parse_args()

    # Run decision gate
    results = run_decision_gate(save_results=not args.no_save)

    # Exit with appropriate code
    exit_code = 0 if results["all_passed"] else 1
    exit(exit_code)
