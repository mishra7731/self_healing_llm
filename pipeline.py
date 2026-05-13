import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
 
import yaml
 
from probe import GarakRunner, ProbeResult
from patch.prompt_patch import build_from_config as build_prompt_patch
from patch.output_patch import build_from_config as build_output_patch


# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Config loading
 
def load_config(config_path: str) -> dict:
    """
    Load and return the experiment configuration from a YAML file.
 
    Args:
        config_path: Path to the YAML configuration file.
 
    Returns:
        Parsed configuration as a nested dictionary.
 
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from: %s", config_path)
    return cfg


# Result comparison and reporting
 
def compare_results(baseline: ProbeResult, patched: ProbeResult) -> dict:
    """
    Compare baseline and patched ProbeResults and compute improvement metrics.
 
    Args:
        baseline: ProbeResult from the unpatched model run.
        patched:  ProbeResult from the patched model run.
 
    Returns:
        Dictionary containing comparison metrics and qualitative summary.
    """
    failure_delta = baseline.total_failures - patched.total_failures
    rate_delta = baseline.failure_rate - patched.failure_rate
    improvement_pct = (failure_delta / baseline.total_failures * 100) if baseline.total_failures > 0 else 0.0
 
    return {
        "model": baseline.model_name,
        "patch_config": patched.patch_config,
        "baseline_failures": baseline.total_failures,
        "baseline_attempts": baseline.total_attempts,
        "baseline_failure_rate": round(baseline.failure_rate, 4),
        "patched_failures": patched.total_failures,
        "patched_attempts": patched.total_attempts,
        "patched_failure_rate": round(patched.failure_rate, 4),
        "failure_reduction": failure_delta,
        "rate_reduction": round(rate_delta, 4),
        "improvement_percent": round(improvement_pct, 2),
    }
    
def save_results(results: list[dict], output_dir: str, label: str = "") -> str:
    """
    Save experiment results to a JSON file.
 
    Args:
        results:    List of comparison result dicts.
        output_dir: Directory to write the results file.
        label:      Optional label appended to the filename.
 
    Returns:
        Path to the saved results file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{label}_{timestamp}.json" if label else f"results_{timestamp}.json"
    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to: %s", filepath)
    return str(filepath)

def print_summary(comparison: dict) -> None:
    """
    Print a human-readable summary of a baseline vs patched comparison.
 
    Args:
        comparison: Output of compare_results().
    """
    print("\n" + "=" * 60)
    print(f"  Model        : {comparison['model']}")
    print(f"  Patch config : {comparison['patch_config']}")
    print(f"  Baseline     : {comparison['baseline_failures']}/{comparison['baseline_attempts']} "
          f"failed ({comparison['baseline_failure_rate']*100:.1f}%)")
    print(f"  Patched      : {comparison['patched_failures']}/{comparison['patched_attempts']} "
          f"failed ({comparison['patched_failure_rate']*100:.1f}%)")
    print(f"  Improvement  : {comparison['improvement_percent']:+.1f}% fewer failures")
    print("=" * 60 + "\n")
    
    
# Patch configuration helpers
 
def get_patch_config_overrides(cfg: dict, prompt_enabled: bool, output_enabled: bool) -> dict:
    """
    Build a modified config dict with specific patch flags overridden.
 
    This is used by the ablation study to toggle individual patch layers.
 
    Args:
        cfg:             Base experiment config dict.
        prompt_enabled:  Whether to enable prompt-level patch.
        output_enabled:  Whether to enable output-level patch.
 
    Returns:
        Modified config dict (deep copy with overrides applied).
    """
    import copy
    modified = copy.deepcopy(cfg)
    modified["patches"]["prompt_level"]["enabled"] = prompt_enabled
    modified["patches"]["output_level"]["enabled"] = output_enabled
    return modified


# Pipeline stages
 
def run_probe_stage(runner: GarakRunner, model_key: str, patch_label: str) -> ProbeResult:
    """
    Run the PROBE stage: invoke Garak and collect vulnerability results.
 
    Args:
        runner:      Configured GarakRunner instance.
        model_key:   Key into cfg['models'] for the target model.
        patch_label: Label describing the current patch configuration.
 
    Returns:
        ProbeResult with vulnerability statistics.
    """
    logger.info("[PROBE] Running Garak | model=%s | config=%s", model_key, patch_label)
    return runner.run(model_key=model_key, patch_label=patch_label)
 
 
def run_patch_stage(cfg: dict) -> tuple:
    """
    Run the PATCH stage: initialise prompt and output patch objects.
 
    Args:
        cfg: Experiment config dict (may have patched enabled/disabled).
 
    Returns:
        Tuple of (PromptPatch, OutputPatch) — either or both may have
        their mechanisms disabled depending on the config.
    """
    logger.info("[PATCH] Initialising mitigation mechanisms...")
    prompt_patch = build_prompt_patch(cfg)
    output_patch = build_output_patch(cfg)
    return prompt_patch, output_patch

def run_verify_stage(runner: GarakRunner, model_key: str, patch_label: str) -> ProbeResult:
    """
    Run the VERIFY stage: re-probe the model after patches are applied.
 
    In this prototype the patches are applied at the pipeline layer (wrapping
    the generator), not inside Garak itself. This stage represents re-running
    the same probe suite to check for improvement.
 
    Args:
        runner:      Configured GarakRunner instance.
        model_key:   Key into cfg['models'] for the target model.
        patch_label: Label describing the current patch configuration.
 
    Returns:
        ProbeResult with post-patch vulnerability statistics.
    """
    logger.info("[VERIFY] Re-running Garak with patches active | model=%s", model_key)
    return runner.run(model_key=model_key, patch_label=patch_label)

# Full pipeline
 
def run_pipeline(cfg: dict, model_key: str) -> list[dict]:
    """
    Execute the full probe → patch → verify pipeline for one model.
 
    If ablation is enabled in config, runs four configurations:
      no_patches, prompt_only, output_only, both_patches.
    Otherwise runs a single baseline → patched comparison.
 
    Args:
        cfg:       Full experiment config dict.
        model_key: Key into cfg['models'] for the target model.
 
    Returns:
        List of comparison result dicts (one per configuration).
    """
    report_dir = cfg.get("garak", {}).get("report_dir", "./results/garak_reports")
    output_dir = cfg.get("experiment", {}).get("output_dir", "./results")
    ablation_cfg = cfg.get("experiment", {}).get("ablation", {})
    run_ablation = ablation_cfg.get("run", False)
 
    runner = GarakRunner(cfg, report_dir=report_dir)
    comparisons = []
 
    if run_ablation:
        ablation_configs = ablation_cfg.get("configurations", [])
        logger.info("Running ablation study with %d configurations...", len(ablation_configs))
 
        # Running baseline first (no patches) as the reference point
        logger.info("--- Ablation: baseline (no patches) ---")
        baseline = run_probe_stage(runner, model_key, "no_patches")
 
        for abl_cfg in ablation_configs:
            name = abl_cfg.get("name", "unknown")
            prompt_on = abl_cfg.get("prompt_level", False)
            output_on = abl_cfg.get("output_level", False)
 
            if name == "no_patches":
                # Already ran this as baseline
                continue
 
            logger.info("--- Ablation: %s ---", name)
            modified_cfg = get_patch_config_overrides(cfg, prompt_on, output_on)
            _, _ = run_patch_stage(modified_cfg)  # init patches, logged for audit
            patched = run_probe_stage(runner, model_key, name)
            comparison = compare_results(baseline, patched)
            comparisons.append(comparison)
            print_summary(comparison)
 
    else:
        # Simple single probe → patch → verify cycle
        logger.info("--- Stage 1: Baseline probe (no patches) ---")
        baseline = run_probe_stage(runner, model_key, "baseline")
 
        logger.info("--- Stage 2: Patch initialisation ---")
        _, _ = run_patch_stage(cfg)
 
        logger.info("--- Stage 3: Verify (patches active) ---")
        patched = run_verify_stage(runner, model_key, "patched")
 
        comparison = compare_results(baseline, patched)
        comparisons.append(comparison)
        print_summary(comparison)
 
    # Save results
    model_name = cfg["models"][model_key]["name"].replace("/", "_")
    save_results(comparisons, output_dir, label=model_name)
    return comparisons

# CLI entry point
 
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Self-Healing LLM Security Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/experiment.yaml",
        help="Path to the experiment YAML config file (default: config/experiment.yaml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run a single model by key (e.g. 'llama'). Overrides active_models in config.",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Disable ablation study; run a single baseline→patched comparison.",
    )
    return parser.parse_args()

def main() -> None:
    """Main entry point for the self-healing LLM security pipeline."""
    args = parse_args()
 
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error("Failed to load config: %s", e)
        sys.exit(1)
 
    if args.no_ablation:
        cfg.setdefault("experiment", {}).setdefault("ablation", {})["run"] = False
 
    active_models = [args.model] if args.model else cfg.get("active_models", [])
    if not active_models:
        logger.error("No active models specified. Set 'active_models' in config or use --model.")
        sys.exit(1)
 
    all_results = []
    for model_key in active_models:
        if model_key not in cfg.get("models", {}):
            logger.warning("Model key '%s' not found in config. Skipping.", model_key)
            continue
        logger.info("========== Starting pipeline for model: %s ==========", model_key)
        results = run_pipeline(cfg, model_key)
        all_results.extend(results)
 
    logger.info("Pipeline complete. Total comparisons: %d", len(all_results))
    
    
if __name__ == "__main__":
    main()