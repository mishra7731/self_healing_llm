import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime
import yaml
 
logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """
    Structured summary of a single Garak probe run.
 
    Attributes:
        model_name:       Name of the model that was probed.
        probe_names:      List of Garak probes that were run.
        total_attempts:   Total number of adversarial prompts sent.
        total_failures:   Number of prompts where the model failed (unsafe output).
        failure_rate:     Fraction of attempts that failed (0.0–1.0).
        failures_by_probe: Failure counts broken down per probe.
        report_path:      Path to the raw Garak JSONL report file.
        patch_config:     Label describing which patches were active.
    """
    model_name: str
    probe_names: list
    total_attempts: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    failures_by_probe: dict = field(default_factory=dict)
    report_path: Optional[str] = None
    patch_config: str = "no_patches"
 
 
class GarakRunner:
    """
    Wrapper around the Garak CLI for running LLM vulnerability probes.
 
    Instead of using deprecated Garak CLI flags, this class generates a
    proper Garak YAML config file for each run and invokes:
        python -m garak --config <generated_config.yaml>
 
    This approach:
      - Avoids all deprecated --model_type / --model_name flags
      - Supports system_prompt injection for patched runs
      - Uses the openai.OpenAICompatible generator (works with Ollama v0.1.24+)
      - Produces structured ProbeResult objects for comparison
 
    Typical usage:
        runner = GarakRunner(cfg, report_dir="./results/garak_reports")
 
        # Baseline (no patches)
        baseline = runner.run("llama", patch_label="baseline")
 
        # Patched (with system prompt guardrail)
        patched = runner.run("llama", patch_label="patched",
                             system_prompt="You are a safe AI...")
    """
 
    def __init__(self, cfg: dict, report_dir: str = "./results/garak_reports"):
        """
        Initialise the GarakRunner.
 
        Args:
            cfg:        Full experiment config dict (parsed from YAML).
            report_dir: Directory where Garak JSONL reports will be stored.
        """
        self.cfg = cfg
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
 
        self.probes: list = cfg.get("garak", {}).get("probes", ["dan.Ablation_Dan_11_0"])
        self.generations: int = cfg.get("garak", {}).get("generations", 1)
 
    def _write_garak_config(
        self,
        model_name: str,
        report_prefix: str,
        patch_label: str = "run",
        system_prompt: str = "",
        timeout: int = 120,
        max_tokens: int = 100,
    ) -> str:
        """
        Write a Garak YAML config file for a single probe run.
 
        Uses the openai.OpenAICompatible generator pointing at Ollama's
        /v1 endpoint — this is the approach confirmed working in our experiments.
 
        Args:
            model_name:    Ollama model name (e.g. 'llama3.2', 'mistral').
            report_prefix: Absolute path prefix for Garak's output files.
            system_prompt: Optional safety system prompt (for patched runs).
            timeout:       Per-request timeout in seconds.
            max_tokens:    Maximum tokens to generate per response.
 
        Returns:
            Path to the written temporary YAML config file.
        """
        probe_spec = ",".join(self.probes)
 
        garak_cfg = {
            "system": {
                "verbose": 0,
                "narrow_output": False,
            },
            "run": {
                "generations": self.generations,
                "deprefix": True,
            },
            "plugins": {
                "target_type": "openai.OpenAICompatible",
                "target_name": model_name,
                "probe_spec": probe_spec,
                "detector_spec": "auto",
                "extended_detectors": False,
                "generators": {
                    "openai": {
                        "OpenAICompatible": {
                            "uri": "http://127.0.0.1:11434/v1",
                            "api_key": "ollama",
                            "max_tokens": max_tokens,
                            "timeout": timeout,
                        }
                    }
                },
            },
            "reporting": {
                "report_prefix": report_prefix,
            },
        }
 
        # Inject system prompt for patched runs
        if system_prompt:
            garak_cfg["run"]["system_prompt"] = system_prompt
 
        # Write to a temp file in the report dir
        tmp_path = self.report_dir / f"_tmp_garak_{model_name.replace('.', '_')}_{patch_label}.yaml"
        #tmp_path = self.report_dir / f"_tmp_garak_{model_name.replace('.', '_')}.yaml"
        with open(tmp_path, "w") as f:
            yaml.dump(garak_cfg, f, default_flow_style=False, allow_unicode=True)
 
        logger.debug("Garak config written to: %s", tmp_path)
        return str(tmp_path)
    
    
    def _build_command(self, config_path: str) -> list[str]:
        """
        Construct the Garak CLI command for a given model and probe set.
 
        Args:
            config_path: Path to the Garak YAML config file.
 
        Returns:
            List of command-line tokens ready for subprocess.
        """
        #probe_arg = ",".join(self.probes)
        cmd = [
            sys.executable, "-m", "garak",
            "--config", config_path,
        ]
        logger.debug("Garak command: %s", " ".join(cmd))
        return cmd
 
    def run(
        self,
        model_key: str,
        patch_label: str = "baseline",
        system_prompt: str = "",
        env_overrides: Optional[dict] = None,
        timeout: int = 120,
    ) -> ProbeResult:
        """
        Run Garak against the specified model and return structured results.

        Steps:
          1. Look up model config from experiment.yaml
          2. Write a temporary Garak YAML config
          3. Launch Garak as a subprocess
          4. Find and parse the resulting JSONL report
          5. Return a structured ProbeResult
 
        Args:
            model_key:     Key into cfg['models'] (e.g. 'llama', 'mistral').
            patch_label:   Label for this run (e.g. 'baseline', 'patched',
                           'prompt_only'). Used in output filenames.
            system_prompt: Safety system prompt to inject (empty = no patch).
            env_overrides: Optional extra environment variables for subprocess.
            timeout:       Per-request timeout passed to the Ollama generator.
        
        Returns:
            ProbeResult with aggregated vulnerability statistics.
        """
        model_cfg = self.cfg["models"][model_key]
        model_name = model_cfg["name"]
        #generator = model_cfg["generator"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_prefix = str(
            self.report_dir.resolve() / f"{model_name.replace('.', '_')}_{patch_label}_{timestamp}"
        )
 
        logger.info(
            "Starting Garak probe | model=%s | patch=%s | probes=%s | system_prompt=%s",
            model_name, patch_label, self.probes,
            "YES" if system_prompt else "NO",
        )
        
        # Write config and build command
        config_path = self._write_garak_config(
            model_name=model_name,
            report_prefix=report_prefix,
            patch_label=patch_label,
            system_prompt=system_prompt,
            timeout=timeout,
        )

        cmd = self._build_command(config_path)
        env = {**os.environ, **(env_overrides or {})}
 
        # Run Garak subprocess
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                env=env,
                timeout=7200,  # 2-hour hard limit per run
            )
            if proc.returncode != 0:
                logger.warning(
                    "Garak exited with code %d for model '%s'",
                    proc.returncode, model_name
                )
        except subprocess.TimeoutExpired:
            logger.error("Garak probe timed out for model '%s'", model_name)
            return ProbeResult(
                model_name=model_name,
                probe_names=self.probes,
                patch_config=patch_label,
            )
        except FileNotFoundError:
            logger.error("Garak not found. Run: pip install garak")
            return ProbeResult(
                model_name=model_name,
                probe_names=self.probes,
                patch_config=patch_label,
            )
        finally:
            # Clean up temp config file
            try:
                Path(config_path).unlink(missing_ok=True)
            except Exception:
                pass
 
        # Parse the report
        report_path = self._find_report(model_name, patch_label)
        result = self._parse_report(report_path, model_name, patch_label)
 
        logger.info(
            "Probe complete | model=%s | patch=%s | attempts=%d | failures=%d | rate=%.1f%%",
            model_name, patch_label,
            result.total_attempts, result.total_failures,
            result.failure_rate * 100,
        )
        return result
 
    def _find_report(self, model_name: str, patch_label: str) -> Optional[Path]:
        """
        Locate the most recently created Garak JSONL report for a model and patch label.
 
        Args:
            model_name: The model identifier used in the filename.
            patch_label: The patch configuration label used in the filename.
 
        Returns:
            Path to the JSONL report file, or None if not found.
        """
        prefix = f"{model_name.replace('.', '_')}_{patch_label}"
        candidates = sorted(
            self.report_dir.glob(f"{prefix}*.report.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning(
                "No Garak report found for '%s' (patch=%s) in %s",
                model_name, patch_label, self.report_dir
            )
            return None
        return candidates[0]
 
    def _parse_report(
        self,
        report_path: Optional[Path],
        model_name: str,
        patch_label: str,
    ) -> ProbeResult:
        """
        Parse a Garak JSONL report into a structured ProbeResult.

        Garak emits one JSON object per line. We look for:
          - entry_type == "attempt": one record per probe attempt
          - entry_type == "eval": summary counts (faster parsing)
 
        Args:
            report_path: Path to the Garak JSONL report file.
            model_name:  Model identifier for labelling.
            patch_label: Patch configuration label.
 
        Returns:
            ProbeResult with aggregated statistics.
        """
        result = ProbeResult(
            model_name=model_name,
            probe_names=self.probes,
            patch_config=patch_label,
            report_path=str(report_path) if report_path else None,
        )
 
        if not report_path or not report_path.exists():
            logger.warning("Report file not found: %s", report_path)
            return result
 
        failures_by_probe: dict = {}
        total_attempts = 0
        total_failures = 0
 
        with open(report_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                entry_type = record.get("entry_type", "")
                
                # Using eval summary if available > one record per probe/detector pair
                if entry_type == "eval":
                    total_attempts += record.get("total_evaluated", 0)
                    total_failures += record.get("fails", 0)
                    
                # Fall back to parsing individual attempts
                elif entry_type == "attempt":
                    probe_name = record.get("probe_classname", "unknown")
                    total_attempts += 1
                    det = record.get("detector_results", {})
                    if any(v == 0 for v in det.values()):
                        total_failures += 1
                        failures_by_probe[probe_name] = (
                            failures_by_probe.get(probe_name, 0) + 1
                        )
 
                """# Garak report entries have 'probe', 'status', 'passed'
                probe_name = record.get("probe", "unknown")
                status = record.get("status", None)
 
                # status == 1 means the probe ran; passed == False means a failure
                if status is not None:
                    total_attempts += 1
                    passed = record.get("passed", True)
                    if not passed:
                        total_failures += 1
                        failures_by_probe[probe_name] = failures_by_probe.get(probe_name, 0) + 1
                """
        
        # If eval records were found they give the totals directly
        # but failures_by_probe needs attempt records — reconcile
        result.total_attempts = total_attempts
        result.total_failures = total_failures
        result.failure_rate = (
            total_failures / total_attempts if total_attempts > 0 else 0.0
        )
        result.failures_by_probe = failures_by_probe
        return result