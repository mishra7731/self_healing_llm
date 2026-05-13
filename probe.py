import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
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
    probe_names: list[str]
    total_attempts: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    failures_by_probe: dict[str, int] = field(default_factory=dict)
    report_path: Optional[str] = None
    patch_config: str = "no_patches"
 
 
class GarakRunner:
    """
    Wrapper around the Garak CLI for running vulnerability probes.
 
    Garak is invoked as a subprocess. Its JSONL report is then parsed to
    extract structured results for comparison and logging.
 
    Typical usage:
        runner = GarakRunner(cfg, report_dir="./results/garak_reports")
        result = runner.run(model_name="llama3.2", patch_label="baseline")
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
 
        self.probes: list[str] = cfg.get("garak", {}).get("probes", [])
        self.generations: int = cfg.get("garak", {}).get("generations", 5)
 
    def _build_command(self, generator: str, model_name: str) -> list[str]:
        """
        Construct the Garak CLI command for a given model and probe set.
 
        Args:
            generator:  Garak generator string (e.g. 'ollama.OllamaGeneratorChat').
            model_name: Model identifier (e.g. 'llama3.2').
 
        Returns:
            List of command-line tokens ready for subprocess.
        """
        probe_arg = ",".join(self.probes)
        cmd = [
            sys.executable, "-m", "garak",
            "--model_type", generator,
            "--model_name", model_name,
            "--probes", probe_arg,
            "--generations", str(self.generations),
            "--report_prefix", str(self.report_dir / f"{model_name.replace('/', '_')}"),
        ]
        logger.debug("Garak command: %s", " ".join(cmd))
        return cmd
 
    def run(
        self,
        model_key: str,
        patch_label: str = "baseline",
        env_overrides: Optional[dict] = None,
    ) -> ProbeResult:
        """
        Run Garak against the specified model and return structured results.
 
        Args:
            model_key:      Key into cfg['models'] (e.g. 'llama', 'mistral').
            patch_label:    Human-readable label for the patch configuration
                            (used in filenames and result logging).
            env_overrides:  Optional environment variables to pass to the subprocess.
 
        Returns:
            ProbeResult with failure counts and path to the raw report.
        """
        model_cfg = self.cfg["models"][model_key]
        model_name = model_cfg["name"]
        generator = model_cfg["generator"]
 
        logger.info(
            "Starting Garak probe | model=%s | probes=%s | patch=%s",
            model_name, self.probes, patch_label,
        )
 
        cmd = self._build_command(generator, model_name)
        env = {**os.environ, **(env_overrides or {})}
 
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,  # 10-minute timeout
            )
            if proc.returncode != 0:
                logger.error("Garak exited with code %d:\n%s", proc.returncode, proc.stderr)
        except subprocess.TimeoutExpired:
            logger.error("Garak probe timed out for model '%s'", model_name)
            return ProbeResult(model_name=model_name, probe_names=self.probes, patch_config=patch_label)
        except FileNotFoundError:
            logger.error("Garak not found. Is it installed? Run: pip install garak")
            return ProbeResult(model_name=model_name, probe_names=self.probes, patch_config=patch_label)
 
        # Find the most recently created JSONL report
        report_path = self._find_report(model_name)
        result = self._parse_report(report_path, model_name, patch_label)
        logger.info(
            "Probe complete | model=%s | attempts=%d | failures=%d | rate=%.1f%%",
            model_name, result.total_attempts, result.total_failures,
            result.failure_rate * 100,
        )
        return result
 
    def _find_report(self, model_name: str) -> Optional[Path]:
        """
        Locate the most recently created Garak JSONL report for a model.
 
        Args:
            model_name: The model identifier used in the filename.
 
        Returns:
            Path to the JSONL report file, or None if not found.
        """
        prefix = model_name.replace("/", "_")
        candidates = sorted(
            self.report_dir.glob(f"{prefix}*.report.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning("No Garak report found for model '%s' in %s", model_name, self.report_dir)
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
 
        Garak emits one JSON object per line. Each object has a 'status' field:
          - 1 = passed (model responded safely)
          - 0 = failed (model produced an unsafe output)
 
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
            logger.warning("Report file not found, returning empty ProbeResult.")
            return result
 
        failures_by_probe: dict[str, int] = {}
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
 
                # Garak report entries have 'probe', 'status', 'passed'
                probe_name = record.get("probe", "unknown")
                status = record.get("status", None)
 
                # status == 1 means the probe ran; passed == False means a failure
                if status is not None:
                    total_attempts += 1
                    passed = record.get("passed", True)
                    if not passed:
                        total_failures += 1
                        failures_by_probe[probe_name] = failures_by_probe.get(probe_name, 0) + 1
 
        result.total_attempts = total_attempts
        result.total_failures = total_failures
        result.failure_rate = total_failures / total_attempts if total_attempts > 0 else 0.0
        result.failures_by_probe = failures_by_probe
        return result