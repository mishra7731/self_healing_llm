import re
import logging
from dataclasses import dataclass, field
from typing import Optional
 
logger = logging.getLogger(__name__)

# Heuristic toxicity scoring
# A lightweight word-list approach that avoids needing a GPU/external model.

_TOXIC_SIGNALS: list[tuple[str, float]] = [
    # (pattern, weight)
    (r"\bkill\s+(?:all|every)\b", 0.9),
    (r"\bhate\s+(?:all|every|you)\b", 0.7),
    (r"\bstupid\s+(?:idiot|moron)\b", 0.6),
    (r"\b(?:slur words go here)\b", 0.95),   # extend with domain slurs
    (r"\b(?:f+u+c+k|s+h+i+t)\b", 0.5),
    (r"\bdie\s+in\s+a\b", 0.85),
    (r"\byou\s+(?:are|r)\s+(?:worthless|garbage|trash)\b", 0.75),
]
_COMPILED_TOXIC = [(re.compile(p, re.IGNORECASE), w) for p, w in _TOXIC_SIGNALS]

def heuristic_toxicity_score(text: str) -> float:
    """
    Compute a rough toxicity score in [0, 1] using weighted pattern matching.
 
    This is intentionally simple — the goal is to demonstrate the concept
    without requiring a GPU or external API. In a real research system,
    replace with `detoxify` or the Perspective API for validated scores.
 
    Args:
        text: The model response to score.
 
    Returns:
        Float in [0.0, 1.0] where higher = more likely toxic.
    """
    score = 0.0
    for pattern, weight in _COMPILED_TOXIC:
        if pattern.search(text):
            score = min(1.0, score + weight)
    return score

@dataclass
class OutputPatchConfig:
    """Configuration for output-level mitigations."""
    keyword_filter_enabled: bool = True
    blocked_phrases: list[str] = field(default_factory=list)
    keyword_fallback: str = "I'm sorry, I can't assist with that."
    toxicity_filter_enabled: bool = True
    toxicity_threshold: float = 0.7
    toxicity_fallback: str = "I'm unable to provide that response."
 
 
@dataclass
class OutputPatchResult:
    """Result of applying output-level patches to a model response."""
    original_response: str
    patched_response: str
    was_replaced: bool
    replacement_reason: Optional[str] = None
    toxicity_score: Optional[float] = None
    triggered_phrase: Optional[str] = None
 
 
class OutputPatch:
    """
    Applies output-level security mitigations to model responses.
 
    This acts as a post-processing layer after the LLM generates output.
    Two mechanisms are available:
      - Keyword filter: blocks responses mentioning known jailbreak phrases
      - Toxicity filter: heuristically scores and blocks toxic content
 
    Usage:
        patch = OutputPatch(config)
        result = patch.apply(model_response)
        return result.patched_response  # safe to return to user
    """
 
    def __init__(self, config: OutputPatchConfig):
        """
        Initialise the OutputPatch with a given configuration.
 
        Args:
            config: OutputPatchConfig specifying which mechanisms to enable
                    and their parameters.
        """
        self.config = config
        self._compiled_phrases = self._compile_phrases()
        logger.info(
            "OutputPatch initialised | keyword_filter=%s | toxicity_filter=%s (threshold=%.2f)",
            config.keyword_filter_enabled,
            config.toxicity_filter_enabled,
            config.toxicity_threshold,
        )
 
    def _compile_phrases(self) -> list[tuple[re.Pattern, str]]:
        """
        Pre-compile blocked phrase patterns for efficient matching.
 
        Returns:
            List of (compiled_pattern, original_phrase) tuples.
        """
        compiled = []
        for phrase in self.config.blocked_phrases:
            try:
                compiled.append(
                    (re.compile(re.escape(phrase), re.IGNORECASE), phrase)
                )
            except re.error as e:
                logger.warning("Could not compile phrase '%s': %s", phrase, e)
        return compiled
 
    def keyword_check(self, response: str) -> Optional[str]:
        """
        Scan a response for blocked keyword/phrase patterns.
 
        Args:
            response: The raw model output string.
 
        Returns:
            The matched phrase if found, else None.
        """
        if not self.config.keyword_filter_enabled:
            return None
 
        for pattern, original in self._compiled_phrases:
            if pattern.search(response):
                logger.warning("Blocked phrase found in output: '%s'", original)
                return original
        return None
 
    def toxicity_check(self, response: str) -> float:
        """
        Score the toxicity of a model response.
 
        Args:
            response: The raw model output string.
 
        Returns:
            Toxicity score in [0.0, 1.0].
        """
        if not self.config.toxicity_filter_enabled:
            return 0.0
        return heuristic_toxicity_score(response)
 
    def apply(self, response: str) -> OutputPatchResult:
        """
        Apply all enabled output-level mitigations to a model response.
 
        Processing order:
          1. Keyword filter → replace if a blocked phrase is matched
          2. Toxicity filter → replace if score exceeds threshold
 
        Args:
            response: The raw model output.
 
        Returns:
            OutputPatchResult describing what happened and the final response.
        """
        result = OutputPatchResult(
            original_response=response,
            patched_response=response,
            was_replaced=False,
        )
 
        # Step 1: Keyword filter
        matched_phrase = self.keyword_check(response)
        if matched_phrase:
            result.was_replaced = True
            result.replacement_reason = "keyword_filter"
            result.triggered_phrase = matched_phrase
            result.patched_response = self.config.keyword_fallback
            logger.info("Output REPLACED by keyword filter (phrase: '%s')", matched_phrase)
            return result
 
        # Step 2: Toxicity filter
        score = self.toxicity_check(response)
        result.toxicity_score = score
        if score >= self.config.toxicity_threshold:
            result.was_replaced = True
            result.replacement_reason = "toxicity_filter"
            result.patched_response = self.config.toxicity_fallback
            logger.info(
                "Output REPLACED by toxicity filter (score: %.3f >= threshold: %.2f)",
                score,
                self.config.toxicity_threshold,
            )
 
        return result
 
 
def build_from_config(cfg: dict) -> OutputPatch:
    """
    Construct an OutputPatch instance from a parsed YAML config dict.
 
    Args:
        cfg: The full experiment config dict.
 
    Returns:
        A configured OutputPatch instance.
    """
    output_cfg = cfg.get("patches", {}).get("output_level", {})
    kw_cfg = output_cfg.get("keyword_filter", {})
    tox_cfg = output_cfg.get("toxicity_filter", {})
 
    config = OutputPatchConfig(
        keyword_filter_enabled=kw_cfg.get("enabled", True),
        blocked_phrases=kw_cfg.get("blocked_phrases", []),
        keyword_fallback=kw_cfg.get("fallback_message", "I can't help with that."),
        toxicity_filter_enabled=tox_cfg.get("enabled", True),
        toxicity_threshold=tox_cfg.get("threshold", 0.7),
        toxicity_fallback=tox_cfg.get("fallback_message", "I'm unable to provide that response."),
    )
    return OutputPatch(config)