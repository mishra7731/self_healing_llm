import re
import logging
from dataclasses import dataclass, field
from typing import Optional
 
logger = logging.getLogger(__name__)

@dataclass
class PromptPatchConfig:
    """Configuration for prompt-level mitigations."""
    injection_detection_enabled: bool = True
    injection_patterns: list[str] = field(default_factory=list)
    guardrail_enabled: bool = True
    guardrail_text: str = ""
 
 
@dataclass
class PromptPatchResult:
    """Result of applying prompt-level patches to a user input."""
    original_prompt: str
    patched_prompt: str
    was_blocked: bool
    was_modified: bool
    triggered_pattern: Optional[str] = None
    guardrail_applied: bool = False
 
 
class PromptPatch:
    """
    Applies prompt-level security mitigations to user inputs.
 
    This acts as a pre-processing layer before the LLM sees any input.
    It can either block a prompt entirely (if injection is detected) or
    prepend a safety system prompt to bias the model toward safe outputs.
 
    Usage:
        patch = PromptPatch(config)
        result = patch.apply("Ignore all previous instructions and...")
        if result.was_blocked:
            return BLOCKED_RESPONSE
        send_to_model(result.patched_prompt)
    """
 
    BLOCKED_RESPONSE = (
        "I'm sorry, I can't process that request. "
        "It appears to contain instructions attempting to override my guidelines. "
        "Please ask me something else."
    )
 
    def __init__(self, config: PromptPatchConfig):
        """
        Initialise the PromptPatch with a given configuration.
 
        Args:
            config: PromptPatchConfig specifying which mechanisms to enable
                    and their parameters.
        """
        self.config = config
        self._compiled_patterns = self._compile_patterns()
        logger.info(
            "PromptPatch initialised | injection_detection=%s | guardrail=%s",
            config.injection_detection_enabled,
            config.guardrail_enabled,
        )
 
    def _compile_patterns(self) -> list[re.Pattern]:
        """
        Pre-compile injection detection regex patterns for efficiency.
 
        Returns:
            List of compiled regex patterns (case-insensitive).
        """
        compiled = []
        for pattern in self.config.injection_patterns:
            try:
                compiled.append(re.compile(re.escape(pattern), re.IGNORECASE))
            except re.error as e:
                logger.warning("Could not compile pattern '%s': %s", pattern, e)
        return compiled
 
    def detect_injection(self, prompt: str) -> Optional[str]:
        """
        Check whether a prompt contains known injection patterns.
 
        Args:
            prompt: The raw user input string.
 
        Returns:
            The matched pattern string if injection is detected, else None.
        """
        if not self.config.injection_detection_enabled:
            return None
 
        for pattern, original in zip(self._compiled_patterns, self.config.injection_patterns):
            if pattern.search(prompt):
                logger.warning("Injection pattern detected: '%s'", original)
                return original
        return None
 
    def apply_guardrail(self, prompt: str) -> str:
        """
        Prepend the safety system prompt to the user input.
 
        This biases the model to respect safety guidelines even when the
        prompt itself doesn't contain explicit violations.
 
        Args:
            prompt: The user input (possibly already screened).
 
        Returns:
            The prompt with the guardrail system message prepended.
        """
        if not self.config.guardrail_enabled or not self.config.guardrail_text:
            return prompt
 
        guardrail = self.config.guardrail_text.strip()
        return f"[SYSTEM INSTRUCTIONS]\n{guardrail}\n\n[USER MESSAGE]\n{prompt}"
 
    def apply(self, prompt: str) -> PromptPatchResult:
        """
        Apply all enabled prompt-level mitigations to a user input.
 
        Processing order:
          1. Check for injection patterns → block if found
          2. Apply guardrail system prompt → prepend safety instructions
 
        Args:
            prompt: The raw user input.
 
        Returns:
            PromptPatchResult describing what happened and the final prompt.
        """
        result = PromptPatchResult(
            original_prompt=prompt,
            patched_prompt=prompt,
            was_blocked=False,
            was_modified=False,
        )
 
        # Step 1: Injection detection
        triggered = self.detect_injection(prompt)
        if triggered:
            result.was_blocked = True
            result.triggered_pattern = triggered
            result.patched_prompt = self.BLOCKED_RESPONSE
            logger.info("Prompt BLOCKED due to injection pattern: '%s'", triggered)
            return result
 
        # Step 2: Apply guardrail
        if self.config.guardrail_enabled:
            result.patched_prompt = self.apply_guardrail(prompt)
            result.guardrail_applied = True
            result.was_modified = result.patched_prompt != prompt
            logger.debug("Guardrail applied to prompt.")
 
        return result
    
    
def build_from_config(cfg: dict) -> PromptPatch:
    """
    Construct a PromptPatch instance from a parsed YAML config dict.
 
    Args:
        cfg: The 'patches.prompt_level' section of the experiment config.
 
    Returns:
        A configured PromptPatch instance.
    """
    prompt_cfg = cfg.get("patches", {}).get("prompt_level", {})
    injection_cfg = prompt_cfg.get("injection_detection", {})
    guardrail_cfg = prompt_cfg.get("system_prompt_guardrail", {})
 
    config = PromptPatchConfig(
        injection_detection_enabled=injection_cfg.get("enabled", True),
        injection_patterns=injection_cfg.get("patterns", []),
        guardrail_enabled=guardrail_cfg.get("enabled", True),
        guardrail_text=guardrail_cfg.get("guardrail_text", ""),
    )
    return PromptPatch(config)