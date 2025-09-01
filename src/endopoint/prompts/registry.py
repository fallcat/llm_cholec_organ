"""Prompt registry for managing different prompt strategies."""

from typing import Any, Callable, Dict, Optional

from .builders import (
    build_pointing_system_prompt,
    build_pointing_system_prompt_qna,
    build_pointing_system_prompt_strict,
    build_pointing_user_prompt,
)


# Prompt configuration type
PromptConfig = Dict[str, Any]

# Registry of prompt configurations
PROMPT_REGISTRY: Dict[str, PromptConfig] = {}


def register_prompt(name: str, config: PromptConfig) -> None:
    """Register a prompt configuration.
    
    Args:
        name: Prompt name
        config: Prompt configuration dict with:
            - system_builder: Function to build system prompt
            - user_builder: Function to build user prompt
            - few_shot_provider: Optional few-shot provider class
            - description: Optional description
    """
    PROMPT_REGISTRY[name] = config


def get_prompt_config(name: str) -> PromptConfig:
    """Get prompt configuration by name.
    
    Args:
        name: Prompt name
        
    Returns:
        Prompt configuration
        
    Raises:
        KeyError: If prompt not found
    """
    if name not in PROMPT_REGISTRY:
        available = list(PROMPT_REGISTRY.keys())
        raise KeyError(
            f"Prompt '{name}' not found in registry. "
            f"Available prompts: {available}"
        )
    
    return PROMPT_REGISTRY[name]


# Register default prompts
register_prompt(
    "base",
    {
        "system_builder": build_pointing_system_prompt,
        "user_builder": build_pointing_user_prompt,
        "description": "Base pointing prompt with minimal instructions",
    }
)

register_prompt(
    "strict", 
    {
        "system_builder": build_pointing_system_prompt_strict,
        "user_builder": build_pointing_user_prompt,
        "description": "Strict pointing prompt with explicit instructions",
    }
)

register_prompt(
    "qna",
    {
        "system_builder": build_pointing_system_prompt_qna,
        "user_builder": build_pointing_user_prompt,
        "description": "Q&A style pointing prompt",
    }
)

# Placeholder for few-shot variants
register_prompt(
    "base_fewshot",
    {
        "system_builder": build_pointing_system_prompt,
        "user_builder": build_pointing_user_prompt,
        "few_shot_provider": None,  # Will be set when fewshot module is implemented
        "description": "Base prompt with few-shot examples",
    }
)

register_prompt(
    "strict_fewshot",
    {
        "system_builder": build_pointing_system_prompt_strict,
        "user_builder": build_pointing_user_prompt,
        "few_shot_provider": None,  # Will be set when fewshot module is implemented
        "description": "Strict prompt with few-shot examples",
    }
)