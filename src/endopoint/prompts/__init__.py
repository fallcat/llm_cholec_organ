"""Prompt management for endopoint."""

from .builders import (
    build_pointing_system_prompt,
    build_pointing_user_prompt,
    build_pointing_system_prompt_strict,
    build_pointing_system_prompt_qna,
)
from .registry import PROMPT_REGISTRY, get_prompt_config, register_prompt

__all__ = [
    "build_pointing_system_prompt",
    "build_pointing_user_prompt",
    "build_pointing_system_prompt_strict",
    "build_pointing_system_prompt_qna",
    "PROMPT_REGISTRY",
    "get_prompt_config",
    "register_prompt",
]