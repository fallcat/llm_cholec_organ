"""Model adapters for endopoint."""

from .base import ModelAdapter, PromptPart, OneQuery, Batch
from .openai_gpt import OpenAIAdapter
from .anthropic_claude import AnthropicAdapter
from .google_gemini import GoogleAdapter
from .vllm import LLaVAModel, QwenVLModel, PixtralModel, DeepSeekVL2Model


def create_model(model_id: str, use_cache: bool = True):
    """Create a model adapter based on model ID.
    
    Args:
        model_id: Model identifier string
        use_cache: Whether to use caching for responses
        
    Returns:
        Appropriate model adapter instance
    """
    # Map model IDs to adapters
    if 'gpt' in model_id.lower():
        adapter = OpenAIAdapter(model_name=model_id, use_cache=use_cache)
    elif 'claude' in model_id.lower():
        adapter = AnthropicAdapter(model_name=model_id, use_cache=use_cache)
    elif 'gemini' in model_id.lower():
        adapter = GoogleAdapter(model_name=model_id, use_cache=use_cache)
    elif 'llava' in model_id.lower():
        adapter = LLaVAModel(model_id)
    elif 'qwen' in model_id.lower():
        adapter = QwenVLModel(model_id)
    elif 'pixtral' in model_id.lower():
        adapter = PixtralModel(model_id)
    elif 'deepseek' in model_id.lower():
        adapter = DeepSeekVL2Model(model_id)
    else:
        # Default to OpenAI adapter
        adapter = OpenAIAdapter(model_name=model_id, use_cache=use_cache)
    
    # Add model_id attribute for compatibility
    adapter.model_id = model_id
    return adapter


__all__ = [
    "ModelAdapter",
    "PromptPart",
    "OneQuery", 
    "Batch",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleAdapter",
    "LLaVAModel",
    "QwenVLModel",
    "PixtralModel",
    "DeepSeekVL2Model",
    "create_model",
]