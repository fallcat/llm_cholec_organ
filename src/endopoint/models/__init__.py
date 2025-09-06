"""Model adapters for endopoint."""

from .base import ModelAdapter, PromptPart, OneQuery, Batch
from .openai_gpt import OpenAIAdapter
from .anthropic_claude import AnthropicAdapter
from .google_gemini import GoogleAdapter
from .vllm import LLaVAModel, QwenVLModel, PixtralModel, DeepSeekVL2Model
from .llama import LlamaAdapter


def create_model(model_id: str, use_cache: bool = True, verbose: bool = True):
    """Create a model adapter based on model ID.
    
    Args:
        model_id: Model identifier string
        use_cache: Whether to use caching for responses
        verbose: Whether to enable verbose error logging
        
    Returns:
        Appropriate model adapter instance
    """
    # Map model IDs to adapters
    if 'gpt' in model_id.lower():
        adapter = OpenAIAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'claude' in model_id.lower():
        adapter = AnthropicAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'gemini' in model_id.lower():
        adapter = GoogleAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'llama' in model_id.lower():
        adapter = LlamaAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'llava' in model_id.lower():
        adapter = LLaVAModel(model_id, use_cache=use_cache, verbose=verbose)
    elif 'qwen' in model_id.lower():
        adapter = QwenVLModel(model_id, use_cache=use_cache, verbose=verbose)
    elif 'pixtral' in model_id.lower():
        adapter = PixtralModel(model_id, use_cache=use_cache, verbose=verbose)
    elif 'deepseek' in model_id.lower():
        adapter = DeepSeekVL2Model(model_id, use_cache=use_cache, verbose=verbose)
    else:
        # Default to OpenAI adapter
        adapter = OpenAIAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    
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
    "LlamaAdapter",
    "LLaVAModel",
    "QwenVLModel",
    "PixtralModel",
    "DeepSeekVL2Model",
    "create_model",
]