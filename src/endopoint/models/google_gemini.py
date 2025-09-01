"""Google Gemini model adapter."""

import concurrent.futures
import os
import time
from typing import Optional, Sequence

from google import genai
from google.genai import types as genai_types

from .base import Batch, OneQuery
from .utils import cache, get_cache_key, image_to_base64, is_image


class GoogleAdapter:
    """Google Gemini API adapter."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        batch_size: int = 24,
        use_cache: bool = True,
        verbose: bool = False,
    ):
        """Initialize Google adapter.
        
        Args:
            model_name: Google model name
            api_key: Google API key (uses env var if not provided)
            num_tries_per_request: Number of retries for failed requests
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            batch_size: Maximum concurrent requests
            use_cache: Whether to use disk cache
            verbose: Whether to print error messages
        """
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
    
    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Process a batch of prompts through the model.
        
        Args:
            prompts: Batch of queries
            system_prompt: System prompt
            
        Returns:
            List of model responses
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [
                executor.submit(self._one_call, prompt=p, system_prompt=system_prompt)
                for p in prompts
            ]
            return [f.result() for f in futures]
    
    def _one_call(self, prompt: OneQuery, system_prompt: str) -> str:
        """Process a single prompt.
        
        Args:
            prompt: Single query tuple
            system_prompt: System prompt
            
        Returns:
            Model response string
        """
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
            if ret is not None and ret != "":
                return ret
        
        # Build content parts from tuple
        content_parts = []
        for p in prompt:
            if isinstance(p, str):
                content_parts.append(p)
            elif is_image(p):
                # For Google, we add images directly as PIL objects
                content_parts.append(p)
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
        
        # Combine system and user prompts
        full_content = [system_prompt] + content_parts
        
        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_content,
                    config=genai_types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                )
                # Extract text from response
                response_text = response.text.strip() if response.text else ""
                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling Google API: {e}")
                time.sleep(3)
        
        if self.use_cache and response_text:
            cache.set(get_cache_key(self.model_name, prompt, system_prompt), response_text)
        
        return response_text