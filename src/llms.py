import os
import concurrent.futures
import time
from typing import Any, List, Optional, Union
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types as genai_types
import numpy as np
import torch
import torchvision.transforms.functional as tvtf
import PIL.Image
import io
import base64
import diskcache
import pickle
import hashlib
from pathlib import Path

# Create a cache in this directory
cache = diskcache.Cache(Path(__file__).parent / ".llms.py.cache")


def get_cache_key(model_name: str, prompt: Any, system_prompt: Optional[str] = None) -> str:
    """Convert a (system_prompt, prompt) into a stable hash string."""
    sys_part = system_prompt or ""
    if isinstance(prompt, str):
        obj = ("user_text", prompt)
    elif isinstance(prompt, tuple):
        objs = []
        for p in prompt:
            if isinstance(p, str):
                objs.append(("text", p))
            elif is_image(p):
                # base64-encode image deterministically for hashing
                objs.append(("image_b64", image_to_base64(p, "PNG")))
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
        obj = ("user_tuple", tuple(objs))
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    return hashlib.sha256(pickle.dumps((model_name, sys_part, obj))).hexdigest()



def is_image(x: Any) -> bool:
    """Check if the input is an image in a supported format."""
    return isinstance(x, (PIL.Image.Image, torch.Tensor, np.ndarray))


def image_to_base64(
    image: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
    image_format: str = "PNG"
) -> str:
    """
    Convert an image to a base64-encoded string.
    
    Args:
        image: Input image in one of the following formats:
            - torch.Tensor: PyTorch tensor (C,H,W)
            - np.ndarray: NumPy array (H,W,C)
            - PIL.Image.Image: PIL Image object
        image_format: The format to save the image in (default: "PNG")

    Returns:
        str: Base64-encoded image string

    Raises:
        ValueError: If the input image format is invalid or conversion fails
    """
    try:
        # Convert to PIL image if needed
        if isinstance(image, torch.Tensor):
            mode = "RGB" if image.ndim == 3 and image.shape[0] == 3 else "L"
            image = tvtf.to_pil_image(image, mode=mode)
        elif isinstance(image, np.ndarray):
            mode = "RGB" if image.ndim == 3 and image.shape[2] == 3 else "L"
            image = tvtf.to_pil_image(image, mode=mode)

        assert isinstance(image, PIL.Image.Image), f"Image is not a PIL.Image.Image: {type(image)}"
        image.load() # Force loading the image

        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format)
            return base64.standard_b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")


def to_pil_image(x: Any) -> PIL.Image.Image:
    """Convert an image to a PIL Image object."""
    if isinstance(x, PIL.Image.Image):
        return x
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        return tvtf.to_pil_image(x)
    else:
        raise ValueError(f"Invalid image type: {type(x)}")


def load_model(model_name: str, api_key: Optional[str] = None):
    """Attempt to load the model based on the name"""
    if "gpt" in model_name or model_name.startswith("o"):
        return MyOpenAIModel(model_name=model_name, api_key=api_key)
    elif "claude" in model_name:
        return MyAnthropicModel(model_name=model_name, api_key=api_key)
    elif "gemini" in model_name:
        return MyGoogleModel(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


class MyOpenAIModel:
    """OpenAI API wrapper implementation with system prompt support."""
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        max_tokens: int = 2048,
        batch_size: int = 24,
        use_cache: bool = True,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=self.api_key)

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        """Process one or more prompts through the LLM API (optionally with a system prompt)."""
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
            if ret is not None and ret != "":
                return ret

        # Build user content (text + images)
        if isinstance(prompt, str):
            user_content = [{"type": "text", "text": prompt}]
        elif isinstance(prompt, tuple):
            user_content = []
            for p in prompt:
                if isinstance(p, str):
                    user_content.append({"type": "text", "text": p})
                elif is_image(p):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_base64(p, 'PNG')}"}
                    })
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # keeping param name as in your original code
                    max_completion_tokens=self.max_tokens,
                )
                response_text = (response.choices[0].message.content or "").strip()
                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling OpenAI's API: {e}")
                time.sleep(3)

        if self.use_cache and response_text:
            cache.set(get_cache_key(self.model_name, prompt, system_prompt), response_text)
        return response_text



class MyAnthropicModel:
    """Anthropic API wrapper implementation with system prompt support."""
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 24,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
            if ret is not None and ret != "":
                return ret

        # Build content list
        if isinstance(prompt, str):
            content = [{"type": "text", "text": prompt}]
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append({"type": "text", "text": p})
                elif is_image(p):
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_base64(p, "PNG")
                        }
                    })
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        messages = [{"role": "user", "content": content}]

        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    system=system_prompt,            # <-- system prompt here
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                # anthropic returns a list of content blocks
                response_text = (response.content[0].text or "").strip()
                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling Anthropic's API: {e}")
                time.sleep(3)

        if self.use_cache and response_text:
            cache.set(get_cache_key(self.model_name, prompt, system_prompt), response_text)
        return response_text



class MyGoogleModel:
    """Google Gemini API wrapper implementation with system prompt support."""
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 24,
        verbose: bool = False,
    ):
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

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
            if ret is not None and ret != "":
                return ret

        # Build multimodal content
        if isinstance(prompt, str):
            contents = [prompt]
        elif isinstance(prompt, tuple):
            contents = []
            for p in prompt:
                if isinstance(p, str):
                    contents.append(p)
                elif is_image(p):
                    contents.append(to_pil_image(p))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        system_instruction=system_prompt  # <-- system prompt here
                    )
                )
                response_text = (response.text or "").strip()
                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling Google's API: {e}")
                time.sleep(3)

        if self.use_cache and response_text:
            cache.set(get_cache_key(self.model_name, prompt, system_prompt), response_text)
        return response_text
