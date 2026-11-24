"""Abstract LLM Client, OpenAI, Ollama, and Mock implementations with caching and retry."""

import os
import re
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import DependencyType


class LLMClient(ABC):
    """Abstract Base Class for LLM Client interface."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        """Generate response string from prompt. Returns (response_text, usage_info)."""
        pass

    def generate_uncached(
        self, prompt: str, temperature: float = 0.2
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a fresh response; cache-aware clients can override this method."""
        return self.generate(prompt, temperature)


class DiskCachedLLMClient(LLMClient):
    """Decorator or mixin wrapper adding response caching to any LLM client."""

    def __init__(self, inner_client: LLMClient, cache_dir: str | None = None):
        self.inner_client = inner_client
        config = get_config()
        self.cache_dir = cache_dir or os.path.join(config.cache_dir, "llm_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        provider = getattr(self.inner_client, "provider", self.inner_client.__class__.__name__)
        model = getattr(self.inner_client, "model", "")
        content = f"v2:{provider}:{model}:{temperature}:{prompt}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        cache_key = self._get_cache_key(prompt, temperature)
        cache_file = os.path.join(self.cache_dir, f"llm_{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                return cached_data["response"], cached_data["usage"]
            except Exception as e:
                logger.debug(f"Cache read error ({e}), re-generating.")

        response_text, usage = self.inner_client.generate(prompt, temperature)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"response": response_text, "usage": usage}, f, indent=2)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

        return response_text, usage

    def generate_uncached(
        self, prompt: str, temperature: float = 0.2
    ) -> Tuple[str, Dict[str, Any]]:
        """Bypass this wrapper's cache so repeated-run checks receive fresh outputs."""
        return self.inner_client.generate(prompt, temperature)


class OpenAILLMClient(LLMClient):
    """OpenAI GPT API Client with exponential backoff retry logic."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        config = get_config()
        self.api_key = api_key or config.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or config.llm_model

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")

        try:
            import openai
            # Support both openai>=1.0.0 and legacy versions
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
            else:
                openai.api_key = self.api_key
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                text = response["choices"][0]["message"]["content"]
                usage_dict = response.get("usage", {})
                prompt_tokens = usage_dict.get("prompt_tokens", 0)
                completion_tokens = usage_dict.get("completion_tokens", 0)

            # Standard GPT-4 cost estimate ($0.03/1k prompt, $0.06/1k completion)
            cost_usd = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000.0

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "provider": "openai",
                "model": self.model
            }
            return text, usage

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise e


def _is_retryable_ollama_exception(exc: BaseException) -> bool:
    try:
        import httpx
    except Exception:
        return False
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response is not None:
            return exc.response.status_code in (500, 502, 503, 504)
    return False


class OllamaLLMClient(LLMClient):
    """Local Ollama REST API Client."""

    def __init__(self, base_url: str | None = None, model: str = "llama3"):
        config = get_config()
        self.base_url = base_url or config.ollama_base_url
        self.model = model or config.llm_model

    @retry(
        retry=retry_if_exception(_is_retryable_ollama_exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=4),
    )
    def _call_ollama(self, url: str, payload: dict) -> dict:
        import httpx
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def generate(self, prompt: str, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }

        try:
            data = self._call_ollama(url, payload)
            text = data.get("response", "")

            # Approximate tokens
            prompt_tokens = len(prompt.split())
            completion_tokens = len(text.split())
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": 0.0,
                "provider": "ollama",
                "model": self.model
            }
            return text, usage
        except Exception as e:
            # Unwrap tenacity RetryError so callers see the original transient failure
            msg_exc = e
            try:
                from tenacity import RetryError
                if isinstance(e, RetryError) and hasattr(e, "last_attempt") and e.last_attempt is not None:
                    inner = e.last_attempt.exception()
                    if inner is not None:
                        msg_exc = inner
            except Exception:
                pass
            logger.error(f"Ollama API call error: {msg_exc}")
            raise RuntimeError(f"Ollama LLM call failed: {msg_exc}")


class MockLLMClient(LLMClient):
    """Deterministic Mock LLM Client for offline testing and fast local evaluation."""

    def __init__(self, model: str = "mock-gpt4", response_text: str | None = None):
        self.model = model
        self.response_text = response_text

    def generate(self, prompt: str, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        if self.response_text is not None:
            output_text = self.response_text
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output_text.split()),
                "cost_usd": 0.0,
                "provider": "mock",
                "model": self.model
            }
            return output_text, usage
        # Extract Requirement A and Requirement B text from prompt
        req_a_match = re.search(r"Requirement A:\s*(.*?)\n", prompt)
        req_b_match = re.search(r"Requirement B:\s*(.*?)\n", prompt)

        text_a = req_a_match.group(1).lower() if req_a_match else ""
        text_b = req_b_match.group(1).lower() if req_b_match else ""

        # Smart heuristic parsing matching domain examples
        if "prerequisite" in text_a or "require" in text_a or ("bcs" in text_a and "bcs" in text_b) or ("resume" in text_a and "stop" in text_b):
            predicted_type = DependencyType.REQUIRES
            confidence = 5
            rationale = "Requirement A explicitly depends on the functionality or system component defined in Requirement B."
        elif "subsystem" in text_a or "subsystem" in text_b or "implement" in text_a or "check" in text_a:
            predicted_type = DependencyType.IMPLEMENTS
            confidence = 5
            rationale = "Requirement B defines a lower-level subsystem implementation detail for high-level Requirement A."
        elif "conflict" in text_a or "not irritate" in text_a or "disengage" in text_a or "alert" in text_a:
            predicted_type = DependencyType.CONFLICTS
            confidence = 5
            rationale = "The fulfillment of Requirement A imposes restrictions or timing conflicts on Requirement B."
        elif "detail" in text_a or "gradually" in text_a or "specific" in text_a or ("stop" in text_a and "obstacle" in text_b):
            predicted_type = DependencyType.DETAILS
            confidence = 5
            rationale = "Both requirements share the same underlying action, but Requirement B provides additional execution details."
        elif "authenticate" in text_a or "verify source" in text_b or "similar" in text_a:
            predicted_type = DependencyType.IS_SIMILAR
            confidence = 5
            rationale = "Requirement A and Requirement B describe identical functionality resulting in semantic redundancy."
        else:
            predicted_type = DependencyType.NO_DEPENDENCY
            confidence = 5
            rationale = "Requirement A and Requirement B describe independent vehicle functions with no direct or indirect dependency."

        output_text = f"""**Dependency_Type: {predicted_type.value}**
**Rationale: {rationale}**
**Confidence Score: {confidence}**"""

        prompt_tokens = len(prompt.split())
        completion_tokens = len(output_text.split())
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": 0.0,
            "provider": "mock",
            "model": self.model
        }
        return output_text, usage


class LLMClientFactory:
    """Factory to instantiate LLM client based on configuration."""

    @staticmethod
    def create(config: LEREDDConfig | None = None) -> LLMClient:
        cfg = config or get_config()
        provider = cfg.llm_provider.lower()

        if provider == "openai":
            base_client = OpenAILLMClient(api_key=cfg.openai_api_key, model=cfg.llm_model)
        elif provider == "ollama":
            base_client = OllamaLLMClient(base_url=cfg.ollama_base_url, model=cfg.llm_model)
        else:
            base_client = MockLLMClient(model=cfg.llm_model)

        return DiskCachedLLMClient(base_client, cache_dir=os.path.join(cfg.cache_dir, "llm_cache"))
