"""Unified LLM client interface managing caching, retries, cost tracking and billing."""

import time

from src.core.llm.cache import LLMCache
from src.core.llm.config import LLMConfig
from src.core.llm.cost_tracker import CostTracker
from src.core.llm.provider import LLMProvider
from src.core.llm.retry import retry_with_backoff
from src.core.llm.tokenizer import Tokenizer
from src.core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Enterprise client managing connection profiles, caches, costs and retries."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize LLMClient.

        Args:
            config: Optional LLMConfig object.
        """
        self.config = config or LLMConfig()
        self.cache = LLMCache(ttl_seconds=self.config.cache_ttl_seconds)

    def generate(self, prompt: str, bypass_cache: bool = False) -> str:
        """Execute chat completions, checking cache and tracking token costs.

        Args:
            prompt: Text prompt request.
            bypass_cache: Set to True to skip checking prompt cache.

        Returns:
            str: Response string.
        """
        logger.info(f"LLMClient: Generating response (provider={self.config.provider.upper()})")
        start_time = time.time()

        # 1. Check cache first
        if not bypass_cache:
            cached_val = self.cache.get(prompt)
            if cached_val is not None:
                logger.info("LLMClient: Caching Hit. Bypassing provider calls.")
                return cached_val

        # 2. Count prompt input tokens
        input_tokens = Tokenizer.count_tokens(prompt, self.config.model)

        # 3. Call provider via exponential backoff decorator
        @retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions_to_retry=[Exception])
        def _execute_call():
            return LLMProvider.call_provider(prompt, self.config)

        logger.info("LLMClient: Calling LLM provider...")
        response = _execute_call()

        # 4. Count completion output tokens
        output_tokens = Tokenizer.count_tokens(response, self.config.model)

        # 5. Save in cache
        if not bypass_cache:
            self.cache.set(prompt, response)

        # 6. Track financial cost metrics
        CostTracker.track_request(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_per_million_input=self.config.cost_per_million_input,
            cost_per_million_output=self.config.cost_per_million_output
        )

        duration = time.time() - start_time
        logger.info(f"LLMClient: Generation complete in {round(duration, 4)}s")
        return response
