"""Token counter estimating prompt and completion lengths."""

from src.core.logger import get_logger

logger = get_logger(__name__)


class Tokenizer:
    """Estimates tokens count utilizing tiktoken for OpenAI or characters heuristic fallback."""

    @staticmethod
    def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
        """Count or estimate the number of tokens in a text string.

        Args:
            text: Input text.
            model_name: Model identifier.

        Returns:
            int: Estimated number of tokens.
        """
        if not text:
            return 0

        try:
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(model_name)
                return len(encoding.encode(text))
            except Exception:
                # Fallback to cl100k_base encoding if model unrecognized
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
        except ImportError:
            # Simple heuristic fallback if tiktoken not installed:
            # 1 token is approximately 4 characters of English text, or 0.75 words.
            char_count = len(text)
            words = text.split()
            word_count = len(words)

            # Combine heuristics for a robust average estimate
            estimate = int(max(char_count / 4.0, word_count * 1.3))
            return max(1, estimate)
