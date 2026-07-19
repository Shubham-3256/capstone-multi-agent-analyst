"""Financial cost and token counters tracker tracking prompt and completion metrics."""


from src.core.logger import get_logger

logger = get_logger(__name__)


class CostTracker:
    """Tracks token consumption and dollar expenses per request and per session."""

    # Class-level session storage
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_estimated_cost: float = 0.0

    @classmethod
    def track_request(
        cls,
        input_tokens: int,
        output_tokens: int,
        cost_per_million_input: float = 5.0,
        cost_per_million_output: float = 15.0
    ) -> dict[str, float]:
        """Record usage metrics for a single request. Accumulates to session stats.

        Args:
            input_tokens: Input tokens count.
            output_tokens: Output tokens count.
            cost_per_million_input: Cost in USD per million input tokens.
            cost_per_million_output: Cost in USD per million output tokens.

        Returns:
            Dict[str, float]: Dict containing request cost metrics.
        """
        # Calculate cost
        input_cost = (input_tokens / 1_000_000.0) * cost_per_million_input
        output_cost = (output_tokens / 1_000_000.0) * cost_per_million_output
        total_cost = input_cost + output_cost

        # Accumulate session metrics
        cls.session_input_tokens += input_tokens
        cls.session_output_tokens += output_tokens
        cls.session_estimated_cost += total_cost

        logger.info(
            f"CostTracker: Request: Input={input_tokens}, Output={output_tokens}, "
            f"Cost=${round(total_cost, 6)}. "
            f"Session Cumulative: Cost=${round(cls.session_estimated_cost, 6)}"
        )

        return {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "cost": round(total_cost, 6)
        }

    @classmethod
    def get_session_summary(cls) -> dict[str, float]:
        """Fetch accumulated usage stats.

        Returns:
            Dict[str, float]: Summary stats dict.
        """
        return {
            "session_input_tokens": float(cls.session_input_tokens),
            "session_output_tokens": float(cls.session_output_tokens),
            "session_cost": round(cls.session_estimated_cost, 6)
        }

    @classmethod
    def reset_session(cls) -> None:
        """Reset session counters."""
        cls.session_input_tokens = 0
        cls.session_output_tokens = 0
        cls.session_estimated_cost = 0.0
        logger.info("CostTracker: Session counters reset.")
