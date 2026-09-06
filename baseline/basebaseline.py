"""
Base definitions for Prompt Engineering Baselines.

This module defines:
- BaselineResponse: a standardized dataclass for baseline execution outputs
- BaseBaseline: an abstract base class that enforces a common interface
    for all prompt engineering methods

The goal of this module is to provide a consistent, extensible
foundation for integrating different prompt engineering techniques.

Author: Egor Morozov
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from models.base import BaseLLM, LLMResponse

# Tasks where the generation *is* the answer, so pulling a short "final answer"
# out of the response would discard it: arithmetic expressions (Game of 24), a
# poem (SonnetWriting), a function or class body (HumanEval / MBPP / ClassEval).
# Shared by ZeroShotCoT (which skips its extraction pass) and FoT (whose
# ``ANSWER:`` capture otherwise stops at the first blank line, truncating a
# stanza-separated sonnet to its first quatrain).
def is_generative_task(instruction: Optional[str] = None,
                       question: Optional[str] = None) -> bool:
    """True when the response body is the answer and must not be summarised."""
    instruction_lower = instruction.lower() if instruction else ""
    question_lower = question.lower() if question else ""
    return (
        "expression" in instruction_lower or
        "equation" in instruction_lower or
        "formula" in instruction_lower or
        "sonnet" in instruction_lower or
        "sonnet" in question_lower or
        "poem" in instruction_lower or
        ("write" in instruction_lower and "line" in instruction_lower) or
        "function body" in instruction_lower or
        "function implementation" in instruction_lower or
        ("implement" in instruction_lower and "function" in instruction_lower) or
        ("implement" in instruction_lower and "class" in instruction_lower) or
        "class implementation" in instruction_lower or
        "class body" in instruction_lower
    )


@dataclass
class BaselineResponse:
    """Container for a standardized baseline execution response.

    Attributes:
        final_answer: The final answer produced by the baseline method.
        reasoning_trace: The complete reasoning process/chain of thought.
        total_input_tokens: Total input tokens consumed across all LLM calls.
        total_output_tokens: Total output tokens generated across all LLM calls.
        num_llm_calls: Number of LLM API calls made during execution.
        baseline_type: Name/type of the baseline method used.
        intermediate_steps: Optional list of intermediate reasoning steps.
        metadata: Additional method-specific metadata.
    """

    final_answer: str
    reasoning_trace: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    num_llm_calls: int = 0
    baseline_type: str = ""
    intermediate_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return {
            "final_answer": self.final_answer,
            "reasoning_trace": self.reasoning_trace,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "num_llm_calls": self.num_llm_calls,
            "baseline_type": self.baseline_type,
            "intermediate_steps": self.intermediate_steps,
            "metadata": self.metadata,
        }

class BaseBaseline(ABC):
    """Abstract base class for all prompt engineering baseline methods.

    This class provides a unified interface for implementing various
    prompt engineering techniques such as:
    - Zero-Shot CoT
    - Self-Consistency
    - Tree-of-Thought (ToT)
    - RoT
    - And more...

    Subclasses must implement the `run` method which executes the
    prompt engineering strategy and returns a BaselineResponse.

    Attributes:
        llm: The LLM client instance to use for generation.
        baseline_name: Human-readable name of this baseline method.
    """

    def __init__(self, llm: BaseLLM, baseline_name: str = "BaseBaseline"):
        """Initialize the baseline with an LLM client.

        Args:
            llm: An instance of a BaseLLM subclass (e.g., GPTClient, GeminiClient).
            baseline_name: Name identifier for this baseline method.
        """
        self.llm = llm
        self.baseline_name = baseline_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.num_llm_calls = 0
        self._counter_lock = threading.Lock()

    def reset_counters(self) -> None:
        """Reset token and call counters before a new run."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.num_llm_calls = 0

    def call_llm(self, prompt: str, temperature: float = 0.0, logprobs: bool = False) -> LLMResponse:
        """Internal method to call the LLM and track usage statistics.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Sampling temperature for generation.
            logprobs: If True, request token log probabilities from the backend.
                      Populated in LLMResponse.avg_logprob when supported.

        Returns:
            LLMResponse containing the generation result.
        """
        response = self.llm.generate(prompt, temperature=temperature, logprobs=logprobs)
        with self._counter_lock:
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.num_llm_calls += 1
        return response

    def create_response(
        self,
        final_answer: str,
        reasoning_trace: str = "",
        intermediate_steps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BaselineResponse:
        """Helper method to create a standardized BaselineResponse.

        Args:
            final_answer: The final answer produced.
            reasoning_trace: The reasoning process.
            intermediate_steps: List of intermediate steps.
            metadata: Additional metadata.

        Returns:
            A populated BaselineResponse object.
        """
        return BaselineResponse(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            num_llm_calls=self.num_llm_calls,
            baseline_type=self.baseline_name,
            intermediate_steps=intermediate_steps or [],
            metadata=metadata or {},
        )

    @abstractmethod
    def run(self, question: str, **kwargs) -> BaselineResponse:
        """Execute the prompt engineering method on the given question.
        
        This is the main entry point for running a baseline. Subclasses
        must implement this method with their specific prompting strategy.
        
        Args:
            question: The input question or problem to solve.
            **kwargs: Additional method-specific arguments (e.g., examples
        for few-shot, num_paths for self-consistency, etc.)
        
        Returns:
            BaselineResponse containing the answer, reasoning, and metrics.
        
        Example usage:
            >>> llm = GPTClient(api_key="...")
            >>> baseline = ZeroShotCoT(llm)
            >>> response = baseline.run("What is 2 + 2?")
            >>> print(response.final_answer)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(baseline_name='{self.baseline_name}', llm={self.llm.__class__.__name__})"