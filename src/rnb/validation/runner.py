"""
RnB Validation Test Runner

Orchestrates personality validation experiments:
- Conformity testing: Does agent match designed personality?
- Consistency testing: Does personality remain stable over time?

Example usage:
    from rnb.validation import ValidationRunner, load_archetypes

    runner = ValidationRunner(llm_client, agent_factory)

    # Quick conformity check
    result = runner.test_conformity("resilient", instrument="tipi")

    # Full consistency test
    result = runner.test_consistency(
        archetype="resilient",
        n_turns=100,
        sample_every=25,
        instrument="bfi2s"
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .assessor import (
    ArchetypeConfig,
    AssessmentResult,
    ConformityResult,
    PersonalityAssessor,
    load_archetypes,
)
from .metrics import (
    ConsistencyMetrics,
    DriftAnalysis,
    calculate_consistency_metrics,
    calculate_drift,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMClient(Protocol):
    """Protocol for LLM client interface."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response to the given prompt."""
        ...


class AgentProtocol(Protocol):
    """Protocol for RnB agent interface."""

    @property
    def agent_id(self) -> str:
        """Agent identifier."""
        ...

    def respond(self, user_message: str) -> str:
        """Generate response to user message."""
        ...

    def get_system_prompt(self) -> str:
        """Get the agent's system prompt including personality context."""
        ...


class AgentFactory(Protocol):
    """Protocol for creating agents from archetypes."""

    def from_archetype(self, archetype_name: str) -> AgentProtocol:
        """Create agent with specified archetype personality."""
        ...

    def from_traits(self, traits: dict[str, float]) -> AgentProtocol:
        """Create agent with specified trait values."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConformityTestResult:
    """Result of a conformity test."""

    archetype_name: str
    instrument: str
    assessment: AssessmentResult
    conformity: ConformityResult
    passed: bool
    tolerance: float  # tolerance used for this test
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_name": self.archetype_name,
            "instrument": self.instrument,
            "assessment": self.assessment.to_dict(),
            "conformity": {
                "correlation": self.conformity.correlation,
                "within_tolerance": self.conformity.within_tolerance,
                "all_within_tolerance": self.conformity.all_within_tolerance,
                "deviations": self.conformity.deviations,
                "tolerance": self.conformity.tolerance,
            },
            "passed": self.passed,
            "tolerance": self.tolerance,
            "timestamp": self.timestamp,
        }


@dataclass
class ConsistencyTestResult:
    """Result of a consistency test."""

    archetype_name: str
    instrument: str
    n_turns: int
    sample_points: list[int]
    assessments: list[AssessmentResult]
    metrics: ConsistencyMetrics
    drift: DriftAnalysis
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_name": self.archetype_name,
            "instrument": self.instrument,
            "n_turns": self.n_turns,
            "sample_points": self.sample_points,
            "n_assessments": len(self.assessments),
            "metrics": {
                "icc": self.metrics.icc,
                "icc_interpretation": self.metrics.icc_interpretation,
                "mean_drift": self.metrics.mean_absolute_drift,
                "max_drift": self.metrics.max_drift,
            },
            "drift": {
                "initial": self.drift.initial_scores,
                "final": self.drift.final_scores,
                "per_trait": self.drift.drift_per_trait,
                "direction": self.drift.drift_direction,
            },
            "passed": self.passed,
            "timestamp": self.timestamp,
        }


@dataclass
class NRunResult:
    """Result of N repeated conformity tests for statistical validity."""

    archetype_name: str
    instrument: str
    n_runs: int
    results: list[ConformityTestResult]
    pass_rate: float
    mean_correlation: float
    std_correlation: float
    correlations: list[float]
    tolerance: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_name": self.archetype_name,
            "instrument": self.instrument,
            "n_runs": self.n_runs,
            "pass_rate": self.pass_rate,
            "mean_correlation": self.mean_correlation,
            "std_correlation": self.std_correlation,
            "correlations": self.correlations,
            "tolerance": self.tolerance,
            "n_passed": sum(1 for r in self.results if r.passed),
            "n_failed": sum(1 for r in self.results if not r.passed),
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Return a formatted summary string."""
        n_passed = sum(1 for r in self.results if r.passed)
        return (
            f"{self.archetype_name} ({self.n_runs} runs): "
            f"pass_rate={self.pass_rate:.1%}, "
            f"r={self.mean_correlation:.3f}±{self.std_correlation:.3f}"
            f"having passed {n_passed} out of {self.n_runs} tests."
        )


@dataclass
class BaselineResult:
    """Result of baseline (non-RnB) agent test."""

    baseline_type: str  # "pure_prompt" or "few_shot"
    archetype_name: str
    instrument: str
    assessments: list[AssessmentResult]
    metrics: ConsistencyMetrics | None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Conversation Prompts
# =============================================================================

# Default conversation prompts for consistency testing
# These are neutral, everyday topics that shouldn't bias personality expression
DEFAULT_CONVERSATION_PROMPTS = [
    "Can you explain how to make a good cup of coffee?",
    "What are some tips for staying organized?",
    "Tell me about an interesting historical event.",
    "How would you approach learning a new skill?",
    "What makes a good book recommendation?",
    "Can you help me plan a weekend trip?",
    "What's a good way to start exercising?",
    "How do you think about solving complex problems?",
    "What makes a conversation engaging?",
    "Can you explain a scientific concept in simple terms?",
    "What's your perspective on work-life balance?",
    "How would you approach a disagreement with someone?",
    "What makes a good team member?",
    "Can you help me understand a difficult topic?",
    "What's important when making decisions?",
    "How do you stay motivated on long projects?",
    "What makes communication effective?",
    "Can you explain how to budget money?",
    "What's a good approach to time management?",
    "How would you handle a stressful situation?",
]


def get_conversation_prompts(n: int) -> list[str]:
    """Get conversation prompts, cycling if needed."""
    prompts = []
    for i in range(n):
        prompts.append(
            DEFAULT_CONVERSATION_PROMPTS[i % len(DEFAULT_CONVERSATION_PROMPTS)]
        )
    return prompts


# =============================================================================
# Validation Runner
# =============================================================================


class ValidationRunner:
    """
    Orchestrates personality validation experiments.

    Supports:
    - Conformity testing against archetypes
    - Consistency testing over extended conversations
    - Baseline comparisons (pure prompt, few-shot)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        agent_factory: AgentFactory | None = None,
        instruments_dir: Path | None = None,
        archetypes_path: Path | None = None,
        conformity_threshold: float = 0.7,
        consistency_threshold: float = 0.85,
        tolerance: float = 0.25,
    ):
        """
        Initialize validation runner.

        Args:
            llm_client: LLM client for assessments
            agent_factory: Factory for creating RnB agents
            instruments_dir: Directory containing instrument YAML files
            archetypes_path: Path to archetypes YAML file
            conformity_threshold: Minimum correlation for conformity pass
            consistency_threshold: Minimum ICC for consistency pass
            tolerance: Tolerance for trait value deviation on RnB scale [-1,1]
        """
        from .assessor import get_archetypes_path, get_instruments_dir

        self.llm_client = llm_client
        self.agent_factory = agent_factory

        self.instruments_dir = instruments_dir or get_instruments_dir()
        self.archetypes_path = archetypes_path or get_archetypes_path()

        self.conformity_threshold = conformity_threshold
        self.consistency_threshold = consistency_threshold
        self.tolerance = tolerance

        # Load archetypes
        self.archetypes = load_archetypes(self.archetypes_path)

        # Cache assessors
        self._assessors: dict[str, PersonalityAssessor] = {}

    def get_assessor(self, instrument: str) -> PersonalityAssessor:
        """Get or create assessor for instrument."""
        if instrument not in self._assessors:
            self._assessors[instrument] = PersonalityAssessor.from_instrument_name(
                instrument, self.instruments_dir
            )
        return self._assessors[instrument]

    def test_conformity(
        self,
        archetype_name: str,
        instrument: str = "tipi",
        agent: AgentProtocol | None = None,
        warm_up_turns: int = 5,
        tolerance: float | None = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> ConformityTestResult:
        """
        Test personality conformity against an archetype.

        Args:
            archetype_name: Name of archetype to test against
            instrument: Assessment instrument ("tipi" or "bfi2s")
            agent: Optional pre-created agent (else created from factory)
            warm_up_turns: Number of conversation turns before assessment
            tolerance: Override default tolerance for this test
            max_retries: Number of retry attempts for invalid LLM responses
            verbose: If True, log raw LLM responses

        Returns:
            ConformityTestResult with assessment and conformity metrics
        """
        if archetype_name not in self.archetypes:
            raise ValueError(f"Unknown archetype: {archetype_name}")

        archetype = self.archetypes[archetype_name]
        tol = tolerance if tolerance is not None else self.tolerance

        # Create agent if not provided
        if agent is None:
            if self.agent_factory is None:
                raise ValueError("No agent provided and no agent_factory configured")
            agent = self.agent_factory.from_archetype(archetype_name)

        # Warm-up conversation
        prompts = get_conversation_prompts(warm_up_turns)
        for prompt in prompts:
            _ = agent.respond(prompt)

        # Get assessor and run assessment
        assessor = self.get_assessor(instrument)

        # Use agent's system prompt for assessment context
        system_prompt = agent.get_system_prompt()
        assessment = assessor.assess(
            self.llm_client,
            system_prompt=system_prompt,
            max_retries=max_retries,
            verbose=verbose,
        )

        # Check conformity with tolerance parameter
        conformity = assessor.check_conformity(assessment, archetype, tolerance=tol)

        # Determine pass/fail
        passed = (
            conformity.correlation >= self.conformity_threshold
            and conformity.all_within_tolerance
        )

        return ConformityTestResult(
            archetype_name=archetype_name,
            instrument=instrument,
            assessment=assessment,
            conformity=conformity,
            passed=passed,
            tolerance=tol,
        )

    def test_conformity_n_runs(
        self,
        archetype_name: str,
        n_runs: int = 10,
        instrument: str = "tipi",
        warm_up_turns: int = 5,
        tolerance: float | None = None,
        max_retries: int = 3,
        verbose: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NRunResult:
        """
        Run conformity test N times to assess statistical reliability.

        This addresses natural variability in LLM responses by running
        multiple tests and computing aggregate statistics.

        Args:
            archetype_name: Name of archetype to test against
            n_runs: Number of test runs (default 10, recommended 10-100)
            instrument: Assessment instrument ("tipi" or "bfi2s")
            warm_up_turns: Number of conversation turns before each assessment
            tolerance: Override default tolerance for this test
            max_retries: Number of retry attempts per run
            verbose: If True, log raw LLM responses
            progress_callback: Optional callback(current_run, total_runs)

        Returns:
            NRunResult with aggregate statistics
        """
        import statistics

        results: list[ConformityTestResult] = []
        correlations: list[float] = []
        tol = tolerance if tolerance is not None else self.tolerance

        for i in range(n_runs):
            if progress_callback:
                progress_callback(i + 1, n_runs)

            try:
                result = self.test_conformity(
                    archetype_name=archetype_name,
                    instrument=instrument,
                    warm_up_turns=warm_up_turns,
                    tolerance=tol,
                    max_retries=max_retries,
                    verbose=verbose,
                )
                results.append(result)
                correlations.append(result.conformity.correlation)
            except Exception as e:
                self.logger.warning(f"Run {i+1}/{n_runs} failed: {e}")
                # Continue with remaining runs

        if not results:
            raise ValueError(f"All {n_runs} runs failed")

        # Calculate statistics
        n_passed = sum(1 for r in results if r.passed)
        pass_rate = n_passed / len(results)
        mean_corr = statistics.mean(correlations)
        std_corr = statistics.stdev(correlations) if len(correlations) > 1 else 0.0

        return NRunResult(
            archetype_name=archetype_name,
            instrument=instrument,
            n_runs=len(results),
            results=results,
            pass_rate=pass_rate,
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            correlations=correlations,
            tolerance=tol,
        )

    def test_consistency(
        self,
        archetype_name: str,
        n_turns: int = 100,
        sample_every: int = 25,
        instrument: str = "tipi",
        agent: AgentProtocol | None = None,
        conversation_prompts: list[str] | None = None,
    ) -> ConsistencyTestResult:
        """
        Test personality consistency over extended conversation.

        Args:
            archetype_name: Name of archetype to use
            n_turns: Total conversation turns
            sample_every: Take assessment every N turns
            instrument: Assessment instrument
            agent: Optional pre-created agent
            conversation_prompts: Optional custom prompts

        Returns:
            ConsistencyTestResult with metrics and drift analysis
        """
        if archetype_name not in self.archetypes:
            raise ValueError(f"Unknown archetype: {archetype_name}")

        _archetype = self.archetypes[archetype_name]

        # Create agent if not provided
        if agent is None:
            if self.agent_factory is None:
                raise ValueError("No agent provided and no agent_factory configured")
            agent = self.agent_factory.from_archetype(archetype_name)

        # Get conversation prompts
        if conversation_prompts is None:
            conversation_prompts = get_conversation_prompts(n_turns)

        # Get assessor
        assessor = self.get_assessor(instrument)
        system_prompt = agent.get_system_prompt()

        # Determine sample points
        sample_points = [0]  # Always sample at start
        for i in range(sample_every, n_turns, sample_every):
            sample_points.append(i)
        if n_turns not in sample_points:
            sample_points.append(n_turns)

        # Run conversation with periodic assessments
        assessments = []

        # Initial assessment
        logger.info("Taking initial assessment at turn 0")
        assessment = assessor.assess(self.llm_client, system_prompt=system_prompt)
        assessments.append(assessment)

        # Conversation loop
        for turn in range(n_turns):
            prompt = conversation_prompts[turn % len(conversation_prompts)]
            _ = agent.respond(prompt)

            # Check if we should assess
            if (turn + 1) in sample_points and (turn + 1) != 0:
                logger.info(f"Taking assessment at turn {turn + 1}")
                assessment = assessor.assess(
                    self.llm_client, system_prompt=system_prompt
                )
                assessments.append(assessment)

        # Calculate metrics
        rnb_scores_list = [a.rnb_scores for a in assessments if a.rnb_scores]

        metrics = calculate_consistency_metrics(
            rnb_scores_list, sample_points[: len(assessments)]
        )
        drift = calculate_drift(rnb_scores_list, sample_points[: len(assessments)])

        # Determine pass/fail
        passed = metrics.icc >= self.consistency_threshold

        return ConsistencyTestResult(
            archetype_name=archetype_name,
            instrument=instrument,
            n_turns=n_turns,
            sample_points=sample_points[: len(assessments)],
            assessments=assessments,
            metrics=metrics,
            drift=drift,
            passed=passed,
        )

    def run_full_validation(
        self,
        archetype_names: list[str] | None = None,
        instruments: list[str] | None = None,
        include_consistency: bool = True,
        consistency_turns: int = 100,
    ) -> dict[str, Any]:
        """
        Run full validation suite on multiple archetypes.

        Args:
            archetype_names: Archetypes to test (default: all)
            instruments: Instruments to use (default: ["tipi", "bfi2s"])
            include_consistency: Whether to run consistency tests
            consistency_turns: Turns for consistency tests

        Returns:
            Dict with all results and summary statistics
        """
        if archetype_names is None:
            archetype_names = list(self.archetypes.keys())

        if instruments is None:
            instruments = ["tipi", "bfi2s"]

        results = {
            "conformity": {},
            "consistency": {},
            "summary": {},
        }

        # Run conformity tests
        for archetype in archetype_names:
            results["conformity"][archetype] = {}

            for instrument in instruments:
                logger.info(f"Testing conformity: {archetype} with {instrument}")
                try:
                    result = self.test_conformity(archetype, instrument)
                    results["conformity"][archetype][instrument] = result.to_dict()
                except Exception as e:
                    logger.error(f"Conformity test failed: {e}")
                    results["conformity"][archetype][instrument] = {"error": str(e)}

        # Run consistency tests
        if include_consistency:
            for archetype in archetype_names:
                results["consistency"][archetype] = {}

                for instrument in instruments:
                    logger.info(f"Testing consistency: {archetype} with {instrument}")
                    try:
                        result = self.test_consistency(
                            archetype, n_turns=consistency_turns, instrument=instrument
                        )
                        results["consistency"][archetype][instrument] = result.to_dict()
                    except Exception as e:
                        logger.error(f"Consistency test failed: {e}")
                        results["consistency"][archetype][instrument] = {
                            "error": str(e)
                        }

        # Calculate summary
        conformity_passed = 0
        conformity_total = 0
        consistency_passed = 0
        consistency_total = 0

        for archetype in results["conformity"]:
            for instrument in results["conformity"][archetype]:
                r = results["conformity"][archetype][instrument]
                if "passed" in r:
                    conformity_total += 1
                    if r["passed"]:
                        conformity_passed += 1

        for archetype in results.get("consistency", {}):
            for instrument in results["consistency"][archetype]:
                r = results["consistency"][archetype][instrument]
                if "passed" in r:
                    consistency_total += 1
                    if r["passed"]:
                        consistency_passed += 1

        results["summary"] = {
            "conformity_pass_rate": (
                conformity_passed / conformity_total if conformity_total > 0 else 0
            ),
            "consistency_pass_rate": (
                consistency_passed / consistency_total if consistency_total > 0 else 0
            ),
            "archetypes_tested": archetype_names,
            "instruments_used": instruments,
            "timestamp": datetime.now().isoformat(),
        }

        return results


# =============================================================================
# Baseline Testing
# =============================================================================


def generate_pure_prompt_system(archetype: ArchetypeConfig) -> str:
    """Generate a pure prompt baseline system message."""
    trait_descriptions = {
        "openness": (
            "open to new experiences",
            "conventional, preferring familiar approaches",
        ),
        "conscientiousness": ("organized and dependable", "spontaneous and flexible"),
        "extraversion": ("outgoing and enthusiastic", "reserved and quiet"),
        "agreeableness": ("cooperative and warm", "independent and straightforward"),
        "neuroticism": ("sensitive and prone to worry", "calm and emotionally stable"),
    }

    lines = ["You are an AI assistant with the following personality traits:"]

    for trait, value in archetype.traits.items():
        if trait in trait_descriptions:
            high_desc, low_desc = trait_descriptions[trait]
            if value > 0.3:
                lines.append(f"- {high_desc.capitalize()}")
            elif value < -0.3:
                lines.append(f"- {low_desc.capitalize()}")

    return "\n".join(lines)


def run_baseline_test(
    llm_client: LLMClient,
    archetype: ArchetypeConfig,
    instrument: str = "tipi",
    n_assessments: int = 3,
    instruments_dir: Path | None = None,
) -> BaselineResult:
    """
    Run baseline test using pure prompt approach (no RnB).

    Args:
        llm_client: LLM client
        archetype: Archetype to simulate
        instrument: Assessment instrument
        n_assessments: Number of assessments to run

    Returns:
        BaselineResult with assessment data
    """
    assessor = PersonalityAssessor.from_instrument_name(instrument, instruments_dir)
    system_prompt = generate_pure_prompt_system(archetype)

    assessments = []
    for _ in range(n_assessments):
        assessment = assessor.assess(llm_client, system_prompt=system_prompt)
        assessments.append(assessment)

    # Calculate consistency if multiple assessments
    metrics = None
    if len(assessments) >= 2:
        rnb_scores = [a.rnb_scores for a in assessments if a.rnb_scores]
        if len(rnb_scores) >= 2:
            metrics = calculate_consistency_metrics(rnb_scores)

    return BaselineResult(
        baseline_type="pure_prompt",
        archetype_name=archetype.name,
        instrument=instrument,
        assessments=assessments,
        metrics=metrics,
    )
