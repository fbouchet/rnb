"""
RnB Personality Validation Module

Provides tools for validating RnB agent personalities using standardized
psychological instruments (TIPI, BFI-2-S).

Quick Start:
    from rnb.validation import create_validation_runner, run_quick_validation

    # Quick validation
    results = run_quick_validation("resilient", verbose=True)

    # Or with full control
    runner = create_validation_runner(
        provider="local",
        model_name="llama3.2:3b"
    )
    result = runner.test_conformity("resilient", instrument="tipi")
    print(f"Correlation: {result.conformity.correlation:.2f}")

Components:
    - PersonalityAssessor: Core assessment logic
    - ValidationRunner: Orchestrates validation experiments
    - Metrics: Statistical analysis using pingouin (ICC, correlation, drift)
    - Integration: Bridges to RnB LLMClient and AgentManager
    - Instruments: TIPI, BFI-2-S in YAML format (in resources/instruments/)
    - Archetypes: Literature-grounded RUO profiles (in resources/data/)

References:
    TIPI: Gosling, S. D., Rentfrow, P. J., & Swann, W. B., Jr. (2003).
    BFI-2: Soto, C. J., & John, O. P. (2017).
    RUO Archetypes: Robins et al. (1996); Asendorpf et al. (2001).
    Pingouin: Vallat, R. (2018). Journal of Open Source Software, 3(31), 1026.
"""

from pathlib import Path

from .assessor import (
    ArchetypeConfig,
    AssessmentResult,
    ConformityResult,
    Instrument,
    # Data classes
    Item,
    # Main classes
    PersonalityAssessor,
    ScaleConfig,
    ScoringConfig,
    calculate_domain_score,
    convert_from_rnb_scale,
    convert_to_rnb_scale,
    generate_batch_prompt,
    generate_single_item_prompt,
    get_archetypes_path,
    get_instruments_dir,
    get_resources_dir,
    list_available_archetypes,
    list_available_instruments,
    load_archetypes,
    # Functions
    load_instrument,
    parse_batch_response,
    parse_single_response,
    reverse_score,
    set_resources_dir,
)
from .integration import (
    # Adapters
    LLMClientAdapter,
    RnBAgent,
    RnBAgentFactory,
    # Convenience functions
    create_validation_runner,
    run_quick_validation,
)
from .metrics import (
    # Data classes
    ConsistencyMetrics,
    CorrelationResult,
    DriftAnalysis,
    calculate_conformity_score,
    calculate_consistency_metrics,
    calculate_drift,
    calculate_expected_range,
    calculate_expected_score,
    calculate_icc,
    calculate_per_trait_stability,
    check_range_conformity,
    # Functions
    pearson_correlation,
    spearman_correlation,
    summarize_validation_run,
)
from .runner import (
    DEFAULT_CONVERSATION_PROMPTS,
    BaselineResult,
    # Data classes
    ConformityTestResult,
    ConsistencyTestResult,
    # Main class
    ValidationRunner,
    generate_pure_prompt_system,
    # Functions
    get_conversation_prompts,
    run_baseline_test,
)

__all__ = [
    # Assessor
    "Item",
    "ScaleConfig",
    "ScoringConfig",
    "Instrument",
    "AssessmentResult",
    "ConformityResult",
    "ArchetypeConfig",
    "PersonalityAssessor",
    "load_instrument",
    "load_archetypes",
    "generate_batch_prompt",
    "generate_single_item_prompt",
    "parse_batch_response",
    "parse_single_response",
    "convert_to_rnb_scale",
    "convert_from_rnb_scale",
    "reverse_score",
    "calculate_domain_score",
    "list_available_instruments",
    "list_available_archetypes",
    "get_instruments_dir",
    "get_archetypes_path",
    "get_resources_dir",
    "set_resources_dir",
    # Metrics
    "ConsistencyMetrics",
    "DriftAnalysis",
    "CorrelationResult",
    "pearson_correlation",
    "spearman_correlation",
    "calculate_icc",
    "calculate_per_trait_stability",
    "calculate_drift",
    "calculate_consistency_metrics",
    "calculate_conformity_score",
    "check_range_conformity",
    "calculate_expected_score",
    "calculate_expected_range",
    "summarize_validation_run",
    # Runner
    "ConformityTestResult",
    "ConsistencyTestResult",
    "BaselineResult",
    "ValidationRunner",
    "get_conversation_prompts",
    "generate_pure_prompt_system",
    "run_baseline_test",
    "DEFAULT_CONVERSATION_PROMPTS",
    # Integration
    "LLMClientAdapter",
    "RnBAgent",
    "RnBAgentFactory",
    "create_validation_runner",
    "run_quick_validation",
]


# Package metadata
__version__ = "0.1.0"
__author__ = "RnB Framework Contributors"


def get_validation_dir() -> Path:
    """Get the validation module directory."""
    return Path(__file__).parent
