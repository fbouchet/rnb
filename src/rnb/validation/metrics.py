"""
RnB Validation Metrics Module

Provides statistical metrics for personality validation:
- Intraclass Correlation Coefficient (ICC) for consistency
- Pearson/Spearman correlation for conformity
- Drift metrics for stability analysis

Uses pingouin library for validated statistical calculations.

References:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in
    assessing rater reliability. Psychological Bulletin, 86(2), 420-428.

    Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source
    Software, 3(31), 1026.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pingouin as pg

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConsistencyMetrics:
    """Metrics for personality consistency over time."""

    icc: float  # Intraclass Correlation Coefficient
    icc_type: str  # ICC type (e.g., "ICC3")
    icc_interpretation: str  # "excellent", "good", "moderate", "poor"
    icc_ci_lower: float  # 95% CI lower bound
    icc_ci_upper: float  # 95% CI upper bound
    mean_absolute_drift: float  # Average |final - initial| across traits
    max_drift: float  # Maximum single-trait drift
    trait_stability: dict[str, float]  # Per-trait stability scores
    within_subject_sd: dict[str, float]  # Per-trait standard deviation
    n_measurements: int
    n_traits: int


@dataclass
class DriftAnalysis:
    """Analysis of personality drift over conversation."""

    initial_scores: dict[str, float]
    final_scores: dict[str, float]
    trajectory: dict[str, list[float]]  # trait -> scores at each measurement
    drift_per_trait: dict[str, float]  # trait -> (final - initial)
    drift_direction: dict[str, str]  # trait -> "stable", "increasing", "decreasing"
    total_drift: float  # Sum of absolute drifts
    measurement_points: list[int]  # Turn numbers where measurements were taken


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""

    method: str  # "pearson" or "spearman"
    r: float  # Correlation coefficient
    p_value: float  # p-value
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n: int  # Sample size


# =============================================================================
# Correlation Functions (using pingouin)
# =============================================================================


def pearson_correlation(
    x: Sequence[float], y: Sequence[float], return_details: bool = False
) -> float | CorrelationResult:
    """
    Calculate Pearson correlation coefficient using pingouin.

    Args:
        x: First sequence of values
        y: Second sequence of values (same length as x)
        return_details: If True, return full CorrelationResult with CI and p-value

    Returns:
        Pearson's r, or CorrelationResult if return_details=True
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"Sequences must have same length: {len(x_arr)} vs {len(y_arr)}"
        )
    if len(x_arr) < 2:
        if return_details:
            return CorrelationResult("pearson", 0.0, 1.0, -1.0, 1.0, len(x_arr))
        return 0.0

    # Use pingouin.corr for correlation with CI
    result = pg.corr(x_arr, y_arr, method="pearson")

    r = float(result["r"].values[0])
    ci = result["CI95%"].values[0]

    if return_details:
        return CorrelationResult(
            method="pearson",
            r=r,
            p_value=float(result["p-val"].values[0]),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            n=len(x_arr),
        )

    return r


def spearman_correlation(
    x: Sequence[float], y: Sequence[float], return_details: bool = False
) -> float | CorrelationResult:
    """
    Calculate Spearman rank correlation coefficient using pingouin.

    Args:
        x: First sequence of values
        y: Second sequence of values (same length as x)
        return_details: If True, return full CorrelationResult with CI and p-value

    Returns:
        Spearman's rho, or CorrelationResult if return_details=True
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"Sequences must have same length: {len(x_arr)} vs {len(y_arr)}"
        )
    if len(x_arr) < 2:
        if return_details:
            return CorrelationResult("spearman", 0.0, 1.0, -1.0, 1.0, len(x_arr))
        return 0.0

    # Use pingouin.corr for correlation with CI
    result = pg.corr(x_arr, y_arr, method="spearman")

    r = float(result["r"].values[0])
    ci = result["CI95%"].values[0]

    if return_details:
        return CorrelationResult(
            method="spearman",
            r=r,
            p_value=float(result["p-val"].values[0]),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            n=len(x_arr),
        )

    return r


# =============================================================================
# Intraclass Correlation Coefficient (using pingouin)
# =============================================================================


def calculate_icc(
    measurements: list[dict[str, float]],
    icc_type: Literal["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"] = "ICC3",
) -> tuple[float, str, float, float]:
    """
    Calculate Intraclass Correlation Coefficient for repeated measurements
    using pingouin.

    Args:
        measurements: List of dicts, each containing trait->score mappings
                     for one measurement occasion
        icc_type: Type of ICC to calculate:
            - ICC1: One-way random effects, single rater, absolute agreement
            - ICC2: Two-way random effects, single rater, absolute agreement
            - ICC3: Two-way mixed effects, single rater, consistency (recommended)
            - ICC1k, ICC2k, ICC3k: Same but for average of k raters

    Returns:
        Tuple of (ICC value, interpretation, CI lower, CI upper)

    Interpretation (Koo & Li, 2016):
        < 0.50: poor
        0.50 - 0.75: moderate
        0.75 - 0.90: good
        > 0.90: excellent

    References:
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations.
        Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and
        reporting intraclass correlation coefficients for reliability research.
    """
    if len(measurements) < 2:
        return 0.0, "insufficient_data", 0.0, 0.0

    # Get all traits present in all measurements
    traits = set(measurements[0].keys())
    for m in measurements[1:]:
        traits &= set(m.keys())
    traits = sorted(traits)

    if len(traits) < 1:
        return 0.0, "no_common_traits", 0.0, 0.0

    # Build long-format DataFrame for pingouin
    # Columns: targets (traits), raters (measurement occasions), ratings (scores)
    data = []
    for rater_idx, measurement in enumerate(measurements):
        for trait in traits:
            data.append(
                {
                    "targets": trait,
                    "raters": f"measurement_{rater_idx}",
                    "ratings": measurement[trait],
                }
            )

    df = pd.DataFrame(data)

    # Calculate ICC using pingouin
    try:
        icc_result = pg.intraclass_corr(
            data=df, targets="targets", raters="raters", ratings="ratings"
        )

        # Find the requested ICC type
        icc_row = icc_result[icc_result["Type"] == icc_type]

        if len(icc_row) == 0:
            # Fall back to ICC3 if specific type not found
            icc_row = icc_result[icc_result["Type"] == "ICC3"]

        if len(icc_row) == 0:
            return 0.0, "type_not_found", 0.0, 0.0

        icc = float(icc_row["ICC"].values[0])
        ci = icc_row["CI95%"].values[0]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])

    except Exception as e:
        warnings.warn(f"ICC calculation failed: {e}. Returning 0.", stacklevel=2)
        return 0.0, "calculation_error", 0.0, 0.0

    # Interpret ICC (Koo & Li, 2016 guidelines)
    if icc < 0.50:
        interpretation = "poor"
    elif icc < 0.75:
        interpretation = "moderate"
    elif icc < 0.90:
        interpretation = "good"
    else:
        interpretation = "excellent"

    return icc, interpretation, ci_lower, ci_upper


def calculate_per_trait_stability(
    measurements: list[dict[str, float]],
) -> dict[str, float]:
    """
    Calculate stability score for each trait separately.

    Uses coefficient of variation (CV) inverted to a stability measure:
    stability = 1 - (CV / max_expected_CV)

    For RnB scale [-1, 1], max range = 2, so max_expected_CV ≈ 1.0

    Returns:
        Dict mapping trait -> stability score [0, 1] where 1 = perfect stability
    """
    if len(measurements) < 2:
        return {}

    # Get all traits
    traits = set(measurements[0].keys())
    for m in measurements[1:]:
        traits &= set(m.keys())

    result = {}
    for trait in traits:
        values = [m[trait] for m in measurements]

        # Calculate standard deviation
        std_val = float(np.std(values, ddof=1))  # Sample std

        if std_val == 0:
            result[trait] = 1.0  # Perfect stability
        else:
            # Stability based on range relative to scale
            value_range = max(values) - min(values)
            # For RnB scale [-1, 1], total range is 2
            result[trait] = max(0.0, 1.0 - (value_range / 2.0))

    return result


# =============================================================================
# Drift Analysis
# =============================================================================


def calculate_drift(
    measurements: list[dict[str, float]], measurement_points: list[int] | None = None
) -> DriftAnalysis:
    """
    Analyze personality drift over a series of measurements.

    Args:
        measurements: List of score dicts at each measurement point
        measurement_points: Optional list of turn numbers for each measurement

    Returns:
        DriftAnalysis with detailed drift metrics
    """
    if len(measurements) < 2:
        raise ValueError("Need at least 2 measurements for drift analysis")

    # Get common traits
    traits = set(measurements[0].keys())
    for m in measurements[1:]:
        traits &= set(m.keys())
    traits = sorted(traits)

    initial = measurements[0]
    final = measurements[-1]

    # Build trajectory
    trajectory = {trait: [m.get(trait, 0) for m in measurements] for trait in traits}

    # Calculate drift per trait
    drift_per_trait = {}
    drift_direction = {}

    for trait in traits:
        drift = final.get(trait, 0) - initial.get(trait, 0)
        drift_per_trait[trait] = drift

        if abs(drift) < 0.1:  # Threshold for "stable"
            drift_direction[trait] = "stable"
        elif drift > 0:
            drift_direction[trait] = "increasing"
        else:
            drift_direction[trait] = "decreasing"

    total_drift = sum(abs(d) for d in drift_per_trait.values())

    if measurement_points is None:
        measurement_points = list(range(len(measurements)))

    return DriftAnalysis(
        initial_scores={t: initial.get(t, 0) for t in traits},
        final_scores={t: final.get(t, 0) for t in traits},
        trajectory=trajectory,
        drift_per_trait=drift_per_trait,
        drift_direction=drift_direction,
        total_drift=total_drift,
        measurement_points=measurement_points,
    )


def calculate_consistency_metrics(
    measurements: list[dict[str, float]],
    measurement_points: list[int] | None = None,
    icc_type: str = "ICC3",
) -> ConsistencyMetrics:
    """
    Calculate comprehensive consistency metrics from repeated measurements.

    Args:
        measurements: List of score dicts at each measurement point
        measurement_points: Optional list of turn numbers
        icc_type: Type of ICC to calculate (default: ICC3)

    Returns:
        ConsistencyMetrics with ICC, drift, and stability measures
    """
    if len(measurements) < 2:
        raise ValueError("Need at least 2 measurements for consistency analysis")

    # Get common traits
    traits = set(measurements[0].keys())
    for m in measurements[1:]:
        traits &= set(m.keys())
    traits = sorted(traits)

    # Calculate overall ICC
    icc, icc_interp, ci_lower, ci_upper = calculate_icc(measurements, icc_type)

    # Calculate per-trait stability
    trait_stability = calculate_per_trait_stability(measurements)

    # Calculate within-subject standard deviation
    within_sd = {}
    for trait in traits:
        values = [m.get(trait, 0) for m in measurements]
        within_sd[trait] = float(np.std(values, ddof=1))

    # Calculate drift metrics
    initial = measurements[0]
    final = measurements[-1]

    drifts = []
    for trait in traits:
        drift = abs(final.get(trait, 0) - initial.get(trait, 0))
        drifts.append(drift)

    mean_drift = sum(drifts) / len(drifts) if drifts else 0
    max_drift = max(drifts) if drifts else 0

    return ConsistencyMetrics(
        icc=icc,
        icc_type=icc_type,
        icc_interpretation=icc_interp,
        icc_ci_lower=ci_lower,
        icc_ci_upper=ci_upper,
        mean_absolute_drift=mean_drift,
        max_drift=max_drift,
        trait_stability=trait_stability,
        within_subject_sd=within_sd,
        n_measurements=len(measurements),
        n_traits=len(traits),
    )


# =============================================================================
# Conformity Metrics
# =============================================================================


def calculate_conformity_score(
    designed: dict[str, float],
    measured: dict[str, float],
    method: Literal["pearson", "spearman"] = "pearson",
) -> tuple[float, dict[str, float], CorrelationResult | None]:
    """
    Calculate conformity between designed and measured personality.

    Args:
        designed: Designed trait values (RnB scale)
        measured: Measured trait values (RnB scale)
        method: Correlation method ("pearson" or "spearman")

    Returns:
        Tuple of (correlation, per-trait deviations, full correlation result)
    """
    common_traits = set(designed.keys()) & set(measured.keys())

    if len(common_traits) < 2:
        return 0.0, {}, None

    # Extract aligned values
    traits_sorted = sorted(common_traits)
    designed_vals = [designed[t] for t in traits_sorted]
    measured_vals = [measured[t] for t in traits_sorted]

    # Calculate correlation using appropriate method
    if method == "spearman":
        corr_result = spearman_correlation(
            designed_vals, measured_vals, return_details=True
        )
    else:
        corr_result = pearson_correlation(
            designed_vals, measured_vals, return_details=True
        )

    # Calculate per-trait deviations
    deviations = {t: measured[t] - designed[t] for t in common_traits}

    return corr_result.r, deviations, corr_result


def check_range_conformity(
    measured_rnb: dict[str, float],
    designed_rnb: dict[str, float],
    tolerance: float = 0.25,
) -> tuple[bool, dict[str, bool], dict[str, tuple[float, float]]]:
    """
    Check if measured RnB scores fall within tolerance of designed values.

    This replaces hardcoded expected ranges with dynamic calculation
    based on a tolerance parameter.

    Args:
        measured_rnb: Measured scores on RnB scale [-1, 1]
        designed_rnb: Designed trait values on RnB scale [-1, 1]
        tolerance: Acceptable deviation from designed value (default: 0.25)
                   A tolerance of 0.25 means ±0.25 on RnB scale,
                   which corresponds to ~0.75 points on TIPI (1-7)
                   or ~0.5 points on BFI-2-S (1-5)

    Returns:
        Tuple of (all_within_tolerance, per_trait_within, expected_ranges)
    """
    within_tolerance = {}
    expected_ranges = {}

    common_traits = set(measured_rnb.keys()) & set(designed_rnb.keys())

    for trait in common_traits:
        designed = designed_rnb[trait]
        measured = measured_rnb[trait]

        # Calculate expected range
        lo = max(-1.0, designed - tolerance)
        hi = min(1.0, designed + tolerance)
        expected_ranges[trait] = (lo, hi)

        # Check if within tolerance
        within_tolerance[trait] = lo <= measured <= hi

    all_within = all(within_tolerance.values()) if within_tolerance else False

    return all_within, within_tolerance, expected_ranges


def calculate_expected_score(
    rnb_value: float, instrument: Literal["tipi", "bfi2s"], trait: str = ""
) -> float:
    """
    Convert RnB trait value to expected instrument score.

    Args:
        rnb_value: Trait value on RnB scale [-1, 1]
        instrument: Target instrument ("tipi" or "bfi2s")
        trait: Trait name (needed for neuroticism handling)

    Returns:
        Expected score on instrument scale
    """
    if instrument == "tipi":
        # TIPI: 1-7 scale, midpoint=4, half_range=3
        # For neuroticism, TIPI measures Emotional Stability (inverse)
        if trait == "neuroticism":
            rnb_value = -rnb_value  # Invert for emotional stability
        return 4.0 + (rnb_value * 3.0)

    elif instrument == "bfi2s":
        # BFI-2-S: 1-5 scale, midpoint=3, half_range=2
        # BFI-2-S uses Negative Emotionality directly (no inversion)
        return 3.0 + (rnb_value * 2.0)

    else:
        raise ValueError(f"Unknown instrument: {instrument}")


def calculate_expected_range(
    rnb_value: float,
    instrument: Literal["tipi", "bfi2s"],
    tolerance: float = 0.25,
    trait: str = "",
) -> tuple[float, float]:
    """
    Calculate expected score range on instrument scale.

    Args:
        rnb_value: Trait value on RnB scale [-1, 1]
        instrument: Target instrument
        tolerance: Tolerance on RnB scale (default: 0.25)
        trait: Trait name (for neuroticism handling)

    Returns:
        Tuple of (lower_bound, upper_bound) on instrument scale
    """
    # Calculate bounds on RnB scale
    rnb_lo = max(-1.0, rnb_value - tolerance)
    rnb_hi = min(1.0, rnb_value + tolerance)

    # Convert to instrument scale
    score_lo = calculate_expected_score(rnb_lo, instrument, trait)
    score_hi = calculate_expected_score(rnb_hi, instrument, trait)

    # Ensure lo <= hi (may be swapped for inverted traits)
    return (min(score_lo, score_hi), max(score_lo, score_hi))


# =============================================================================
# Summary Statistics
# =============================================================================


def summarize_validation_run(
    conformity_results: list[
        tuple[str, float, bool]
    ],  # (archetype, correlation, in_range)
    consistency_results: list[ConsistencyMetrics] | None = None,
) -> dict[str, any]:
    """
    Summarize results from a validation run.

    Args:
        conformity_results: List of (archetype_name, correlation, all_in_range)
        consistency_results: Optional list of consistency metrics

    Returns:
        Summary dict with aggregate statistics
    """
    summary = {
        "conformity": {
            "n_archetypes_tested": len(conformity_results),
            "mean_correlation": 0.0,
            "min_correlation": 0.0,
            "max_correlation": 0.0,
            "std_correlation": 0.0,
            "n_all_in_range": 0,
            "pass_rate": 0.0,
        }
    }

    if conformity_results:
        correlations = np.array([r[1] for r in conformity_results])
        in_range_count = sum(1 for r in conformity_results if r[2])

        summary["conformity"]["mean_correlation"] = float(np.mean(correlations))
        summary["conformity"]["min_correlation"] = float(np.min(correlations))
        summary["conformity"]["max_correlation"] = float(np.max(correlations))
        summary["conformity"]["std_correlation"] = (
            float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0
        )
        summary["conformity"]["n_all_in_range"] = in_range_count
        summary["conformity"]["pass_rate"] = in_range_count / len(conformity_results)

    if consistency_results:
        icc_values = np.array([r.icc for r in consistency_results])
        drift_values = np.array([r.mean_absolute_drift for r in consistency_results])

        summary["consistency"] = {
            "n_tests": len(consistency_results),
            "mean_icc": float(np.mean(icc_values)),
            "min_icc": float(np.min(icc_values)),
            "max_icc": float(np.max(icc_values)),
            "std_icc": (
                float(np.std(icc_values, ddof=1)) if len(icc_values) > 1 else 0.0
            ),
            "mean_drift": float(np.mean(drift_values)),
            "interpretations": [r.icc_interpretation for r in consistency_results],
        }

    return summary
