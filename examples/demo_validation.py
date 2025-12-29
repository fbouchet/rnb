"""
RnB Personality Validation Framework Demonstration

This example demonstrates the validation framework for testing whether
RnB agents exhibit personality traits consistent with their design.

Demonstrates:
1. Loading instruments (TIPI, BFI-2-S) and archetypes
2. Running conformity tests against literature-grounded archetypes
3. Configurable tolerance for trait deviation
4. Statistical metrics with pingouin (correlations, ICC)
5. Integration with RnB LLMClient and AgentFactory

Requirements:
    - Redis running (for personality state)
    - Ollama with llama3.2:3b (or other LLM)
    - pingouin installed: pip install pingouin

Usage:
    poetry run python examples/demo_validation.py

    # With different model
    poetry run python examples/demo_validation.py --model llama3.1:8b

    # With commercial API
    poetry run python examples/demo_validation.py --provider anthropic --model claude-3-5-sonnet-20241022

References:
    TIPI: Gosling, S. D., Rentfrow, P. J., & Swann, W. B., Jr. (2003).
    BFI-2: Soto, C. J., & John, O. P. (2017).
    RUO Archetypes: Robins et al. (1996); Asendorpf et al. (2001).
"""

import argparse

from rnb.console import blank, rule, say_model, say_system
from rnb.logging import configure_logging

# =============================================================================
# Demo Functions
# =============================================================================


def demo_load_resources():
    """Demonstrate loading instruments and archetypes."""
    rule("1. Loading Validation Resources")

    from rnb.validation import (
        get_archetypes_path,
        get_instruments_dir,
        load_archetypes,
        load_instrument,
    )

    # Show default paths
    say_system("Default resource paths:")
    say_model(f"  Instruments: {get_instruments_dir()}")
    say_model(f"  Archetypes:  {get_archetypes_path()}")
    blank(1)

    # Load TIPI instrument
    say_system("Loading TIPI (Ten-Item Personality Inventory)...")
    try:
        tipi = load_instrument(get_instruments_dir() / "tipi.yaml")
        say_model(f"  Name: {tipi.name}")
        say_model(f"  Items: {tipi.items_count}")
        say_model(f"  Scale: {tipi.scale.min}-{tipi.scale.max}")
        say_model(f"  Domains: {list(tipi.scoring.domains.keys())}")
    except FileNotFoundError:
        say_system(
            "  TIPI not found - please ensure instruments are in resources/instruments/"
        )
    blank(1)

    # Load BFI-2-S instrument
    say_system("Loading BFI-2-S (Big Five Inventory-2 Short)...")
    try:
        bfi2s = load_instrument(get_instruments_dir() / "bfi2s.yaml")
        say_model(f"  Name: {bfi2s.name}")
        say_model(f"  Items: {bfi2s.items_count}")
        say_model(f"  Scale: {bfi2s.scale.min}-{bfi2s.scale.max}")
        say_model(
            f"  Facets: {len(bfi2s.scoring.facets) if bfi2s.scoring.facets else 0}"
        )
    except FileNotFoundError:
        say_system(
            "  BFI-2-S not found - please ensure instruments are in resources/instruments/"
        )
    blank(1)

    # Load archetypes
    say_system("Loading RUO Archetypes...")
    try:
        archetypes = load_archetypes(get_archetypes_path())
        say_model(f"  Loaded {len(archetypes)} archetypes:")
        for name, arch in archetypes.items():
            say_model(f"    • {name}: {arch.description[:60]}...")
    except FileNotFoundError:
        say_system(
            "  Archetypes not found - please ensure archetypes.yaml is in resources/data/"
        )


def demo_archetype_details():
    """Demonstrate archetype structure and RUO literature grounding."""
    rule("2. RUO Archetype Details")

    from rnb.validation import get_archetypes_path, load_archetypes

    try:
        archetypes = load_archetypes(get_archetypes_path())
    except FileNotFoundError:
        say_system("Archetypes not found - skipping this demo")
        return

    # Show RUO archetypes (literature-grounded)
    ruo_names = ["resilient", "overcontrolled", "undercontrolled"]

    for name in ruo_names:
        if name not in archetypes:
            continue

        arch = archetypes[name]
        blank(1)
        say_system(f"=== {name.upper()} ===")
        say_model(f"Description: {arch.description[:100]}...")
        blank(1)

        say_system("Trait Profile (RnB scale [-1, 1]):")
        for trait, value in sorted(arch.traits.items()):
            bar = "█" * int(abs(value) * 10)
            sign = "+" if value >= 0 else "-"
            say_model(f"  {trait:20} {sign}{abs(value):.1f} {bar}")

        blank(1)
        say_system("Behavioral Markers:")
        for marker in arch.behavioral_markers[:3]:
            say_model(f"  • {marker}")

        blank(1)
        say_system("Literature References:")
        for ref in arch.references[:2]:
            say_model(f"  → {ref}")


def demo_tolerance_system():
    """Demonstrate the configurable tolerance system."""
    rule("3. Configurable Tolerance System")

    from rnb.validation import (
        check_range_conformity,
        get_archetypes_path,
        load_archetypes,
    )

    say_system(
        "Tolerance controls how much measured traits can deviate from designed values."
    )
    say_system(
        "On RnB scale [-1, 1], tolerance of 0.25 means ±0.25 around designed value."
    )
    blank(1)

    try:
        archetypes = load_archetypes(get_archetypes_path())
        resilient = archetypes["resilient"]
    except (FileNotFoundError, KeyError):
        say_system("Archetypes not found - using example values")
        designed = {"openness": 0.5, "conscientiousness": 0.6, "neuroticism": -0.7}
    else:
        designed = resilient.traits

    # Simulated measured scores (close to resilient but not perfect)
    measured = {
        "openness": 0.4,  # Designed: 0.5, diff: -0.1
        "conscientiousness": 0.7,  # Designed: 0.6, diff: +0.1
        "extraversion": 0.3,  # Designed: 0.5, diff: -0.2
        "agreeableness": 0.6,  # Designed: 0.5, diff: +0.1
        "neuroticism": -0.5,  # Designed: -0.7, diff: +0.2
    }

    say_system("Simulated measured scores vs designed (resilient):")
    for trait in sorted(designed.keys()):
        d = designed[trait]
        m = measured.get(trait, 0)
        diff = m - d
        say_model(
            f"  {trait:20} designed={d:+.1f}  measured={m:+.1f}  diff={diff:+.2f}"
        )
    blank(1)

    # Test with different tolerances
    tolerances = [0.15, 0.25, 0.35]

    say_system("Conformity results at different tolerances:")
    for tol in tolerances:
        all_within, per_trait, ranges = check_range_conformity(
            measured, designed, tolerance=tol
        )
        passed = sum(per_trait.values())
        total = len(per_trait)
        status = "✓ PASS" if all_within else "✗ FAIL"
        say_model(
            f"  tolerance=±{tol}: {passed}/{total} traits within range → {status}"
        )


def demo_statistical_metrics():
    """Demonstrate pingouin-based statistical metrics."""
    rule("4. Statistical Metrics (Pingouin)")

    from rnb.validation import (
        calculate_icc,
        pearson_correlation,
        spearman_correlation,
    )

    say_system("The framework uses pingouin for validated statistical analysis.")
    say_system("Demonstrating with SIMULATED data (no LLM calls):")
    blank(1)

    # Correlation example
    say_system("Correlation with confidence intervals:")
    say_model("  (Simulated: designed resilient traits vs hypothetical measurements)")
    designed = [0.5, 0.6, 0.5, 0.5, -0.7]  # Resilient traits
    measured = [0.4, 0.7, 0.3, 0.6, -0.5]  # Simulated measurements

    # Pearson
    result = pearson_correlation(designed, measured, return_details=True)
    say_model(f"  Pearson r  = {result.r:.3f}")
    say_model(f"  95% CI     = [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    say_model(f"  p-value    = {result.p_value:.4f}")
    blank(1)

    # Spearman
    result_sp = spearman_correlation(designed, measured, return_details=True)
    say_model(f"  Spearman ρ = {result_sp.r:.3f}")
    say_model(f"  95% CI     = [{result_sp.ci_lower:.3f}, {result_sp.ci_upper:.3f}]")
    blank(1)

    # ICC example (simulated repeated measurements)
    say_system("ICC for consistency testing:")
    say_model("  (Simulated: 5 repeated assessments of same agent)")
    repeated_measurements = [
        {"O": 0.50, "C": 0.65, "E": 0.48, "A": 0.52, "N": -0.68},
        {"O": 0.48, "C": 0.62, "E": 0.45, "A": 0.55, "N": -0.65},
        {"O": 0.52, "C": 0.68, "E": 0.50, "A": 0.48, "N": -0.70},
        {"O": 0.49, "C": 0.64, "E": 0.47, "A": 0.53, "N": -0.67},
        {"O": 0.51, "C": 0.66, "E": 0.49, "A": 0.50, "N": -0.69},
    ]

    icc, interpretation, ci_low, ci_high = calculate_icc(repeated_measurements)
    say_model(f"  ICC(3)     = {icc:.3f}")
    say_model(f"  95% CI     = [{ci_low:.3f}, {ci_high:.3f}]")
    say_model(f"  Interp.    = {interpretation}")
    blank(1)

    say_system("ICC Interpretation (Koo & Li, 2016):")
    say_model("  < 0.50: poor")
    say_model("  0.50-0.75: moderate")
    say_model("  0.75-0.90: good")
    say_model("  > 0.90: excellent")


def demo_quick_validation(provider: str, model_name: str):
    """Demonstrate quick validation with actual LLM."""
    rule("5. Quick Validation (Live LLM)")

    say_system(f"Running validation with {provider}/{model_name}")
    say_system("This will query the LLM with personality assessment items.")
    blank(1)

    try:
        from rnb.validation import run_quick_validation

        say_system("Testing 'resilient' archetype with TIPI...")
        blank(1)

        _results = run_quick_validation(
            archetype="resilient",
            instrument="tipi",
            provider=provider,
            model_name=model_name,
            tolerance=0.25,
            verbose=True,
        )

        say_system("Validation complete!")

    except ImportError as e:
        say_system(f"Import error: {e}")
        say_system("Make sure all RnB modules are available.")
    except ConnectionError as e:
        say_system(f"Connection error: {e}")
        say_system("Ensure the LLM is available and responding.")
    except ValueError as e:
        # This catches invalid ratings, parsing errors, etc.
        say_system(f"Validation error: {e}")
        say_system("The LLM returned an invalid response (e.g., rating outside scale).")
        say_system("This may indicate the model needs stronger prompting constraints.")
    except Exception as e:
        say_system(f"Unexpected error: {type(e).__name__}: {e}")
        say_system("Check the error details above for more information.")


def demo_full_runner(provider: str, model_name: str, verbose: bool = False):
    """Demonstrate full ValidationRunner with multiple archetypes."""
    rule("6. Full ValidationRunner Demo")

    say_system(f"Creating ValidationRunner with {provider}/{model_name}")
    blank(1)

    try:
        from rnb.validation import create_validation_runner

        runner = create_validation_runner(
            provider=provider,
            model_name=model_name,
            tolerance=0.25,
        )

        say_system("Runner created successfully!")
        say_model(f"  Conformity threshold: {runner.conformity_threshold}")
        say_model(f"  Consistency threshold: {runner.consistency_threshold}")
        say_model(f"  Default tolerance: {runner.tolerance}")
        say_model(f"  Archetypes loaded: {list(runner.archetypes.keys())}")
        blank(1)

        # Test multiple archetypes
        archetypes_to_test = ["resilient", "overcontrolled", "undercontrolled"]
        results_summary = []

        for arch_name in archetypes_to_test:
            if arch_name not in runner.archetypes:
                continue

            say_system(f"Testing {arch_name}...")

            try:
                result = runner.test_conformity(
                    archetype_name=arch_name,
                    instrument="tipi",
                    tolerance=0.25,
                    verbose=verbose,
                )

                # Show detailed result table
                print_conformity_result(result, show_details=True)

                results_summary.append(
                    {
                        "archetype": arch_name,
                        "correlation": result.conformity.correlation,
                        "all_within": result.conformity.all_within_tolerance,
                        "passed": result.passed,
                    }
                )
                blank(1)

            except ValueError as e:
                say_system(f"  Error testing {arch_name}: {e}")
                say_system(
                    "    (LLM returned invalid response - may need stronger prompting)"
                )
                blank(1)
            except Exception as e:
                say_system(f"  Error testing {arch_name}: {type(e).__name__}: {e}")
                blank(1)

        # Summary table
        if results_summary:
            say_system("=== SUMMARY ===")
            say_model(
                f"{'Archetype':<20} {'Correlation':>12} {'Within Tol':>12} {'Result':>10}"
            )
            say_model("-" * 56)
            for r in results_summary:
                status = "PASS" if r["passed"] else "FAIL"
                within = "Yes" if r["all_within"] else "No"
                say_model(
                    f"{r['archetype']:<20} {r['correlation']:>12.3f} {within:>12} {status:>10}"
                )
        else:
            say_system("=== NO SUCCESSFUL TESTS ===")
            say_system("All tests failed. This typically happens when:")
            say_model(
                "  • The LLM returns ratings outside the valid scale (e.g., 8 or 9 on a 1-7 scale)"
            )
            say_model("  • The LLM doesn't follow JSON format instructions")
            say_model("  • Smaller models may need more constrained prompting")
            say_system(
                "Try with a larger model (e.g., llama3.1:8b) or a commercial API."
            )

    except ImportError as e:
        say_system(f"Import error: {e}")
        say_system("The integration module requires full RnB package context.")
    except Exception as e:
        say_system(f"Runner failed: {e}")
        import traceback

        traceback.print_exc()


def demo_standalone_assessment():
    """Demonstrate standalone assessment without full integration."""
    rule("7. Standalone Assessment Demo")

    say_system("This demo shows assessment scoring without LLM integration.")
    say_system("Useful for testing the scoring logic with mock responses.")
    blank(1)

    from rnb.validation import (
        PersonalityAssessor,
        get_instruments_dir,
        load_instrument,
    )

    try:
        tipi = load_instrument(get_instruments_dir() / "tipi.yaml")
    except FileNotFoundError:
        say_system("TIPI not found - skipping standalone demo")
        return

    assessor = PersonalityAssessor(tipi)

    # Simulate responses for a "resilient" personality
    # High on all positive traits, low on neuroticism items
    say_system("Simulating TIPI responses for resilient personality:")

    mock_responses = {
        1: 6,  # Extraverted, enthusiastic (E+)
        2: 2,  # Critical, quarrelsome (A-) → low = agreeable
        3: 6,  # Dependable, self-disciplined (C+)
        4: 2,  # Anxious, easily upset (N+) → low = stable
        5: 6,  # Open to new experiences (O+)
        6: 2,  # Reserved, quiet (E-) → low = extraverted
        7: 6,  # Sympathetic, warm (A+)
        8: 2,  # Disorganized, careless (C-) → low = conscientious
        9: 6,  # Calm, emotionally stable (N-)
        10: 5,  # Conventional, uncreative (O-) → moderate
    }

    say_model("  Mock responses (1-7 scale):")
    for item_id, response in sorted(mock_responses.items()):
        item = next(i for i in tipi.items if i.id == item_id)
        rev = " (R)" if item.reversed else ""
        say_model(f"    {item_id}. {item.text[:40]}...{rev} → {response}")
    blank(1)

    # Score the responses
    result = assessor._calculate_scores(mock_responses)

    say_system("Domain scores (TIPI scale 1-7):")
    for domain, score in sorted(result.domain_scores.items()):
        say_model(f"  {domain:20} {score:.2f}")
    blank(1)

    say_system("RnB scores (scale -1 to +1):")
    if result.rnb_scores:
        for trait, score in sorted(result.rnb_scores.items()):
            bar = "█" * int(abs(score) * 10)
            sign = "+" if score >= 0 else "-"
            say_model(f"  {trait:20} {sign}{abs(score):.2f} {bar}")


def demo_n_run_validation(
    provider: str,
    model_name: str,
    n_runs: int = 10,
    verbose: bool = False,
    archetype: str = "resilient",
    show_prompt: bool = False,
):
    """Demonstrate N-run validation for statistical reliability."""
    rule("8. N-Run Statistical Validation")

    say_system(
        f"Running {n_runs} validation attempts to assess statistical reliability."
    )
    say_system("This addresses natural variability in LLM responses.")
    say_system(f"Using {provider}/{model_name}")
    blank(1)

    try:
        from rnb.validation import create_validation_runner

        runner = create_validation_runner(
            provider=provider,
            model_name=model_name,
            tolerance=0.25,
        )

        # Validate archetype exists
        if archetype not in runner.archetypes:
            say_system(f"Unknown archetype: {archetype}")
            say_system(f"Available archetypes: {list(runner.archetypes.keys())}")
            return

        say_system(f"Testing '{archetype}' archetype with {n_runs} runs...")
        say_system("(This may take a few minutes)")
        blank(1)

        # Show the system prompt if requested
        if show_prompt:
            say_system("=== RnB SYSTEM PROMPT ===")
            # Create a temporary agent to get the prompt
            temp_agent = runner.agent_factory.from_archetype(archetype)
            system_prompt = temp_agent.get_system_prompt()
            say_model(f"  {system_prompt}")
            say_system("=" * 60)
            blank(1)

        def progress_callback(current: int, total: int):
            say_model(f"  Run {current}/{total}...")

        result = runner.test_conformity_n_runs(
            archetype_name=archetype,
            n_runs=n_runs,
            instrument="tipi",
            tolerance=0.25,
            verbose=verbose,
            progress_callback=progress_callback,
        )

        blank(1)
        say_system("=== N-RUN RESULTS ===")
        say_model(f"  Archetype:          {result.archetype_name}")
        say_model(f"  Total runs:         {result.n_runs}")
        say_model(f"  Passed:             {sum(1 for r in result.results if r.passed)}")
        say_model(
            f"  Failed:             {sum(1 for r in result.results if not r.passed)}"
        )
        say_model(f"  Pass rate:          {result.pass_rate:.1%}")
        blank(1)
        say_system("Correlation Statistics:")
        say_model(f"  Mean correlation:   {result.mean_correlation:.3f}")
        say_model(f"  Std deviation:      {result.std_correlation:.3f}")
        say_model(f"  Min correlation:    {min(result.correlations):.3f}")
        say_model(f"  Max correlation:    {max(result.correlations):.3f}")
        blank(1)

        # Show distribution of correlations with trait detail
        say_system("Per-run results:")
        say_model("  (Pass requires: r ≥ 0.7 AND all 5 traits within ±tolerance)")
        blank(1)
        for i, corr in enumerate(result.correlations, 1):
            res = result.results[i - 1]
            passed = res.passed
            status = "✓" if passed else "✗"

            # Count traits within tolerance
            n_within = sum(res.conformity.within_tolerance.values())
            n_total = len(res.conformity.within_tolerance)
            traits_status = f"{n_within}/{n_total} traits"

            # Show which traits failed if any
            if n_within < n_total:
                failed_traits = [
                    t for t, w in res.conformity.within_tolerance.items() if not w
                ]
                failed_str = ",".join(t[:3].upper() for t in failed_traits)  # O,C,E,A,N
                say_model(
                    f"  Run {i:2}: r={corr:+.3f}  {traits_status}  {status}  (failed: {failed_str})"
                )
            else:
                say_model(f"  Run {i:2}: r={corr:+.3f}  {traits_status}  {status}")

        blank(1)

        # Show WHY tests are failing (aggregate trait deviations)
        if result.pass_rate < 1.0 and result.results:
            say_system("Trait Analysis (across all runs):")
            say_model(
                "  Pass condition: correlation ≥ 0.7 AND all traits within ±tolerance"
            )
            blank(1)

            # Aggregate which traits fail most often
            trait_failures = {}
            trait_deviations = {}
            for res in result.results:
                for trait, within in res.conformity.within_tolerance.items():
                    if trait not in trait_failures:
                        trait_failures[trait] = 0
                        trait_deviations[trait] = []
                    if not within:
                        trait_failures[trait] += 1
                    dev = res.conformity.deviations.get(trait, 0)
                    trait_deviations[trait].append(dev)

            # Show per-trait analysis
            first_result = result.results[0]
            say_system("  Per-trait conformity:")
            for trait in sorted(trait_failures.keys()):
                designed = first_result.conformity.designed_traits.get(trait, 0)
                fail_count = trait_failures[trait]
                fail_pct = fail_count / len(result.results) * 100
                avg_dev = sum(trait_deviations[trait]) / len(trait_deviations[trait])
                status = "✓" if fail_count == 0 else "✗"
                say_model(
                    f"    {trait:18} designed={designed:+.2f}  avg_dev={avg_dev:+.2f}  fails={fail_count:2}/{len(result.results)} ({fail_pct:5.1f}%) {status}"
                )

        blank(1)
        say_system("Interpretation:")
        if result.pass_rate >= 0.8:
            say_model("  ✓ High reliability: >80% of runs passed")
        elif result.pass_rate >= 0.5:
            say_model("  ~ Moderate reliability: 50-80% of runs passed")
        else:
            say_model("  ✗ Low reliability: <50% of runs passed")
            if result.mean_correlation >= 0.7:
                say_model("  Note: High correlation but traits outside tolerance range")

        if result.std_correlation < 0.1:
            say_model("  ✓ Low variance: Std dev < 0.1")
        elif result.std_correlation < 0.2:
            say_model("  ~ Moderate variance: Std dev 0.1-0.2")
        else:
            say_model("  ✗ High variance: Std dev > 0.2")

    except ImportError as e:
        say_system(f"Import error: {e}")
        say_system("The integration module requires full RnB package context.")
    except Exception as e:
        say_system(f"N-run validation failed: {e}")
        import traceback

        traceback.print_exc()


def demo_baseline_comparison(
    provider: str,
    model_name: str,
    n_runs: int = 10,
    verbose: bool = False,
    archetype: str = "resilient",
):
    """Compare RnB-backed agents vs baseline (pure prompt) agents."""
    rule("9. RnB vs Baseline Comparison")

    say_system("Comparing personality conformity:")
    say_system("  • RnB: Personality state + influence operators")
    say_system("  • Baseline: Pure prompt description only")
    say_system(f"Using {provider}/{model_name}, archetype: {archetype}")
    blank(1)

    try:
        import statistics

        from rnb.validation import (
            # PersonalityAssessor,
            create_validation_runner,
            generate_pure_prompt_system,
            get_archetypes_path,
            load_archetypes,
        )
        from rnb.validation.integration import LLMClientAdapter

        # Load archetype
        archetypes = load_archetypes(get_archetypes_path())
        if archetype not in archetypes:
            say_system(f"Unknown archetype: {archetype}")
            say_system(f"Available: {list(archetypes.keys())}")
            return

        arch_config = archetypes[archetype]

        # Create runner for RnB tests
        runner = create_validation_runner(
            provider=provider,
            model_name=model_name,
            tolerance=0.25,
        )

        # Create LLM client for baseline tests
        llm_client = LLMClientAdapter.from_provider(provider, model_name)
        assessor = runner.get_assessor("tipi")

        # Show the two system prompts
        say_system("=== SYSTEM PROMPTS ===")
        blank(1)

        # Baseline prompt
        baseline_prompt = generate_pure_prompt_system(arch_config)
        say_system("BASELINE (pure prompt):")
        for line in baseline_prompt.split("\n"):
            say_model(f"  {line}")
        blank(1)

        # RnB prompt (from a temporary agent)
        temp_agent = runner.agent_factory.from_archetype(archetype)
        rnb_prompt = temp_agent.get_system_prompt()
        say_system("RnB (with influence operators):")
        for line in rnb_prompt.split("\n")[:15]:  # First 15 lines
            say_model(f"  {line}")
        if rnb_prompt.count("\n") > 15:
            say_model(f"  ... ({rnb_prompt.count(chr(10)) - 15} more lines)")

        say_system("=" * 60)
        blank(1)

        # =====================================================================
        # Run RnB tests
        # =====================================================================
        say_system(f"Running {n_runs} tests WITH RnB...")

        rnb_results = []
        rnb_correlations = []

        for i in range(n_runs):
            say_model(f"  RnB run {i+1}/{n_runs}...")
            try:
                result = runner.test_conformity(
                    archetype_name=archetype,
                    instrument="tipi",
                    tolerance=0.25,
                    verbose=verbose,
                )
                rnb_results.append(result)
                rnb_correlations.append(result.conformity.correlation)
            except Exception as e:
                say_model(f"    Failed: {e}")

        blank(1)

        # =====================================================================
        # Run Baseline tests
        # =====================================================================
        say_system(f"Running {n_runs} tests WITHOUT RnB (baseline)...")

        baseline_results = []
        baseline_correlations = []

        for i in range(n_runs):
            say_model(f"  Baseline run {i+1}/{n_runs}...")
            try:
                assessment = assessor.assess(
                    llm_client,
                    system_prompt=baseline_prompt,
                    verbose=verbose,
                )
                conformity = assessor.check_conformity(
                    assessment, arch_config, tolerance=0.25
                )
                passed = (
                    conformity.correlation >= 0.7 and conformity.all_within_tolerance
                )

                baseline_results.append(
                    {
                        "assessment": assessment,
                        "conformity": conformity,
                        "passed": passed,
                    }
                )
                baseline_correlations.append(conformity.correlation)
            except Exception as e:
                say_model(f"    Failed: {e}")

        blank(1)

        # =====================================================================
        # Compare results
        # =====================================================================
        say_system("=" * 70)
        say_system("                    COMPARISON RESULTS")
        say_system("=" * 70)
        blank(1)

        # Calculate statistics
        def calc_stats(correlations, results_list, is_rnb=True):
            if not correlations:
                return None

            if is_rnb:
                n_passed = sum(1 for r in results_list if r.passed)
            else:
                n_passed = sum(1 for r in results_list if r["passed"])

            return {
                "n": len(correlations),
                "pass_rate": n_passed / len(correlations) if correlations else 0,
                "mean_r": statistics.mean(correlations),
                "std_r": statistics.stdev(correlations) if len(correlations) > 1 else 0,
                "min_r": min(correlations),
                "max_r": max(correlations),
            }

        rnb_stats = calc_stats(rnb_correlations, rnb_results, is_rnb=True)
        baseline_stats = calc_stats(
            baseline_correlations, baseline_results, is_rnb=False
        )

        # Display comparison table
        say_system(f"{'Metric':<25} {'RnB':>15} {'Baseline':>15} {'Δ':>12}")
        say_system("-" * 70)

        if rnb_stats and baseline_stats:
            # Pass rate
            delta_pass = rnb_stats["pass_rate"] - baseline_stats["pass_rate"]
            delta_sign = "+" if delta_pass > 0 else ""
            say_model(
                f"{'Pass rate':<25} {rnb_stats['pass_rate']:>14.1%} {baseline_stats['pass_rate']:>14.1%} {delta_sign}{delta_pass:>11.1%}"
            )

            # Mean correlation
            delta_r = rnb_stats["mean_r"] - baseline_stats["mean_r"]
            delta_sign = "+" if delta_r > 0 else ""
            say_model(
                f"{'Mean correlation':<25} {rnb_stats['mean_r']:>15.3f} {baseline_stats['mean_r']:>15.3f} {delta_sign}{delta_r:>11.3f}"
            )

            # Std deviation
            delta_std = rnb_stats["std_r"] - baseline_stats["std_r"]
            delta_sign = "+" if delta_std > 0 else ""
            # better_std = "←" if rnb_stats["std_r"] < baseline_stats["std_r"] else "→"
            say_model(
                f"{'Std deviation':<25} {rnb_stats['std_r']:>15.3f} {baseline_stats['std_r']:>15.3f} {delta_sign}{delta_std:>11.3f}"
            )

            # Range
            say_model(
                f"{'Min correlation':<25} {rnb_stats['min_r']:>15.3f} {baseline_stats['min_r']:>15.3f}"
            )
            say_model(
                f"{'Max correlation':<25} {rnb_stats['max_r']:>15.3f} {baseline_stats['max_r']:>15.3f}"
            )

            blank(1)
            say_system("=" * 70)

            # Interpretation
            say_system("Interpretation:")

            if rnb_stats["pass_rate"] > baseline_stats["pass_rate"]:
                say_model(f"  ✓ RnB improves pass rate by {delta_pass:.1%}")
            elif rnb_stats["pass_rate"] < baseline_stats["pass_rate"]:
                say_model(f"  ✗ Baseline has higher pass rate by {-delta_pass:.1%}")
            else:
                say_model("  ~ Pass rates are equal")

            if rnb_stats["std_r"] < baseline_stats["std_r"]:
                say_model("  ✓ RnB is more consistent (lower variance)")
            elif rnb_stats["std_r"] > baseline_stats["std_r"]:
                say_model("  ✗ Baseline is more consistent (lower variance)")
            else:
                say_model("  ~ Variance is similar")

            if rnb_stats["mean_r"] > baseline_stats["mean_r"]:
                say_model("  ✓ RnB has higher mean correlation")
            elif rnb_stats["mean_r"] < baseline_stats["mean_r"]:
                say_model("  ✗ Baseline has higher mean correlation")
            else:
                say_model("  ~ Mean correlations are similar")
        else:
            say_system("Insufficient data for comparison")

    except ImportError as e:
        say_system(f"Import error: {e}")
        say_system("The integration module requires full RnB package context.")
    except Exception as e:
        say_system(f"Baseline comparison failed: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# Helper Functions
# =============================================================================


def print_conformity_result(result, show_details: bool = True):
    """Print a formatted conformity test result."""
    say_system(f"{'='*60}")
    say_system(f"Validation: {result.archetype_name} with {result.instrument.upper()}")
    say_system(f"{'='*60}")
    say_model(f"Correlation:        {result.conformity.correlation:.3f}")
    say_model(f"All within tol:     {result.conformity.all_within_tolerance}")
    say_model(f"Tolerance used:     ±{result.tolerance}")
    say_model(f"Passed:             {'✓' if result.passed else '✗'}")
    blank(1)

    if show_details:
        say_system("Designed vs Measured (RnB scale):")
        for trait in sorted(result.conformity.designed_traits.keys()):
            designed = result.conformity.designed_traits[trait]
            measured = result.conformity.measured_scores.get(trait, 0)
            within = result.conformity.within_tolerance.get(trait, False)
            status = "✓" if within else "✗"
            say_model(
                f"  {trait:20} designed={designed:+.2f}  measured={measured:+.2f}  {status}"
            )
    say_system(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run validation framework demonstrations."""
    parser = argparse.ArgumentParser(
        description="RnB Personality Validation Framework Demo"
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic"],
        default="local",
        help="LLM provider (default: local)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Model name (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip demos that require LLM",
    )
    parser.add_argument(
        "--only",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Run only specific demo (1-8)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show raw LLM responses",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of runs for N-run validation (default: 10)",
    )
    parser.add_argument(
        "--archetype",
        default="resilient",
        help="Archetype to test (default: resilient). Options: resilient, overcontrolled, undercontrolled, average, helpful_assistant",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Show the RnB system prompt sent to the LLM",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level="INFO", rich=True)

    # Header
    blank(1)
    say_system(
        "╔══════════════════════════════════════════════════════════════════════╗"
    )
    say_system(
        "║         RnB Personality Validation Framework Demonstration           ║"
    )
    say_system(
        "╚══════════════════════════════════════════════════════════════════════╝"
    )
    blank(1)

    demos = [
        (1, demo_load_resources, False),
        (2, demo_archetype_details, False),
        (3, demo_tolerance_system, False),
        (4, demo_statistical_metrics, False),
        (5, lambda: demo_quick_validation(args.provider, args.model), True),
        (6, lambda: demo_full_runner(args.provider, args.model, args.verbose), True),
        (7, demo_standalone_assessment, False),
        (
            8,
            lambda: demo_n_run_validation(
                args.provider,
                args.model,
                args.n_runs,
                args.verbose,
                args.archetype,
                args.show_prompt,
            ),
            True,
        ),
        (
            9,
            lambda: demo_baseline_comparison(
                args.provider,
                args.model,
                args.n_runs,
                args.verbose,
                args.archetype,
            ),
            True,
        ),
    ]

    for num, demo_func, requires_llm in demos:
        if args.only and args.only != num:
            continue
        if requires_llm and args.skip_llm:
            say_system(f"Skipping demo {num} (requires LLM)")
            continue

        try:
            demo_func()
        except Exception as e:
            say_system(f"Demo {num} failed: {e}")
            import traceback

            traceback.print_exc()

        blank(1)

    # Footer
    rule("Demo Complete")
    say_system("The validation framework enables empirical testing of RnB agent")
    say_system(
        "personalities using standardized psychological instruments (TIPI, BFI-2-S)"
    )
    say_system(
        "and literature-grounded archetypes (Resilient, Overcontrolled, Undercontrolled)."
    )
    blank(1)
    say_system("Key features:")
    say_model("  • Configurable tolerance (not hardcoded)")
    say_model("  • Pingouin-based statistics with confidence intervals")
    say_model("  • Integration with RnB LLMClient and AgentFactory")
    say_model("  • Support for conformity and consistency testing")
    say_model("  • N-run validation for statistical reliability")


if __name__ == "__main__":
    main()
