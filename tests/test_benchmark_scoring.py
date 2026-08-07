from benchmarks.vlm_bench import aggregate, percentile, score_output


def test_task_specific_quality_scores_are_deterministic():
    assert score_output("A red bird on a branch.", "keywords", ["red", "bird", "branch"]) == 1.0
    assert score_output("Hello, WORLD!", "exact", "hello world") == 1.0
    assert score_output("There are 12 objects.", "number", 12) == 1.0
    assert score_output("There are 11 objects.", "number", 12) == 0.0


def test_percentile_interpolates_small_samples():
    assert percentile([10.0, 20.0], 0.5) == 15.0


def test_aggregate_keeps_latency_ttft_throughput_and_quality_separate():
    samples = [
        {"latency_ms": 100.0, "ttft_ms": 40.0, "output_tokens_per_second": 20.0, "quality": 1.0},
        {"latency_ms": 200.0, "ttft_ms": 60.0, "output_tokens_per_second": 30.0, "quality": 0.5},
    ]
    result = aggregate(samples)
    assert result["latency_ms"]["p50"] == 150.0
    assert result["ttft_ms"]["p50"] == 50.0
    assert result["output_tokens_per_second_mean"] == 25.0
    assert result["quality_mean"] == 0.75

