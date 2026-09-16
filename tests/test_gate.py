from bsa.gate import check_gate

GATE = {
    "min_test": {"accuracy": 0.75, "f1_macro": 0.75},
    "compare_metric": "f1_macro",
    "max_regression": 0.01,
}


def candidate(accuracy=0.82, f1=0.82, fp="abc"):
    return {"test": {"accuracy": accuracy, "f1_macro": f1}, "dataset": {"fingerprints": {"test": fp}}}


def champion(f1=0.82, fp="abc"):
    return {"run_id": "r0", "version": "1", "test_fingerprint": fp,
            "metrics": {"accuracy": 0.82, "f1_macro": f1}}


def failed(result):
    return [check.name for check in result.checks if not check.passed]


def test_passes_without_champion():
    assert check_gate(candidate(), GATE, None).passed


def test_fails_below_floor():
    result = check_gate(candidate(accuracy=0.70), GATE, None)
    assert failed(result) == ["min_accuracy"]


def test_small_regression_within_tolerance_passes():
    assert check_gate(candidate(f1=0.815), GATE, champion(f1=0.82)).passed


def test_regression_beyond_tolerance_fails():
    assert failed(check_gate(candidate(f1=0.80), GATE, champion(f1=0.82))) == ["no_regression_f1_macro"]


def test_different_test_set_is_not_comparable():
    assert failed(check_gate(candidate(), GATE, champion(fp="other"))) == ["comparable_test_set"]
