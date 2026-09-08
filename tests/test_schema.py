from compute_cost.schema import FailureCode, StageStatus, unavailable


def test_status_and_failure_codes_are_stable_strings():
    assert StageStatus.PASS.value == "PASS"
    assert StageStatus.PARTIAL.value == "PARTIAL"
    assert FailureCode.RUNTIME_UNAVAILABLE.value == "RUNTIME_UNAVAILABLE"
    assert FailureCode.CAPTURE_GAP.value == "CAPTURE_GAP"


def test_unavailable_requires_a_reason_and_preserves_collector():
    value = unavailable("nvidia-smi not found", collector="nvidia-smi")
    assert value == {
        "availability": "unavailable",
        "reason": "nvidia-smi not found",
        "collector": "nvidia-smi",
    }
