from compute_cost.schema import FailureCode, MeasurementKind, ResultClass, StageStatus, unavailable


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


def test_characterization_schema_is_stable():
    assert ResultClass.ANSWER_CORRECT.value == "ANSWER_CORRECT"
    assert ResultClass.ANSWER_WRONG.value == "ANSWER_WRONG"
    assert ResultClass.THINK_TRUNCATED.value == "THINK_TRUNCATED"
    assert ResultClass.ANSWER_TRUNCATED.value == "ANSWER_TRUNCATED"
    assert ResultClass.NO_FINAL_ANSWER.value == "NO_FINAL_ANSWER"
    assert ResultClass.FORMAT_FAILURE.value == "FORMAT_FAILURE"
    assert ResultClass.TOOL_FAILURE.value == "TOOL_FAILURE"
    assert ResultClass.CONTEXT_FAILURE.value == "CONTEXT_FAILURE"
    assert ResultClass.SCORER_DEFECT.value == "SCORER_DEFECT"
    assert ResultClass.RUNTIME_FAILURE.value == "RUNTIME_FAILURE"
    assert [kind.value for kind in MeasurementKind] == [
        "MEASURED",
        "DERIVED",
        "ESTIMATED",
        "UNAVAILABLE",
    ]
