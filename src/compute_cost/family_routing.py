"""Zero-model-call request -> capability-family routing.

The classifier is intentionally deterministic and dependency-free.  It is fit
from one prototype fixture per family, then evaluated on the remaining labeled
fixture corpus.  That keeps the preflight confusion pass outside Test 1.2 model
calls while ensuring the exact same classifier can be used by runtime routing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EPS = 1e-12


def _family(case: dict[str, Any]) -> str:
    meta = case.get("capability_map") or {}
    return str(case.get("family_id") or meta.get("family_id") or case.get("category") or "")


def _difficulty(case: dict[str, Any]) -> int:
    value = case.get("difficulty_level", 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _words(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower().replace("_", " "))


def _features(text: str) -> Counter[str]:
    words = _words(text)
    out: Counter[str] = Counter()
    for word in words:
        out["w:" + word] += 1
    for left, right in zip(words, words[1:]):
        out[f"b:{left} {right}"] += 1
    compact = "^" + " ".join(words) + "$"
    for n in (3, 4):
        for index in range(max(0, len(compact) - n + 1)):
            gram = compact[index:index+n]
            if gram.strip():
                out[f"c{n}:{gram}"] += 0.15
    return out


def _idf(documents: list[Counter[str]]) -> dict[str, float]:
    doc_count = max(1, len(documents))
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update(document.keys())
    return {
        feature: math.log((1.0 + doc_count) / (1.0 + seen)) + 1.0
        for feature, seen in frequency.items()
    }


def _vector(features: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weighted: dict[str, float] = {}
    for feature, count in features.items():
        if feature not in idf:
            continue
        raw_count = float(count)
        tf = raw_count if raw_count < 1.0 else (1.0 + math.log(raw_count))
        weighted[feature] = tf * idf[feature]
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm <= _EPS:
        return {}
    return {feature: value / norm for feature, value in weighted.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


def build_family_classifier(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one deterministic prototype per family from the easiest fixture.

    The prototype fixture is excluded from confusion-pass evaluation.
    """
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        family = _family(case)
        if family:
            by_family[family].append(case)
    if not by_family:
        raise ValueError("family classifier requires labeled fixtures")

    prototypes: dict[str, dict[str, Any]] = {}
    raw_documents: list[Counter[str]] = []
    ordered_families = sorted(by_family)
    for family in ordered_families:
        ordered = sorted(
            by_family[family],
            key=lambda row: (_difficulty(row), str(row.get("id") or "")),
        )
        case = ordered[0]
        prompt = str(case.get("prompt") or "")
        prototype_text = f"{family.replace('_', ' ')}\n{prompt}"
        feats = _features(prototype_text)
        raw_documents.append(feats)
        prototypes[family] = {
            "fixture_id": str(case.get("id") or ""),
            "difficulty_level": _difficulty(case),
            "prompt": prompt,
            "_features": feats,
        }

    idf = _idf(raw_documents)
    serializable: dict[str, Any] = {}
    for family in ordered_families:
        payload = prototypes[family]
        vector = _vector(payload["_features"], idf)
        serializable[family] = {
            "fixture_id": payload["fixture_id"],
            "difficulty_level": payload["difficulty_level"],
            "prompt": payload["prompt"],
            "vector": vector,
        }
    return {
        "schema_version": 1,
        "classifier_type": "DETERMINISTIC_TFIDF_PROTOTYPE",
        "model_calls_required": 0,
        "family_count": len(serializable),
        "idf": idf,
        "prototypes": serializable,
        "confidence_semantics": "COSINE_MARGIN_NOT_CALIBRATED_PROBABILITY",
    }


def classify_family(text: str, classifier: dict[str, Any]) -> dict[str, Any]:
    idf = {
        str(key): float(value)
        for key, value in (classifier.get("idf") or {}).items()
    }
    query = _vector(_features(text), idf)
    scored: list[tuple[float, str]] = []
    for family, payload in (classifier.get("prototypes") or {}).items():
        vector = {
            str(key): float(value)
            for key, value in (payload.get("vector") or {}).items()
        }
        scored.append((_dot(query, vector), str(family)))
    if not scored:
        raise ValueError("classifier has no family prototypes")
    scored.sort(key=lambda item: (-item[0], item[1]))
    top_score, top_family = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = max(0.0, top_score - second_score)
    return {
        "family_id": top_family,
        "score": top_score,
        "runner_up_family_id": scored[1][1] if len(scored) > 1 else None,
        "runner_up_score": second_score,
        "classification_margin": margin,
        "confidence_semantics": classifier.get(
            "confidence_semantics",
            "COSINE_MARGIN_NOT_CALIBRATED_PROBABILITY",
        ),
    }


def evaluate_family_classifier(
    cases: list[dict[str, Any]],
    classifier: dict[str, Any],
) -> dict[str, Any]:
    prototype_ids = {
        str(payload.get("fixture_id") or "")
        for payload in (classifier.get("prototypes") or {}).values()
    }
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_family_total: Counter[str] = Counter()
    per_family_correct: Counter[str] = Counter()
    margins: dict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []

    evaluated = 0
    correct = 0
    for case in cases:
        fixture_id = str(case.get("id") or "")
        actual = _family(case)
        if not actual or fixture_id in prototype_ids:
            continue
        result = classify_family(str(case.get("prompt") or ""), classifier)
        predicted = str(result["family_id"])
        evaluated += 1
        per_family_total[actual] += 1
        confusion[actual][predicted] += 1
        margins[actual].append(float(result["classification_margin"]))
        if predicted == actual:
            correct += 1
            per_family_correct[actual] += 1
        else:
            errors.append({
                "fixture_id": fixture_id,
                "actual_family": actual,
                "predicted_family": predicted,
                "runner_up_family": result.get("runner_up_family_id"),
                "classification_margin": result["classification_margin"],
            })

    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in errors:
        pair = tuple(sorted((row["actual_family"], row["predicted_family"])))
        pair_counts[pair] += 1

    families: dict[str, Any] = {}
    for family in sorted(per_family_total):
        total = int(per_family_total[family])
        hit = int(per_family_correct[family])
        family_confusion = {
            predicted: int(count)
            for predicted, count in sorted(confusion[family].items())
        }
        family_margins = margins.get(family) or []
        families[family] = {
            "evaluated": total,
            "correct": hit,
            "accuracy": (hit / total) if total else None,
            "mean_classification_margin": (
                sum(family_margins) / len(family_margins)
                if family_margins else None
            ),
            "predictions": family_confusion,
        }

    return {
        "schema_version": 1,
        "analysis_type": "ZERO_MODEL_CALL_RUNTIME_FAMILY_CLASSIFIER_CONFUSION",
        "model_calls_added": 0,
        "prototype_fixture_count": len(prototype_ids),
        "evaluated_fixture_count": evaluated,
        "correct_fixture_count": correct,
        "top1_accuracy": (correct / evaluated) if evaluated else None,
        "evaluation_scope": "EXISTING_SYNTHETIC_FIXTURE_CORPUS_ONLY",
        "generalization_claim": False,
        "prototype_holdout_policy": "EASIEST_FIXTURE_PER_FAMILY_EXCLUDED_FROM_EVALUATION",
        "confidence_semantics": classifier.get("confidence_semantics"),
        "families": families,
        "confusable_family_pairs": [
            {
                "family_a": pair[0],
                "family_b": pair[1],
                "error_count": int(count),
                "evidence_status": "HYPOTHESIS_ONLY",
                "runtime_harm_adjacency_eligible": False,
            }
            for pair, count in pair_counts.most_common()
        ],
        "errors": errors,
    }
