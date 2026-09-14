# Field Traffic → Verified Fixture Pipeline

## Purpose

Production traffic reveals the real task distribution. It does **not** provide
ground truth and therefore may not directly update policy, routing rules,
mechanism weights, acceptance thresholds, or model weights.

The only automatic production loop is:

```
field traffic
  → privacy-safe event log
  → clustering / coverage-gap discovery
  → fixture proposal
  → human or trusted curator establishes ground truth
  → deterministic scorer created
  → next-cycle DISCOVERY / VALIDATION assignment
  → Test 1.2 discovery
  → Test 2 proof
  → knockout / compiler
  → human acceptance gate
  → new frozen policy
```

Continuous routing may use the **currently frozen proven policy**. Continuous
learning from unverified field outcomes is prohibited.

## Field-event record

Every field event entering this path must satisfy this shape:

```json
{
  "schema_version": 1,
  "record_type": "UNVERIFIED_FIELD_EVENT",
  "event_id": "stable-opaque-id",
  "timestamp_utc": "ISO-8601",
  "request_text_redacted": "privacy-safe task text or null",
  "request_sha256": "sha256-of-original-or-redacted-request",
  "predicted_family": "one of the declared capability families or UNKNOWN",
  "family_confidence": 0.0,
  "frozen_policy_id": "policy/version actually used",
  "mechanism_ids_applied": [],
  "response_text_redacted": "privacy-safe response or null",
  "runtime_identity_sha256": "exact deployed runtime identity",
  "field_signals": {
    "user_retry_observed": false,
    "user_correction_observed": false,
    "tool_error_observed": false,
    "abstention_observed": false
  },
  "ground_truth_status": "UNVERIFIED",
  "capability_evidence_eligible": false,
  "policy_update_eligible": false,
  "training_data_eligible": false
}
```

Field signals are **triage signals only**. A retry, correction, abandonment,
latency spike, confidence score, or user reaction is not a capability label.

## Clustering and fixture proposal

Field events may be clustered to identify:

- task families missing from the current 40-family taxonomy;
- families whose real-world frequency differs from the synthetic suite;
- repeated task structures that the frozen policy routes poorly;
- recurring tool/schema/state/context structures;
- novel failure phenotypes and ambiguity patterns.

Clustering output may change **what fixtures should be built next**. It may not
change the current policy.

A fixture proposal must record:

```json
{
  "record_type": "FIELD_DERIVED_FIXTURE_PROPOSAL",
  "source_event_ids": [],
  "proposed_family": "family-or-NEW_FAMILY_CANDIDATE",
  "task_text": "sanitized standalone task",
  "ground_truth_status": "NEEDS_CURATOR",
  "scorer_status": "NEEDS_DETERMINISTIC_SCORER",
  "current_cycle_eligible": false,
  "policy_update_eligible": false
}
```

## Promotion to a real fixture

A proposal becomes a test fixture only after all of these are true:

1. The task is sanitized and self-contained.
2. A trusted curator establishes the correct answer or valid answer set.
3. A deterministic scorer is defined whenever technically possible.
4. The fixture is assigned a difficulty level and capability family.
5. The fixture has no protected-partition leakage.
6. The fixture enters a **future** campaign cycle, never the current cycle that
   generated or observed it.
7. The source field outcome remains unverified evidence and is not retroactively
   converted into a capability label.

Promoted fixture shape:

```json
{
  "record_type": "CURATED_FIELD_DERIVED_FIXTURE",
  "fixture_id": "new-immutable-id",
  "family_id": "declared-family",
  "difficulty_level": 0,
  "prompt": "standalone sanitized task",
  "expected": "verified answer or answer contract",
  "scorer": "deterministic scorer",
  "ground_truth_status": "CURATED_VERIFIED",
  "source_provenance": {
    "source": "FIELD_TRAFFIC",
    "source_event_ids": []
  },
  "earliest_cycle": "NEXT_CYCLE",
  "current_cycle_eligible": false
}
```

## Hard prohibitions

- No automatic policy update from field success/failure guesses.
- No weight update from unverified field traffic.
- No conversion of confidence into correctness.
- No acceptance decision from field telemetry alone.
- No current-cycle fixture creation from traffic observed during that cycle.
- No protected-holdout reuse.
- No raw private/user data in fixtures unless explicitly authorized and
  sanitized for the intended use.

## Continuous vs slow loop

**Continuous:** privacy-safe data collection, clustering, coverage-gap discovery,
and deterministic routing through the current frozen policy.

**Slow gated loop:** fixture curation → discovery → proof → knockout/compiler →
acceptance → new frozen policy.

The schedule may be quarterly or another deliberate cadence. The scientific
boundary is the human/curator gate plus fresh proof, not the calendar interval.

## Primary product

The durable asset is the growing corpus of **validated hard fixtures with known
answers and deterministic scorers**. Policy improvements are a downstream
product of that corpus.
