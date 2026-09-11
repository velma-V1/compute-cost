from compute_cost.cli import build_parser


def test_cli_exposes_gpt20b_test1_and_zero_call_dry_run():
    parser = build_parser()
    args = parser.parse_args([
        "gpt20b-test1",
        "--model",
        "gpt-oss:20b",
        "--baseline-run",
        "prior-run",
        "--dry-run",
    ])

    assert args.command == "gpt20b-test1"
    assert args.model == "gpt-oss:20b"
    assert args.baseline_run == "prior-run"
    assert args.dry_run is True
