from compute_cost.cli import build_parser


def test_cli_exposes_gpt20b_test2_and_synthetic_dry_run():
    parser = build_parser()
    args = parser.parse_args([
        "gpt20b-test2",
        "--model",
        "gpt-oss:20b",
        "--dry-run",
    ])

    assert args.command == "gpt20b-test2"
    assert args.model == "gpt-oss:20b"
    assert args.test1_run is None
    assert args.dry_run is True


def test_cli_accepts_completed_test1_run_for_real_test2():
    parser = build_parser()
    args = parser.parse_args([
        "gpt20b-test2",
        "--model",
        "gpt-oss:20b",
        "--test1-run",
        "test1-20260911-example",
    ])

    assert args.command == "gpt20b-test2"
    assert args.test1_run == "test1-20260911-example"
    assert args.dry_run is False
