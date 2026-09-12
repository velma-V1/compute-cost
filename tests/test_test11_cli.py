from compute_cost.cli import build_parser


def test_cli_exposes_gpt20b_test11_dry_run():
    parser = build_parser()
    args = parser.parse_args([
        "gpt20b-test1.1",
        "--model",
        "gpt-oss:20b",
        "--dry-run",
    ])

    assert args.command == "gpt20b-test1.1"
    assert args.model == "gpt-oss:20b"
    assert args.test1_run is None
    assert args.dry_run is True


def test_cli_accepts_test1_source_run():
    parser = build_parser()
    args = parser.parse_args([
        "gpt20b-test1.1",
        "--test1-run",
        "test1-20260911-112554-e0d1049f",
    ])

    assert args.command == "gpt20b-test1.1"
    assert args.test1_run == "test1-20260911-112554-e0d1049f"
    assert args.dry_run is False
