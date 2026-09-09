from datetime import datetime, timezone

from compute_cost.progress import ProgressDisplay, format_duration


def test_format_duration_uses_wall_clock_hh_mm_ss():
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"


def test_progress_reports_done_left_percent_elapsed_eta_and_finish_time():
    now = [100.0]
    wall = [datetime(2026, 9, 7, 21, 0, 0, tzinfo=timezone.utc)]
    progress = ProgressDisplay(
        total_tasks=10,
        monotonic=lambda: now[0],
        wall_clock=lambda: wall[0],
        width_getter=lambda: 140,
    )
    progress.start("preflight")
    now[0] = 160.0
    wall[0] = datetime(2026, 9, 7, 21, 1, 0, tzinfo=timezone.utc)
    progress.complete_task("base 1/10")

    line = progress.render()
    assert "10%" in line
    assert "1/10 done" in line
    assert "9 left" in line
    assert "elapsed 00:01:00" in line
    assert "ETA 00:09:00" in line
    assert "finish 21:10:00" in line
    assert "base 1/10" in line


def test_progress_reaches_exactly_100_percent_with_zero_left():
    now = [10.0]
    progress = ProgressDisplay(total_tasks=2, monotonic=lambda: now[0], width_getter=lambda: 100)
    progress.start("cold")
    now[0] = 20.0
    progress.complete_task("warmup")
    now[0] = 30.0
    progress.complete_task("complete")

    line = progress.render()
    assert "100%" in line
    assert "2/2 done" in line
    assert "0 left" in line
    assert "ETA 00:00:00" in line


def test_progress_line_never_exceeds_current_terminal_width():
    now = [0.0]
    progress = ProgressDisplay(total_tasks=20, monotonic=lambda: now[0], width_getter=lambda: 62)
    progress.start("long-context retrieval at 32768 tokens with a very long task label")
    now[0] = 30.0
    progress.complete_task("long-context retrieval at 32768 tokens with a very long task label")

    assert len(progress.render()) <= 62


def test_wider_terminal_gets_a_longer_progress_bar():
    now = [0.0]
    narrow = ProgressDisplay(total_tasks=10, monotonic=lambda: now[0], width_getter=lambda: 80)
    wide = ProgressDisplay(total_tasks=10, monotonic=lambda: now[0], width_getter=lambda: 160)
    narrow.start("base")
    wide.start("base")
    now[0] = 5.0
    narrow.complete_task("base")
    wide.complete_task("base")

    narrow_line = narrow.render()
    wide_line = wide.render()
    assert len(wide_line) > len(narrow_line)
    assert wide_line.count("█") > narrow_line.count("█")
