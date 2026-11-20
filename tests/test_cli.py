"""Smoke tests for the command-line interface.

argparse applies ``%``-formatting to help strings, so a literal percent sign
in a help text must be escaped. These tests render the help for every
sub-command to catch that class of bug before it reaches a user.
"""

import pytest

from pneumonia_stats.cli import build_parser, main


def test_top_level_help_renders(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "pneumonia-stats" in out
    assert "95% intervals" in out


@pytest.mark.parametrize(
    "command",
    ["demo", "report", "compare", "mcnemar", "delong", "bootstrap", "wilson", "check"],
)
def test_subcommand_help_renders(command, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0
    assert command in capsys.readouterr().out


def test_unknown_command_fails():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["nope"])
    assert excinfo.value.code != 0


def test_defaults_are_wired():
    args = build_parser().parse_args(
        ["bootstrap", "--a", "a.csv", "--b", "b.csv"]
    )
    assert args.alpha == 0.05
    assert args.n_resamples == 10_000
    assert args.seed == 42
    assert args.metric == "accuracy"
