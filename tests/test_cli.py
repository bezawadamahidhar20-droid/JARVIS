"""CLI tests — argument parsing and command dispatch."""


import jarvis_cli
from jarvis_cli import _build_parser, main


def test_parser_defaults():
    args = _build_parser().parse_args([])
    assert args.text is False
    assert args.debug is False
    assert args.doctor is False
    assert args.benchmark is False
    assert args.benchmark_models is False
    assert args.hardware is False
    assert args.gui is False
    assert args.startup is None


def test_parser_flags():
    args = _build_parser().parse_args(
        ["--text", "--debug", "--benchmark", "--benchmark-models",
         "--hardware", "--doctor"]
    )
    assert args.text is True
    assert args.debug is True
    assert args.benchmark is True
    assert args.benchmark_models is True
    assert args.hardware is True
    assert args.doctor is True


def test_parser_startup_choices():
    args = _build_parser().parse_args(["--startup", "enable"])
    assert args.startup == "enable"
    args = _build_parser().parse_args(["--startup", "disable"])
    assert args.startup == "disable"


def test_version_prints_and_exits_zero(capsys):
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "JARVIS version" in out


def test_doctor_dispatches(monkeypatch, capsys):
    calls = {}

    def fake_doctor(verbose=False):
        calls["verbose"] = verbose
        print("FAKE DOCTOR OUTPUT")
        return 0

    monkeypatch.setattr("jarvis_cli.doctor.run_doctor", fake_doctor)
    assert main(["--doctor"]) == 0
    assert calls["verbose"] is False
    assert "FAKE DOCTOR OUTPUT" in capsys.readouterr().out


def test_benchmark_models_dispatches(monkeypatch):
    calls = {}

    def fake_benchmark(verbose=False):
        calls["verbose"] = verbose
        return 0

    monkeypatch.setattr(
        "jarvis_cli.benchmark.run_benchmark", fake_benchmark
    )
    assert main(["--benchmark-models", "--debug"]) == 0
    assert calls["verbose"] is True


def test_hardware_dispatches(monkeypatch):
    calls = {}

    def fake_hardware():
        calls["ran"] = True
        return 0

    monkeypatch.setattr("jarvis_cli.hardware.run_hardware", fake_hardware)
    assert main(["--hardware"]) == 0
    assert calls["ran"] is True


def test_startup_dispatches(monkeypatch):
    calls = {}

    def fake_startup(action):
        calls["action"] = action
        return 0

    monkeypatch.setattr("jarvis_cli.startup.handle_startup", fake_startup)
    assert main(["--startup", "disable"]) == 0
    assert calls["action"] == "disable"


def test_run_assistant_dispatches(monkeypatch):
    calls = {}

    def fake_run(text_mode=False, debug=False, benchmark=False):
        calls.update(text_mode=text_mode, debug=debug, benchmark=benchmark)
        return 0

    monkeypatch.setattr("main.run_assistant", fake_run)
    assert main(["--text", "--debug", "--benchmark"]) == 0
    assert calls == {"text_mode": True, "debug": True, "benchmark": True}


def test_gui_dispatches(monkeypatch):
    monkeypatch.setattr("jarvis_cli._run_gui", lambda: 0)
    assert main(["--gui"]) == 0


def test_run_gui_reports_missing_deps(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jarvis_ui.ui_main":
            raise ImportError("no PySide6")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = jarvis_cli._run_gui()
    assert result == 1
    assert "GUI dependencies missing" in capsys.readouterr().out
