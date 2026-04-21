import ast
import inspect
import os
import sys
import traceback
from typing import Callable, Iterable

from qiskit import transpile


def parse_cli_value(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw


def parse_kwargs_text(raw):
    kwargs = {}
    text = raw.strip()
    if not text:
        return kwargs
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value pair, got '{item}'")
        key, value = item.split("=", 1)
        kwargs[key.strip()] = parse_cli_value(value.strip())
    return kwargs


def format_signature_help(fn):
    sig = inspect.signature(fn)
    parts = []
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            parts.append(name)
        else:
            parts.append(f"{name}={param.default!r}")
    return ", ".join(parts) if parts else "(no parameters)"


def run_interactive_scenario_repl(scenarios: Iterable[tuple[str, Callable]], *, sep: str):
    if not sys.stdin.isatty():
        return
    scenario_pairs = list(scenarios)
    scenario_map = {label.upper(): fn for label, fn in scenario_pairs}
    print(f"\n{sep}")
    print("INTERACTIVE RE-RUN MODE")
    print(sep)
    print("Select a scenario for rerun with custom parameters.")
    print(f"Available labels: {', '.join(label for label, _ in scenario_pairs)}")
    print("Enter a scenario label or press Enter to exit.")

    while True:
        try:
            choice = input("\nScenario label to rerun: ").strip().upper()
        except EOFError:
            print("\nInteractive mode closed.")
            return
        if not choice:
            print("Interactive rerun mode finished.")
            return
        if choice not in scenario_map:
            print(f"Unknown scenario '{choice}'. Available: {', '.join(scenario_map)}")
            continue
        fn = scenario_map[choice]
        print(f"Selected scenario {choice}: {fn.__name__}")
        print(f"Parameters: {format_signature_help(fn)}")
        print("Enter overrides as comma-separated key=value pairs.")
        print("Press Enter to use defaults.")
        try:
            raw_kwargs = input("Custom parameters: ")
            kwargs = parse_kwargs_text(raw_kwargs)
            print(f"\nExecuting scenario {choice} with parameters: {kwargs if kwargs else 'defaults'}")
            fn(**kwargs)
        except Exception as exc:
            print(f"\nScenario {choice} failed during custom execution.")
            print(f"Error: {exc}")
            traceback.print_exc()


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def transpile_for_hardware(qc, coupling_map=None, basis_gates=None, optimization_level=3):
    t_qc = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    return t_qc, int(t_qc.depth()), t_qc.count_ops()
