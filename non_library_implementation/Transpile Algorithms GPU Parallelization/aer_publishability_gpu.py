from __future__ import annotations

import ast
import functools
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    import qiskit
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]
    qiskit = None  # type: ignore[assignment]

_AER_IMPORT_ERROR: Exception | None = None
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
except Exception as exc:  # pragma: no cover
    _AER_IMPORT_ERROR = exc
    AerSimulator = None  # type: ignore[assignment]
    SamplerV2 = None  # type: ignore[assignment]
    NoiseModel = None  # type: ignore[assignment]
    ReadoutError = None  # type: ignore[assignment]
    depolarizing_error = None  # type: ignore[assignment]

try:
    from entanglement_monitor_gpu import EntanglementConfig, profile_circuit_entanglement
except Exception:  # pragma: no cover
    EntanglementConfig = None  # type: ignore[assignment]
    profile_circuit_entanglement = None  # type: ignore[assignment]


_DEFAULT_LOG_NAME = "backend_validation_log.jsonl"
_CAPTURE_PROMPT_ENV = "AER_PUBLISHABILITY_CAPTURE_PROMPT"
_AER_GPU_DEVICE = "GPU"
_AER_GPU_HINT = (
    "This transpilation folder now expects qiskit-aer-gpu on a CUDA-capable Linux/x86_64 system. "
    "If you use CUDA 11, install qiskit-aer-gpu-cu11 instead. "
    "If you see a libnvidia-ml.so.1 error, the NVIDIA driver runtime is missing."
)
_NOISE_PRESETS: dict[str, tuple[float, float, float]] = {
    "ideal": (0.0, 0.0, 0.0),
    "light": (0.001, 0.01, 0.02),
    "medium": (0.003, 0.03, 0.05),
    "heavy": (0.01, 0.08, 0.10),
}


@dataclass(frozen=True)
class PublishabilityConfig:
    enabled: bool = True
    mode: str = "backend_validation"
    shots: int = 1024
    seed: int = 42
    # <= 0 means no limit.
    max_qubits: int = 0
    noise_level: str = "ideal"
    one_qubit_error: float = 0.0
    two_qubit_error: float = 0.0
    readout_error: float = 0.0
    log_dir: str | None = None
    log_name: str = _DEFAULT_LOG_NAME
    entanglement_enabled: bool = False
    # <= 0 means no limit.
    entanglement_max_qubits: int = 0
    entanglement_max_snapshots: int = 64
    entanglement_every_step: bool = False
    state_enabled: bool = False
    # <= 0 means no limit.
    state_max_qubits: int = 0
    state_top_k: int = 16
    state_include_full: bool = False
    state_every_step: bool = False
    validation_workers: int = 0

    @property
    def structured_log_path(self) -> str | None:
        if not self.log_dir:
            return None
        return os.path.join(self.log_dir, self.log_name)

    def summary(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        ent_limit = "unlimited" if self.entanglement_max_qubits <= 0 else self.entanglement_max_qubits
        state_limit = "unlimited" if self.state_max_qubits <= 0 else self.state_max_qubits
        validation_limit = "unlimited" if self.max_qubits <= 0 else self.max_qubits
        entanglement = (
            "enabled("
            f"limit={ent_limit}, snapshots={self.entanglement_max_snapshots}, "
            f"every_step={self.entanglement_every_step}"
            ")"
            if self.entanglement_enabled
            else "disabled"
        )
        states = (
            "enabled("
            f"limit={state_limit}, top_k={self.state_top_k}, "
            f"full={self.state_include_full}, every_step={self.state_every_step}"
            ")"
            if self.state_enabled
            else "disabled"
        )
        workers = self.validation_workers or 1
        return (
            "[Backend Validation Configuration] "
            f"mode={self.mode} ({state}), validation_limit={validation_limit}, shots={self.shots}, "
            f"seed={self.seed}, noise={self.noise_level} "
            f"(1q={self.one_qubit_error:.4g}, 2q={self.two_qubit_error:.4g}, readout={self.readout_error:.4g}), "
            f"validation_log={self.structured_log_path or 'disabled'}, workers={workers}, "
            f"entanglement={entanglement}, states={states}"
        )


@dataclass
class _RecordedCircuit:
    source: str
    circuit: "QuantumCircuit"


def _limit_allows(limit: int, num_qubits: int) -> bool:
    return int(limit) <= 0 or num_qubits <= int(limit)


def figure_metadata_path(figure_path: str) -> str:
    root, _ = os.path.splitext(str(figure_path))
    return root + ".metadata.json"


def write_figure_metadata(figure_path: str, metadata: dict[str, Any]) -> str:
    payload = dict(metadata or {})
    payload.setdefault("figure_path", str(figure_path))
    payload.setdefault("metadata_version", 1)
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    meta_path = figure_metadata_path(str(figure_path))
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved figure metadata to: {meta_path}")
    return meta_path


def save_figure_with_metadata(
    fig: Any,
    figure_path: str,
    metadata: dict[str, Any],
    *,
    dpi: int = 220,
    bbox_inches: str = "tight",
) -> str:
    fig.savefig(figure_path, dpi=dpi, bbox_inches=bbox_inches)
    write_figure_metadata(figure_path, metadata)
    return figure_path


def _has_measurements(qc: "QuantumCircuit") -> bool:
    return any(inst.operation.name == "measure" for inst in qc.data)


def _normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    shots = float(sum(counts.values()))
    if shots <= 0:
        return {}
    return {key: float(val) / shots for key, val in counts.items()}


def _total_variation_distance(left: dict[str, float], right: dict[str, float]) -> float:
    support = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in support)


def _extract_sampler_counts(result_item: Any) -> dict[str, int]:
    data = getattr(result_item, "data", None)
    if data is None:
        return {}
    for key in getattr(data, "keys", lambda: [])():
        payload = getattr(data, key, None)
        if hasattr(payload, "get_counts"):
            return payload.get_counts()
    return {}


def _consume_cli_value(argv: Sequence[str], index: int, flag: str) -> tuple[str, int]:
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} requires a value.")
    return str(argv[index + 1]), index + 2


def _resolve_noise_values(
    noise_level: str,
    one_qubit_error: float | None,
    two_qubit_error: float | None,
    readout_error: float | None,
) -> tuple[str, float, float, float]:
    noise_key = str(noise_level).strip().lower()
    if noise_key in {"off", "none"}:
        noise_key = "ideal"

    if noise_key not in _NOISE_PRESETS and noise_key != "custom":
        raise ValueError(
            "Unsupported noise level. Choose from ideal, light, medium, heavy, custom, off."
        )

    if noise_key == "custom":
        return (
            "custom",
            float(one_qubit_error or 0.0),
            float(two_qubit_error or 0.0),
            float(readout_error or 0.0),
        )

    base_one, base_two, base_readout = _NOISE_PRESETS[noise_key]
    return (
        noise_key,
        float(base_one if one_qubit_error is None else one_qubit_error),
        float(base_two if two_qubit_error is None else two_qubit_error),
        float(base_readout if readout_error is None else readout_error),
    )


def _capture_prompt_allowed() -> bool:
    raw = str(os.environ.get(_CAPTURE_PROMPT_ENV, "1")).strip().lower()
    if raw in {"0", "false", "off", "no", "disabled"}:
        return False
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def _prompt_capture_mode() -> str:
    print("\nCapture mode selection")
    print("Full mode records state and entanglement at every step and can take more computing time.")
    print("Light mode records sampled checkpoints and is faster.")
    while True:
        choice = input("Choose [F]ull or [L]ight (press Enter for Full): ").strip().lower()
        if choice in {"", "f", "full"}:
            return "full"
        if choice in {"l", "light"}:
            return "light"
        print("Please enter F, L, or press Enter for Full.")


def parse_publishability_cli(
    argv: Sequence[str],
    *,
    default_max_qubits: int = 0,
    default_shots: int = 1024,
    default_seed: int = 42,
    default_log_dir: str | None = None,
    default_log_name: str = _DEFAULT_LOG_NAME,
    default_entanglement_max_qubits: int = 0,
    default_entanglement_max_snapshots: int = 64,
    default_state_max_qubits: int = 0,
    default_state_top_k: int = 16,
    default_validation_workers: int = 0,
) -> tuple[list[str], PublishabilityConfig]:
    def apply_capture_mode(mode_name: str) -> None:
        nonlocal entanglement_enabled
        nonlocal entanglement_max_snapshots
        nonlocal entanglement_every_step
        nonlocal state_enabled
        nonlocal state_include_full
        nonlocal state_every_step

        mode_key = str(mode_name).strip().lower()
        if mode_key == "full":
            entanglement_enabled = True
            entanglement_every_step = True
            state_enabled = True
            state_include_full = True
            state_every_step = True
            return
        if mode_key == "light":
            entanglement_enabled = True
            entanglement_every_step = False
            entanglement_max_snapshots = min(int(default_entanglement_max_snapshots), 8)
            state_enabled = True
            state_include_full = True
            state_every_step = False
            return
        raise ValueError("Unsupported --capture-mode value. Choose 'full' or 'light'.")

    remaining: list[str] = []
    mode = "backend_validation"
    max_qubits = int(default_max_qubits)
    shots = int(default_shots)
    seed = int(default_seed)
    noise_level = "ideal"
    one_qubit_error: float | None = None
    two_qubit_error: float | None = None
    readout_error: float | None = None
    log_name = str(default_log_name)
    validation_workers = int(default_validation_workers)
    entanglement_enabled = True
    entanglement_max_qubits = int(default_entanglement_max_qubits)
    entanglement_max_snapshots = int(default_entanglement_max_snapshots)
    entanglement_every_step = True
    state_enabled = True
    state_max_qubits = int(default_state_max_qubits)
    state_top_k = int(default_state_top_k)
    state_include_full = True
    state_every_step = True
    capture_mode: str | None = None
    capture_mode_explicit = False
    monitoring_flags_explicit = False

    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--mode":
            raw, index = _consume_cli_value(argv, index, "--mode")
            mode = raw.strip().lower()
            continue
        if token == "--qubits":
            raw, index = _consume_cli_value(argv, index, "--qubits")
            max_qubits = int(raw)
            continue
        if token == "--shots":
            raw, index = _consume_cli_value(argv, index, "--shots")
            shots = int(raw)
            continue
        if token == "--seed":
            raw, index = _consume_cli_value(argv, index, "--seed")
            seed = int(raw)
            continue
        if token == "--noise":
            raw, index = _consume_cli_value(argv, index, "--noise")
            noise_level = raw.strip().lower()
            continue
        if token == "--noise-1q":
            raw, index = _consume_cli_value(argv, index, "--noise-1q")
            one_qubit_error = float(raw)
            continue
        if token == "--noise-2q":
            raw, index = _consume_cli_value(argv, index, "--noise-2q")
            two_qubit_error = float(raw)
            continue
        if token == "--readout":
            raw, index = _consume_cli_value(argv, index, "--readout")
            readout_error = float(raw)
            continue
        if token in {"--publish-log", "--validation-log"}:
            raw, index = _consume_cli_value(argv, index, token)
            log_name = raw.strip()
            continue
        if token == "--capture-mode":
            raw, index = _consume_cli_value(argv, index, "--capture-mode")
            capture_mode = str(raw).strip().lower()
            capture_mode_explicit = True
            apply_capture_mode(capture_mode)
            continue
        if token == "--entanglement":
            monitoring_flags_explicit = True
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
                raw, index = _consume_cli_value(argv, index, "--entanglement")
                entanglement_enabled = str(raw).strip().lower() not in {"0", "false", "off", "no", "disabled"}
            else:
                entanglement_enabled = True
                index += 1
            continue
        if token == "--no-entanglement":
            monitoring_flags_explicit = True
            entanglement_enabled = False
            index += 1
            continue
        if token == "--entanglement-qubits":
            monitoring_flags_explicit = True
            raw, index = _consume_cli_value(argv, index, "--entanglement-qubits")
            entanglement_max_qubits = int(raw)
            continue
        if token == "--entanglement-snapshots":
            monitoring_flags_explicit = True
            raw, index = _consume_cli_value(argv, index, "--entanglement-snapshots")
            entanglement_max_snapshots = int(raw)
            continue
        if token == "--entanglement-every-step":
            monitoring_flags_explicit = True
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
                raw, index = _consume_cli_value(argv, index, "--entanglement-every-step")
                entanglement_every_step = str(raw).strip().lower() not in {"0", "false", "off", "no", "disabled"}
            else:
                entanglement_every_step = True
                index += 1
            continue
        if token in {"--states", "--state-monitor"}:
            monitoring_flags_explicit = True
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
                raw, index = _consume_cli_value(argv, index, token)
                state_enabled = str(raw).strip().lower() not in {"0", "false", "off", "no", "disabled"}
            else:
                state_enabled = True
                index += 1
            continue
        if token == "--no-states":
            monitoring_flags_explicit = True
            state_enabled = False
            index += 1
            continue
        if token == "--state-qubits":
            monitoring_flags_explicit = True
            raw, index = _consume_cli_value(argv, index, "--state-qubits")
            state_max_qubits = int(raw)
            continue
        if token == "--state-top-k":
            monitoring_flags_explicit = True
            raw, index = _consume_cli_value(argv, index, "--state-top-k")
            state_top_k = int(raw)
            continue
        if token == "--validation-workers":
            raw, index = _consume_cli_value(argv, index, "--validation-workers")
            validation_workers = max(0, int(raw))
            continue
        if token == "--state-full":
            monitoring_flags_explicit = True
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
                raw, index = _consume_cli_value(argv, index, "--state-full")
                state_include_full = str(raw).strip().lower() not in {"0", "false", "off", "no", "disabled"}
            else:
                state_include_full = True
                index += 1
            continue
        if token == "--state-every-step":
            monitoring_flags_explicit = True
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
                raw, index = _consume_cli_value(argv, index, "--state-every-step")
                state_every_step = str(raw).strip().lower() not in {"0", "false", "off", "no", "disabled"}
            else:
                state_every_step = True
                index += 1
            continue
        remaining.append(token)
        index += 1

    if mode in {"off", "disabled", "none"}:
        enabled = False
        mode = "disabled"
    elif mode in {"backend_validation", "validation", "publishability"}:
        enabled = True
        mode = "backend_validation"
    else:
        raise ValueError(
            "Unsupported --mode value. Choose 'backend_validation' or 'disabled'. "
            "The legacy alias 'publishability' is still accepted."
        )

    if any(value is not None for value in (one_qubit_error, two_qubit_error, readout_error)) and noise_level == "ideal":
        noise_level = "custom"

    if capture_mode is None and not capture_mode_explicit and not monitoring_flags_explicit and _capture_prompt_allowed():
        capture_mode = _prompt_capture_mode()
        apply_capture_mode(capture_mode)

    if (state_include_full or state_every_step) and not state_enabled:
        state_enabled = True

    if state_every_step and not entanglement_every_step:
        entanglement_every_step = True

    if entanglement_every_step and not entanglement_enabled:
        entanglement_enabled = True

    if state_enabled and not entanglement_enabled:
        entanglement_enabled = True

    noise_level, one_qubit, two_qubit, readout = _resolve_noise_values(
        noise_level,
        one_qubit_error,
        two_qubit_error,
        readout_error,
    )

    config = PublishabilityConfig(
        enabled=enabled,
        mode=mode,
        shots=shots,
        seed=seed,
        max_qubits=max_qubits,
        noise_level=noise_level,
        one_qubit_error=one_qubit,
        two_qubit_error=two_qubit,
        readout_error=readout,
        log_dir=default_log_dir,
        log_name=log_name,
        entanglement_enabled=entanglement_enabled,
        entanglement_max_qubits=entanglement_max_qubits,
        entanglement_max_snapshots=entanglement_max_snapshots,
        entanglement_every_step=entanglement_every_step,
        state_enabled=state_enabled,
        state_max_qubits=state_max_qubits,
        state_top_k=state_top_k,
        state_include_full=state_include_full,
        state_every_step=state_every_step,
        validation_workers=validation_workers,
    )
    return remaining, config


def _parse_cli_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        return raw


def _split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False

    for ch in text:
        if quote is not None:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
            current.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def parse_scenario_cli(argv: Sequence[str]) -> tuple[str | None, dict[str, Any]]:
    tokens = [str(token).strip() for token in argv if str(token).strip()]
    if not tokens:
        return None, {}

    label: str | None = None
    if tokens and "=" not in tokens[0]:
        label = tokens.pop(0).upper()

    raw_kwargs = ", ".join(tokens)
    kwargs: dict[str, Any] = {}
    if raw_kwargs:
        for item in _split_top_level_commas(raw_kwargs):
            if "=" not in item:
                raise ValueError(f"Expected key=value pair, got '{item}'")
            key, value = item.split("=", 1)
            kwargs[key.strip()] = _parse_cli_value(value.strip())

    if "case" in kwargs:
        label = str(kwargs.pop("case")).strip().upper()
    if "scenario" in kwargs:
        label = str(kwargs.pop("scenario")).strip().upper()

    if label is None and kwargs:
        raise ValueError(
            "Scenario parameters were provided without a scenario label. "
            "Prefix the command with a label like 'A' or pass case=A."
        )
    return label, kwargs


def run_cli_scenario(
    argv: Sequence[str],
    scenarios: Sequence[tuple[str, Any]],
    *,
    label_name: str = "scenario",
) -> bool:
    label, kwargs = parse_scenario_cli(argv)
    if label is None:
        return False

    scenario_pairs = list(scenarios)
    scenario_map = {str(case_label).upper(): fn for case_label, fn in scenario_pairs}
    if label not in scenario_map:
        raise ValueError(
            f"Unknown {label_name} label '{label}'. Available: {', '.join(scenario_map)}"
        )

    print("Direct command-line execution requested; skipping the default suite.")
    print(f"Command-line {label_name} selection: {label}")
    print(f"Command-line parameters: {kwargs if kwargs else 'defaults'}")
    scenario_map[label](**kwargs)
    return True


def prepare_backend_validation_artifacts(config: PublishabilityConfig) -> None:
    log_path = config.structured_log_path
    if not config.enabled or not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8"):
        pass

    for filename in (
        "backend_validation_summary.png",
        "backend_validation_summary.txt",
        "backend_validation_resource_profile.png",
        "backend_validation_agreement_profile.png",
        "backend_validation_entanglement_profile.png",
        "backend_validation_entanglement_profile.txt",
        "backend_validation_entanglement_profile.json",
        "backend_validation_state_profile.txt",
        "backend_validation_state_profile.json",
    ):
        artifact_path = os.path.join(os.path.dirname(log_path), filename)
        if os.path.exists(artifact_path):
            os.remove(artifact_path)


def _load_backend_validation_records(log_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not log_path or not os.path.exists(log_path):
        return records
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _default_suite_title(config: PublishabilityConfig) -> str:
    if config.log_dir:
        label = os.path.basename(config.log_dir)
        if label.startswith("[RESULT]"):
            label = label[len("[RESULT]"):]
        return label.replace("_", " ")
    return "Backend Validation"


def _normalize_validation_status(status: str) -> str:
    key = str(status).strip()
    if key in {"skipped_too_many_qubits", "deferred_heavy_validation"}:
        return "deferred_heavy_validation"
    return key


def _dominant_probability(counts: Any) -> float:
    if not isinstance(counts, dict) or not counts:
        return 0.0
    total = float(sum(float(value) for value in counts.values()))
    if total <= 0.0:
        return 0.0
    return max(float(value) for value in counts.values()) / total


def _support_size(counts: Any) -> int:
    if not isinstance(counts, dict):
        return 0
    return sum(1 for value in counts.values() if float(value) > 0.0)


def _entanglement_record_label(record: dict[str, Any]) -> str:
    scenario = str(record.get("scenario", "unknown"))
    source = str(record.get("source", "unknown") or "unknown")
    return f"{scenario}:{source}"


def _render_entanglement_summary(
    config: PublishabilityConfig,
    records: list[dict[str, Any]],
    *,
    suite_title: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    if not config.entanglement_enabled or not config.structured_log_path:
        return None, None, None

    output_dir = os.path.dirname(config.structured_log_path)
    summary_txt = os.path.join(output_dir, "backend_validation_entanglement_profile.txt")
    summary_png = os.path.join(output_dir, "backend_validation_entanglement_profile.png")
    summary_json = os.path.join(output_dir, "backend_validation_entanglement_profile.json")

    ent_records = [
        record
        for record in records
        if "entanglement_status" in record and str(record.get("status", "")) != "scenario_complete"
    ]
    if not ent_records:
        return None, None, None

    status_counts: dict[str, int] = {}
    summary_lines = [
        "Backend Validation Entanglement Summary",
        "=======================================",
        f"suite={suite_title or _default_suite_title(config)}",
        f"log_path={config.structured_log_path}",
        f"entanglement_limit={config.entanglement_max_qubits}",
        f"snapshot_limit={config.entanglement_max_snapshots}",
        f"entanglement_every_step={config.entanglement_every_step}",
        "metric_note=single-qubit entropy is exact entanglement on pure-state segments; "
        "after measurement/reset it becomes a mixed-state entropy proxy.",
        "",
    ]

    json_payload_records: list[dict[str, Any]] = []
    for record in ent_records:
        label = _entanglement_record_label(record)
        ent_status = str(record.get("entanglement_status", "unknown"))
        status_counts[ent_status] = status_counts.get(ent_status, 0) + 1
        line = f"{label}: status={ent_status}"
        if ent_status == "ok":
            line += (
                f", mode={record.get('entanglement_simulation_mode')}"
                f", every_step={bool(record.get('entanglement_entanglement_every_step', False))}"
                f", peak_mean={float(record.get('entanglement_peak_mean_single_qubit_entropy', 0.0)):.6f}"
                f", peak_total={float(record.get('entanglement_peak_total_single_qubit_entropy', 0.0)):.6f}"
                f", peak_q={float(record.get('entanglement_peak_meyer_wallach_q', 0.0)):.6f}"
                f", snapshots={int(record.get('entanglement_snapshots_recorded', 0))}"
            )
        elif ent_status == "skipped_too_many_qubits":
            line += (
                f", qubits={int(record.get('entanglement_qubits', 0) or 0)}"
                f", required_limit={int(record.get('entanglement_required_entanglement_limit', 0) or 0)}"
            )
        elif "entanglement_error" in record:
            line += f", error={record.get('entanglement_error')}"
        summary_lines.append(line)

        json_payload_records.append(
            {
                "label": label,
                "scenario": record.get("scenario"),
                "source": record.get("source"),
                "backend_status": record.get("status"),
                "entanglement_status": ent_status,
                "simulation_mode": record.get("entanglement_simulation_mode"),
                "qubits": record.get("entanglement_qubits", record.get("qubits")),
                "tracked_quantum_steps": record.get("entanglement_tracked_quantum_steps"),
                "entanglement_every_step": record.get("entanglement_entanglement_every_step"),
                "snapshots_recorded": record.get("entanglement_snapshots_recorded"),
                "initial_mean_single_qubit_entropy": record.get("entanglement_initial_mean_single_qubit_entropy"),
                "initial_total_single_qubit_entropy": record.get("entanglement_initial_total_single_qubit_entropy"),
                "initial_meyer_wallach_q": record.get("entanglement_initial_meyer_wallach_q"),
                "peak_mean_single_qubit_entropy": record.get("entanglement_peak_mean_single_qubit_entropy"),
                "peak_total_single_qubit_entropy": record.get("entanglement_peak_total_single_qubit_entropy"),
                "peak_meyer_wallach_q": record.get("entanglement_peak_meyer_wallach_q"),
                "final_mean_single_qubit_entropy": record.get("entanglement_final_mean_single_qubit_entropy"),
                "final_total_single_qubit_entropy": record.get("entanglement_final_total_single_qubit_entropy"),
                "final_meyer_wallach_q": record.get("entanglement_final_meyer_wallach_q"),
                "contains_midcircuit_collapse": record.get("entanglement_contains_midcircuit_collapse"),
                "first_nonunitary_step": record.get("entanglement_first_nonunitary_step"),
                "trace": record.get("entanglement_trace", []),
            }
        )

    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "suite": suite_title or _default_suite_title(config),
                "entanglement_limit": config.entanglement_max_qubits,
                "snapshot_limit": config.entanglement_max_snapshots,
                "records": json_payload_records,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"Backend entanglement figure skipped ({exc}).")
        print(f"Saved entanglement summary to: {summary_txt}")
        print(f"Saved entanglement trace export to: {summary_json}")
        return None, summary_txt, summary_json

    ok_records = [record for record in ent_records if str(record.get("entanglement_status")) == "ok"]
    if not ok_records:
        print(f"Saved entanglement summary to: {summary_txt}")
        print(f"Saved entanglement trace export to: {summary_json}")
        return None, summary_txt, summary_json

    labels = [_entanglement_record_label(record) for record in ok_records]
    xs = np.arange(len(ok_records), dtype=float)
    peak_mean = np.array(
        [float(record.get("entanglement_peak_mean_single_qubit_entropy", 0.0) or 0.0) for record in ok_records],
        dtype=float,
    )
    peak_total = np.array(
        [float(record.get("entanglement_peak_total_single_qubit_entropy", 0.0) or 0.0) for record in ok_records],
        dtype=float,
    )
    peak_q = np.array(
        [float(record.get("entanglement_peak_meyer_wallach_q", 0.0) or 0.0) for record in ok_records],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        f"{suite_title or _default_suite_title(config)}\nBackend Validation Entanglement Profile",
        fontsize=14,
        fontweight="bold",
    )

    ax_a, ax_b, ax_c, ax_d = axes.flat

    ax_a.bar(xs, peak_mean, color="#1f77b4", alpha=0.85)
    ax_a.scatter(xs, peak_q, color="black", s=30, zorder=3, label="peak Meyer-Wallach Q")
    ax_a.set_title("Peak Mean Single-Qubit Entropy")
    ax_a.set_ylabel("Entropy / Q")
    ax_a.set_xticks(xs, labels, rotation=20, ha="right")
    ax_a.legend(fontsize=8)

    ax_b.bar(xs, peak_total, color="#ff7f0e", alpha=0.85)
    ax_b.set_title("Peak Total Single-Qubit Entropy")
    ax_b.set_ylabel("Entropy sum")
    ax_b.set_xticks(xs, labels, rotation=20, ha="right")

    mode_style = {"statevector": ("#2ca02c", "-"), "density_matrix": ("#d62728", "--")}
    for record in ok_records:
        trace = record.get("entanglement_trace", []) or []
        if not trace:
            continue
        tracked_steps = int(record.get("entanglement_tracked_quantum_steps", len(trace)) or len(trace) or 1)
        progress = np.array(
            [float(point.get("quantum_step", 0)) / float(max(1, tracked_steps)) for point in trace],
            dtype=float,
        )
        mean_entropy = np.array(
            [float(point.get("mean_single_qubit_entropy", 0.0) or 0.0) for point in trace],
            dtype=float,
        )
        mode = str(record.get("entanglement_simulation_mode", "statevector"))
        color, linestyle = mode_style.get(mode, ("#7f7f7f", "-"))
        ax_c.plot(progress, mean_entropy, color=color, linestyle=linestyle, alpha=0.45, linewidth=1.8)
    ax_c.set_title("Mean Entanglement Trace")
    ax_c.set_xlabel("Normalized circuit progress")
    ax_c.set_ylabel("Mean single-qubit entropy")

    status_labels = sorted(status_counts)
    status_values = np.array([float(status_counts[label]) for label in status_labels], dtype=float)
    ax_d.bar(np.arange(len(status_labels), dtype=float), status_values, color="#9467bd", alpha=0.85)
    ax_d.set_title("Entanglement Capture Outcomes")
    ax_d.set_ylabel("Record count")
    ax_d.set_xticks(np.arange(len(status_labels), dtype=float), status_labels, rotation=20, ha="right")

    for axis in axes.flat:
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    save_figure_with_metadata(
        fig,
        summary_png,
        {
            "figure_kind": "backend_validation_entanglement_profile",
            "suite": suite_title or _default_suite_title(config),
            "entanglement_limit": config.entanglement_max_qubits,
            "snapshot_limit": config.entanglement_max_snapshots,
            "record_labels": labels,
            "status_labels": status_labels,
        },
    )
    plt.close(fig)

    print(f"Saved entanglement validation figure to: {summary_png}")
    print(f"Saved entanglement summary to: {summary_txt}")
    print(f"Saved entanglement trace export to: {summary_json}")
    return summary_png, summary_txt, summary_json


def _render_state_summary(
    config: PublishabilityConfig,
    records: list[dict[str, Any]],
    *,
    suite_title: str | None = None,
) -> tuple[str | None, str | None]:
    if not config.state_enabled or not config.structured_log_path:
        return None, None

    output_dir = os.path.dirname(config.structured_log_path)
    summary_txt = os.path.join(output_dir, "backend_validation_state_profile.txt")
    summary_json = os.path.join(output_dir, "backend_validation_state_profile.json")

    state_records = [
        record
        for record in records
        if "entanglement_state_status" in record and str(record.get("status", "")) != "scenario_complete"
    ]
    if not state_records:
        return None, None

    summary_lines = [
        "Backend Validation State Summary",
        "================================",
        f"suite={suite_title or _default_suite_title(config)}",
        f"log_path={config.structured_log_path}",
        f"state_limit={config.state_max_qubits}",
        f"state_top_k={config.state_top_k}",
        f"state_include_full={config.state_include_full}",
        f"state_every_step={config.state_every_step}",
        "note=the primary object here is the global n-qubit state. "
        "Local alpha/beta only appears when one qubit factorizes cleanly from the rest.",
        "",
    ]

    json_records: list[dict[str, Any]] = []
    for record in state_records:
        label = _entanglement_record_label(record)
        state_status = str(record.get("entanglement_state_status", "unknown"))
        line = f"{label}: status={state_status}"
        state_trace = record.get("entanglement_state_trace", []) or []
        initial_snapshot = record.get("entanglement_initial_state_snapshot") or {}
        final_snapshot = record.get("entanglement_final_pre_measurement_state_snapshot") or {}
        final_post_terminal_snapshot = record.get("entanglement_final_post_terminal_state_snapshot") or {}
        if state_status == "ok" and final_snapshot:
            state_kind = str(final_snapshot.get("state_kind", "unknown"))
            if state_kind == "statevector":
                top_basis = (final_snapshot.get("top_basis_states") or [{}])[0]
                dominant_basis = top_basis.get("basis", "n/a")
                dominant_prob = float(top_basis.get("probability", 0.0) or 0.0)
            else:
                top_basis = (final_snapshot.get("top_basis_probabilities") or [{}])[0]
                dominant_basis = top_basis.get("basis", "n/a")
                dominant_prob = float(top_basis.get("probability", 0.0) or 0.0)
            line += (
                f", final_kind={state_kind}"
                f", dominant_basis={dominant_basis}"
                f", dominant_probability={dominant_prob:.6f}"
                f", snapshots={len(state_trace)}"
                f", initial_snapshot={'yes' if initial_snapshot else 'no'}"
                f", final_pre_measurement_step={int(record.get('entanglement_final_pre_measurement_quantum_step', 0) or 0)}"
                f", final_post_terminal_snapshot={'yes' if final_post_terminal_snapshot else 'no'}"
            )
        elif state_status == "skipped_too_many_qubits":
            line += f", required_limit={int(record.get('entanglement_required_state_limit', 0) or 0)}"
        summary_lines.append(line)

        json_records.append(
            {
                "label": label,
                "scenario": record.get("scenario"),
                "source": record.get("source"),
                "backend_status": record.get("status"),
                "state_status": state_status,
                "state_top_k": record.get("entanglement_state_top_k"),
                "state_include_full": record.get("entanglement_state_include_full"),
                "state_every_step": record.get("entanglement_state_every_step"),
                "state_snapshots_recorded": record.get("entanglement_state_snapshots_recorded"),
                "state_measurement_boundaries_recorded": record.get("entanglement_state_measurement_boundaries_recorded"),
                "initial_state_snapshot": record.get("entanglement_initial_state_snapshot"),
                "final_pre_measurement_quantum_step": record.get("entanglement_final_pre_measurement_quantum_step"),
                "final_pre_measurement_state_snapshot": final_snapshot,
                "final_post_terminal_quantum_step": record.get("entanglement_final_post_terminal_quantum_step"),
                "final_post_terminal_state_snapshot": record.get("entanglement_final_post_terminal_state_snapshot"),
                "trace": state_trace,
            }
        )

    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "suite": suite_title or _default_suite_title(config),
                "state_limit": config.state_max_qubits,
                "state_top_k": config.state_top_k,
                "state_include_full": config.state_include_full,
                "records": json_records,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(f"Saved state summary to: {summary_txt}")
    print(f"Saved state trace export to: {summary_json}")
    return summary_txt, summary_json


def render_backend_validation_summary(
    config: PublishabilityConfig,
    *,
    suite_title: str | None = None,
) -> tuple[str | None, str | None]:
    log_path = config.structured_log_path
    if not config.enabled or not log_path:
        return None, None

    records = _load_backend_validation_records(log_path)
    if not records:
        return None, None

    output_dir = os.path.dirname(log_path)
    summary_txt = os.path.join(output_dir, "backend_validation_summary.txt")
    summary_png = os.path.join(output_dir, "backend_validation_summary.png")
    resource_png = os.path.join(output_dir, "backend_validation_resource_profile.png")
    agreement_png = os.path.join(output_dir, "backend_validation_agreement_profile.png")

    scenario_labels = sorted({str(record.get("scenario", "unknown")) for record in records})
    status_order = ["ok", "deferred_heavy_validation", "no_circuits", "backend_unavailable", "error"]
    status_counts: dict[str, dict[str, int]] = {label: {status: 0 for status in status_order} for label in scenario_labels}
    ok_records: dict[str, list[dict[str, Any]]] = {label: [] for label in scenario_labels}
    source_labels = sorted(
        {
            str(record.get("source", "unknown") or "unknown")
            for record in records
            if str(record.get("status", "")) != "scenario_complete"
        }
    )
    source_status_counts: dict[str, dict[str, int]] = {
        label: {status: 0 for status in status_order} for label in source_labels
    }
    source_ok_records: dict[str, list[dict[str, Any]]] = {label: [] for label in source_labels}

    for record in records:
        scenario = str(record.get("scenario", "unknown"))
        status = _normalize_validation_status(str(record.get("status", "unknown")))
        if status == "scenario_complete":
            continue
        source = str(record.get("source", "unknown") or "unknown")
        if scenario not in status_counts:
            status_counts[scenario] = {name: 0 for name in status_order}
            ok_records[scenario] = []
            scenario_labels.append(scenario)
        if source not in source_status_counts:
            source_status_counts[source] = {name: 0 for name in status_order}
            source_ok_records[source] = []
            source_labels.append(source)
        if status in status_counts[scenario]:
            status_counts[scenario][status] += 1
        if status in source_status_counts[source]:
            source_status_counts[source][status] += 1
        if status == "ok":
            ok_records[scenario].append(record)
            source_ok_records[source].append(record)

    scenario_labels = sorted(set(scenario_labels))
    source_labels = sorted(set(source_labels))
    summary_lines = [
        "Backend Validation Summary",
        "==========================",
        f"suite={suite_title or _default_suite_title(config)}",
        f"log_path={log_path}",
        f"shots={config.shots}",
        f"validation_limit={config.max_qubits}",
        f"noise={config.noise_level}",
        "status_note=ok means AerSimulator/SamplerV2 validation completed successfully for that captured circuit.",
        "",
    ]

    for scenario in scenario_labels:
        ok_group = ok_records.get(scenario, [])
        total_ok = len(ok_group)
        top_matches = sum(1 for record in ok_group if record.get("sim_top") == record.get("sampler_top"))
        match_rate = 100.0 * top_matches / total_ok if total_ok else float("nan")
        mean_tvd = (
            sum(float(record.get("tvd", 0.0)) for record in ok_group) / total_ok
            if total_ok
            else float("nan")
        )
        max_tvd = max((float(record.get("tvd", 0.0)) for record in ok_group), default=float("nan"))
        mean_depth = (
            sum(float(record.get("depth", 0.0)) for record in ok_group) / total_ok
            if total_ok
            else float("nan")
        )
        max_depth = max((float(record.get("depth", 0.0)) for record in ok_group), default=float("nan"))
        counts = status_counts.get(scenario, {})
        summary_lines.append(
            f"{scenario}: ok={counts.get('ok', 0)}, deferred={counts.get('deferred_heavy_validation', 0)}, "
            f"no_circuits={counts.get('no_circuits', 0)}, unavailable={counts.get('backend_unavailable', 0)}, "
            f"errors={counts.get('error', 0)}, top_agreement_pct={match_rate if total_ok else 'n/a'}, "
            f"mean_tvd={mean_tvd if total_ok else 'n/a'}, max_tvd={max_tvd if total_ok else 'n/a'}, "
            f"mean_depth={mean_depth if total_ok else 'n/a'}, max_depth={max_depth if total_ok else 'n/a'}"
        )

    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"Backend validation figure skipped ({exc}).")
        print(f"Saved backend validation summary to: {summary_txt}")
        return None, summary_txt

    x = np.arange(len(scenario_labels), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        f"{suite_title or _default_suite_title(config)}\nBackend Validation Summary",
        fontsize=14,
        fontweight="bold",
    )

    ax_status, ax_agree, ax_tvd, ax_depth = axes.flat
    colors = {
        "ok": "#1f77b4",
        "deferred_heavy_validation": "#ff7f0e",
        "no_circuits": "#7f7f7f",
        "backend_unavailable": "#d62728",
        "error": "#9467bd",
    }
    status_display = {
        "ok": "validated",
        "deferred_heavy_validation": "deferred: heavy qubits",
        "no_circuits": "no circuits",
        "backend_unavailable": "backend unavailable",
        "error": "error",
    }
    bottom = np.zeros(len(scenario_labels), dtype=float)
    for status in status_order:
        values = np.array([status_counts[label].get(status, 0) for label in scenario_labels], dtype=float)
        ax_status.bar(x, values, bottom=bottom, label=status_display.get(status, status), color=colors.get(status))
        bottom += values
    ax_status.set_title("Validation Outcomes by Scenario")
    ax_status.set_ylabel("Record count")
    ax_status.set_xticks(x, scenario_labels)
    ax_status.legend(fontsize=8)

    agreement_vals = []
    max_tvd_vals = []
    mean_tvd_vals = []
    mean_depth_vals = []
    max_depth_vals = []
    for label in scenario_labels:
        group = ok_records.get(label, [])
        if group:
            top_matches = sum(1 for record in group if record.get("sim_top") == record.get("sampler_top"))
            agreement_vals.append(100.0 * top_matches / len(group))
            max_tvd_vals.append(max(float(record.get("tvd", 0.0)) for record in group))
            mean_tvd_vals.append(sum(float(record.get("tvd", 0.0)) for record in group) / len(group))
            mean_depth_vals.append(sum(float(record.get("depth", 0.0)) for record in group) / len(group))
            max_depth_vals.append(max(float(record.get("depth", 0.0)) for record in group))
        else:
            agreement_vals.append(0.0)
            max_tvd_vals.append(0.0)
            mean_tvd_vals.append(0.0)
            mean_depth_vals.append(0.0)
            max_depth_vals.append(0.0)

    ax_agree.bar(x, agreement_vals, color="#2ca02c")
    ax_agree.set_title("Top-State Agreement")
    ax_agree.set_ylabel("Agreement (%)")
    ax_agree.set_ylim(0, 105)
    ax_agree.set_xticks(x, scenario_labels)

    ax_tvd.bar(x, max_tvd_vals, color="#d62728", alpha=0.75, label="max TVD")
    ax_tvd.scatter(x, mean_tvd_vals, color="black", s=35, zorder=3, label="mean TVD")
    ax_tvd.set_title("Distribution Gap")
    ax_tvd.set_ylabel("Total variation distance")
    ax_tvd.set_xticks(x, scenario_labels)
    ax_tvd.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_tvd.legend(fontsize=8)

    ax_depth.bar(x, mean_depth_vals, color="#1f77b4", alpha=0.75, label="mean depth")
    ax_depth.scatter(x, max_depth_vals, color="black", s=35, zorder=3, label="max depth")
    ax_depth.set_title("Circuit Depth")
    ax_depth.set_ylabel("Transpiled depth")
    ax_depth.set_xticks(x, scenario_labels)
    ax_depth.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    save_figure_with_metadata(
        fig,
        summary_png,
        {
            "figure_kind": "backend_validation_summary",
            "suite": suite_title or _default_suite_title(config),
            "shots": config.shots,
            "validation_limit": config.max_qubits,
            "noise_level": config.noise_level,
            "scenario_labels": list(scenario_labels),
            "status_order": list(status_order),
        },
    )
    plt.close(fig)

    scenario_qubit_mean: list[float] = []
    scenario_qubit_max: list[float] = []
    scenario_depth_mean: list[float] = []
    scenario_depth_max: list[float] = []
    scenario_required_limit: list[float] = []
    for label in scenario_labels:
        active = [record for record in records if str(record.get("scenario", "unknown")) == label]
        qubits = [float(record.get("qubits", 0.0) or 0.0) for record in active]
        depths = [float(record.get("depth", 0.0) or 0.0) for record in active if record.get("depth") is not None]
        required = [
            float(record.get("required_validation_limit", 0.0) or 0.0)
            for record in active
            if record.get("required_validation_limit") is not None
        ]
        scenario_qubit_mean.append(sum(qubits) / len(qubits) if qubits else 0.0)
        scenario_qubit_max.append(max(qubits) if qubits else 0.0)
        scenario_depth_mean.append(sum(depths) / len(depths) if depths else 0.0)
        scenario_depth_max.append(max(depths) if depths else 0.0)
        scenario_required_limit.append(max(required) if required else 0.0)

    xs = np.arange(len(source_labels), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        f"{suite_title or _default_suite_title(config)}\nBackend Validation Resource Profile",
        fontsize=14,
        fontweight="bold",
    )
    ax_q, ax_d, ax_source, ax_limit = axes.flat

    ax_q.bar(x, scenario_qubit_mean, color="#1f77b4", alpha=0.8, label="mean qubits")
    ax_q.scatter(x, scenario_qubit_max, color="black", s=35, zorder=3, label="max qubits")
    ax_q.set_title("Circuit Width by Scenario")
    ax_q.set_ylabel("Qubits")
    ax_q.set_xticks(x, scenario_labels)
    ax_q.legend(fontsize=8)

    ax_d.bar(x, scenario_depth_mean, color="#2ca02c", alpha=0.8, label="mean depth")
    ax_d.scatter(x, scenario_depth_max, color="black", s=35, zorder=3, label="max depth")
    ax_d.set_title("Circuit Depth by Scenario")
    ax_d.set_ylabel("Transpiled depth")
    ax_d.set_xticks(x, scenario_labels)
    ax_d.legend(fontsize=8)

    source_bottom = np.zeros(len(source_labels), dtype=float)
    for status in status_order:
        values = np.array([source_status_counts[label].get(status, 0) for label in source_labels], dtype=float)
        ax_source.bar(
            xs,
            values,
            bottom=source_bottom,
            label=status_display.get(status, status),
            color=colors.get(status),
        )
        source_bottom += values
    ax_source.set_title("Validation Outcomes by Capture Source")
    ax_source.set_ylabel("Record count")
    ax_source.set_xticks(xs, source_labels, rotation=15, ha="right")
    ax_source.legend(fontsize=8)

    ax_limit.bar(x, scenario_required_limit, color="#ff7f0e", alpha=0.85, label="required validation limit")
    ax_limit.axhline(float(config.max_qubits), color="black", linestyle="--", linewidth=1.1, label="current limit")
    ax_limit.set_title("Heavy-Circuit Validation Thresholds")
    ax_limit.set_ylabel("Qubit limit")
    ax_limit.set_xticks(x, scenario_labels)
    ax_limit.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    save_figure_with_metadata(
        fig,
        resource_png,
        {
            "figure_kind": "backend_validation_resource_profile",
            "suite": suite_title or _default_suite_title(config),
            "shots": config.shots,
            "validation_limit": config.max_qubits,
            "noise_level": config.noise_level,
            "scenario_labels": list(scenario_labels),
            "source_labels": list(source_labels),
        },
    )
    plt.close(fig)

    source_agreement: list[float] = []
    source_max_tvd: list[float] = []
    source_mean_tvd: list[float] = []
    source_mean_prob_gap: list[float] = []
    source_sampler_top_prob: list[float] = []
    source_support_size: list[float] = []
    for label in source_labels:
        group = source_ok_records.get(label, [])
        if group:
            top_matches = sum(1 for record in group if record.get("sim_top") == record.get("sampler_top"))
            source_agreement.append(100.0 * top_matches / len(group))
            source_max_tvd.append(max(float(record.get("tvd", 0.0) or 0.0) for record in group))
            source_mean_tvd.append(sum(float(record.get("tvd", 0.0) or 0.0) for record in group) / len(group))
            source_mean_prob_gap.append(
                sum(
                    abs(
                        _dominant_probability(record.get("sim_counts"))
                        - _dominant_probability(record.get("sampler_counts"))
                    )
                    for record in group
                )
                / len(group)
            )
            source_sampler_top_prob.append(
                sum(_dominant_probability(record.get("sampler_counts")) for record in group) / len(group)
            )
            source_support_size.append(
                sum(_support_size(record.get("sampler_counts")) for record in group) / len(group)
            )
        else:
            source_agreement.append(0.0)
            source_max_tvd.append(0.0)
            source_mean_tvd.append(0.0)
            source_mean_prob_gap.append(0.0)
            source_sampler_top_prob.append(0.0)
            source_support_size.append(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        f"{suite_title or _default_suite_title(config)}\nBackend Validation Agreement Profile",
        fontsize=14,
        fontweight="bold",
    )
    ax_a, ax_b, ax_c, ax_d2 = axes.flat

    ax_a.bar(xs, source_agreement, color="#2ca02c")
    ax_a.set_title("Top-State Agreement by Source")
    ax_a.set_ylabel("Agreement (%)")
    ax_a.set_ylim(0, 105)
    ax_a.set_xticks(xs, source_labels, rotation=15, ha="right")

    ax_b.bar(xs, source_max_tvd, color="#d62728", alpha=0.75, label="max TVD")
    ax_b.scatter(xs, source_mean_tvd, color="black", s=35, zorder=3, label="mean TVD")
    ax_b.set_title("Distribution Gap by Source")
    ax_b.set_ylabel("Total variation distance")
    ax_b.set_xticks(xs, source_labels, rotation=15, ha="right")
    ax_b.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_b.legend(fontsize=8)

    ax_c.bar(xs, source_mean_prob_gap, color="#9467bd")
    ax_c.set_title("Mean Dominant-Probability Gap")
    ax_c.set_ylabel("|p_top(sim) - p_top(sampler)|")
    ax_c.set_xticks(xs, source_labels, rotation=15, ha="right")
    ax_c.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    width = 0.36
    ax_d2.bar(xs - width / 2.0, source_sampler_top_prob, width=width, color="#1f77b4", label="mean sampler top probability")
    ax_d2.bar(xs + width / 2.0, source_support_size, width=width, color="#ff7f0e", label="mean support size")
    ax_d2.set_title("Sampling Concentration")
    ax_d2.set_ylabel("Probability / support")
    ax_d2.set_xticks(xs, source_labels, rotation=15, ha="right")
    ax_d2.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    save_figure_with_metadata(
        fig,
        agreement_png,
        {
            "figure_kind": "backend_validation_agreement_profile",
            "suite": suite_title or _default_suite_title(config),
            "shots": config.shots,
            "validation_limit": config.max_qubits,
            "noise_level": config.noise_level,
            "source_labels": list(source_labels),
        },
    )
    plt.close(fig)

    _render_entanglement_summary(
        config,
        records,
        suite_title=suite_title,
    )
    _render_state_summary(
        config,
        records,
        suite_title=suite_title,
    )

    print(f"Saved backend validation figure to: {summary_png}")
    print(f"Saved backend validation figure to: {resource_png}")
    print(f"Saved backend validation figure to: {agreement_png}")
    print(f"Saved backend validation summary to: {summary_txt}")
    return summary_png, summary_txt


def _build_noise_model(config: PublishabilityConfig) -> Any:
    if (
        config.noise_level == "ideal"
        or NoiseModel is None
        or ReadoutError is None
        or depolarizing_error is None
    ):
        return None

    noise_model = NoiseModel()
    if config.one_qubit_error > 0.0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(config.one_qubit_error, 1),
            ["u", "u1", "u2", "u3", "p", "id", "x", "sx", "rz", "ry", "rx", "h", "z"],
        )
    if config.two_qubit_error > 0.0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(config.two_qubit_error, 2),
            ["cx", "cz", "ecr", "swap"],
        )
    if config.readout_error > 0.0:
        p = min(max(config.readout_error, 0.0), 0.499999)
        noise_model.add_all_qubit_readout_error(ReadoutError([[1.0 - p, p], [p, 1.0 - p]]))
    return noise_model


class ScenarioAerPublishability:
    """Record circuits touched by a scenario and cross-check them with Aer backends."""

    def __init__(self, scenario_label: str, *, config: PublishabilityConfig) -> None:
        self.scenario_label = str(scenario_label)
        self.config = config
        self._noise_model = _build_noise_model(config)
        self._records: list[_RecordedCircuit] = []
        self._seen_ids: set[int] = set()
        self._patches: list[tuple[Any, str, Any]] = []
        self._log_lock = threading.Lock()

    def _append_log(self, payload: dict[str, Any]) -> None:
        log_path = self.config.structured_log_path
        if not log_path:
            return
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with self._log_lock:
            with open(log_path, "a", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")

    def _base_log_record(self, *, status: str, index: int | None = None, source: str | None = None) -> dict[str, Any]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": self.scenario_label,
            "status": status,
            "index": index,
            "source": source,
            "mode": self.config.mode,
            "shots": self.config.shots,
            "seed": self.config.seed,
            "max_qubits": self.config.max_qubits,
            "noise_level": self.config.noise_level,
            "one_qubit_error": self.config.one_qubit_error,
            "two_qubit_error": self.config.two_qubit_error,
            "readout_error": self.config.readout_error,
        }

    def _record(self, circuit: Any, source: str) -> None:
        if QuantumCircuit is None or not isinstance(circuit, QuantumCircuit):
            return
        circuit_id = id(circuit)
        if circuit_id in self._seen_ids:
            return
        self._seen_ids.add(circuit_id)
        self._records.append(_RecordedCircuit(source=source, circuit=circuit))

    def _wrap_transpile(self, fn: Any, source_name: str):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            circuits_arg = kwargs.get("circuits", args[0] if args else None)
            if isinstance(circuits_arg, (list, tuple)):
                for idx, circ in enumerate(circuits_arg):
                    self._record(circ, f"{source_name}:input[{idx}]")
            else:
                self._record(circuits_arg, f"{source_name}:input")

            if isinstance(out, (list, tuple)):
                for idx, circ in enumerate(out):
                    self._record(circ, f"{source_name}:output[{idx}]")
            else:
                self._record(out, f"{source_name}:output")
            return out

        return wrapped

    def _wrap_statevector(self, fn: Any):
        @functools.wraps(fn)
        def wrapped(_cls, instruction, *args, **kwargs):
            self._record(instruction, "Statevector.from_instruction")
            return fn(instruction, *args, **kwargs)

        return classmethod(wrapped)

    def patch_namespace(
        self,
        namespace: MutableMapping[str, Any],
        names: Sequence[str] = ("transpile", "qk_transpile"),
    ) -> None:
        for name in names:
            fn = namespace.get(name)
            if callable(fn):
                self._patches.append((namespace, name, fn))
                namespace[name] = self._wrap_transpile(fn, name)

    def patch_object(self, obj: Any, attr_names: Sequence[str] = ("transpile", "qk_transpile")) -> None:
        for attr_name in attr_names:
            fn = getattr(obj, attr_name, None)
            if callable(fn):
                self._patches.append((obj, attr_name, fn))
                setattr(obj, attr_name, self._wrap_transpile(fn, f"{obj.__class__.__name__}.{attr_name}"))

    def patch_qiskit_global(self) -> None:
        if qiskit is None:
            return
        for obj, attr_name in (
            (qiskit, "transpile"),
            (getattr(qiskit, "compiler", None), "transpile"),
        ):
            if obj is None:
                continue
            fn = getattr(obj, attr_name, None)
            if callable(fn):
                self._patches.append((obj, attr_name, fn))
                setattr(obj, attr_name, self._wrap_transpile(fn, f"{getattr(obj, '__name__', obj.__class__.__name__)}.{attr_name}"))

    def patch_statevector(self) -> None:
        if Statevector is None:
            return
        fn = getattr(Statevector, "from_instruction", None)
        descriptor = Statevector.__dict__.get("from_instruction")
        if callable(fn):
            self._patches.append((Statevector, "from_instruction", descriptor))
            setattr(Statevector, "from_instruction", self._wrap_statevector(fn))

    def restore(self) -> None:
        while self._patches:
            obj, attr_name, fn = self._patches.pop()
            if isinstance(obj, MutableMapping):
                obj[attr_name] = fn
            else:
                setattr(obj, attr_name, fn)

    def _suggest_heavy_validation_cli(self, required_qubits: int) -> str | None:
        if not self.config.log_dir:
            return None
        result_dir = os.path.basename(self.config.log_dir)
        if not result_dir.startswith("[RESULT]"):
            return None
        script_name = result_dir[len("[RESULT]"):] + ".py"
        script_path = os.path.join(os.path.dirname(self.config.log_dir), script_name)
        if not os.path.exists(script_path):
            return None
        return (
            f'python3 "{script_path}" {self.scenario_label} --qubits {required_qubits}'
        )

    def _suggest_entanglement_cli(self, required_qubits: int) -> str | None:
        base = self._suggest_heavy_validation_cli(required_qubits)
        if base is None:
            return None
        return f"{base} --entanglement --entanglement-qubits {required_qubits}"

    def _profile_entanglement(self, circuit: "QuantumCircuit") -> dict[str, Any]:
        if not self.config.entanglement_enabled:
            return {"entanglement_status": "disabled"}
        if EntanglementConfig is None or profile_circuit_entanglement is None:
            return {
                "entanglement_status": "monitor_unavailable",
                "entanglement_error": "entanglement_monitor_gpu import failed",
            }
        try:
            profile = profile_circuit_entanglement(
                circuit,
                config=EntanglementConfig(
                    enabled=True,
                    max_qubits=self.config.entanglement_max_qubits,
                    max_snapshots=self.config.entanglement_max_snapshots,
                    entanglement_every_step=self.config.entanglement_every_step,
                    state_enabled=self.config.state_enabled,
                    state_max_qubits=self.config.state_max_qubits,
                    state_top_k=self.config.state_top_k,
                    state_include_full=self.config.state_include_full,
                    state_every_step=self.config.state_every_step,
                ),
            )
        except Exception as exc:
            return {
                "entanglement_status": "error",
                "entanglement_error": str(exc),
            }

        payload = {f"entanglement_{key}": value for key, value in profile.items() if key != "status"}
        payload["entanglement_status"] = profile.get("status", "error")
        required_limit = profile.get("required_entanglement_limit")
        if required_limit is not None:
            payload["entanglement_suggested_cli"] = self._suggest_entanglement_cli(int(required_limit))
        return payload

    def _validate_one(self, record: _RecordedCircuit, index: int) -> None:
        base_record = self._base_log_record(status="ok", index=index, source=record.source)
        entanglement_payload: dict[str, Any] = {}
        try:
            circuit = record.circuit.copy()
            entanglement_payload = self._profile_entanglement(circuit)
            entanglement_status = str(entanglement_payload.get("entanglement_status", "disabled"))
            state_status = str(entanglement_payload.get("entanglement_state_status", "disabled"))
            ent_limit_label = "unlimited" if self.config.entanglement_max_qubits <= 0 else self.config.entanglement_max_qubits
            state_limit_label = "unlimited" if self.config.state_max_qubits <= 0 else self.config.state_max_qubits
            if not _limit_allows(self.config.max_qubits, circuit.num_qubits):
                suggested_cli = self._suggest_heavy_validation_cli(circuit.num_qubits)
                print(
                    f"  [{index}] {record.source}: backend validation deferred "
                    f"(qubits={circuit.num_qubits} > validation_limit={self.config.max_qubits})"
                )
                if entanglement_status == "ok":
                    print(
                        "       Entanglement trace recorded "
                        f"(mode={entanglement_payload.get('entanglement_simulation_mode')}, "
                        f"peak_mean={float(entanglement_payload.get('entanglement_peak_mean_single_qubit_entropy', 0.0)):.4f}, "
                        f"peak_total={float(entanglement_payload.get('entanglement_peak_total_single_qubit_entropy', 0.0)):.4f}, "
                        f"every_step={bool(entanglement_payload.get('entanglement_entanglement_every_step', False))})."
                    )
                elif entanglement_status == "skipped_too_many_qubits":
                    print(
                        "       Entanglement trace also deferred "
                        f"(qubits={circuit.num_qubits} > entanglement_limit={ent_limit_label})."
                    )
                if state_status == "ok":
                    print(
                        "       State snapshots recorded "
                        f"(top_k={int(entanglement_payload.get('entanglement_state_top_k', 0) or 0)}, "
                        f"full={bool(entanglement_payload.get('entanglement_state_include_full', False))}, "
                        f"every_step={bool(entanglement_payload.get('entanglement_state_every_step', False))})."
                    )
                elif state_status == "skipped_too_many_qubits":
                    print(
                        "       State snapshots also deferred "
                        f"(qubits={circuit.num_qubits} > state_limit={state_limit_label})."
                    )
                print("       This circuit is computationally heavy.")
                if suggested_cli:
                    print("       To force this validation, rerun the scenario from the CLI with a larger validation limit, for example:")
                    print(f"       {suggested_cli}")
                else:
                    print(
                        f"       To force this validation, rerun the scenario from the CLI with "
                        f"--qubits {circuit.num_qubits} or larger."
                    )
                self._append_log(
                    {
                        **base_record,
                        "status": "deferred_heavy_validation",
                        "qubits": circuit.num_qubits,
                        "required_validation_limit": circuit.num_qubits,
                        "suggested_cli": suggested_cli,
                        **entanglement_payload,
                    }
                )
                return

            if not _has_measurements(circuit):
                circuit.measure_all()

            if AerSimulator is None or SamplerV2 is None or transpile is None:
                print(f"  [{index}] {record.source}: qiskit-aer primitives unavailable; skipped.")
                if _AER_IMPORT_ERROR is not None:
                    print(
                        "       "
                        f"{_AER_GPU_HINT} Original error: {type(_AER_IMPORT_ERROR).__name__}: {_AER_IMPORT_ERROR}"
                    )
                self._append_log(
                    {
                        **base_record,
                        "status": "backend_unavailable",
                        "aer_import_error": str(_AER_IMPORT_ERROR) if _AER_IMPORT_ERROR is not None else None,
                        **entanglement_payload,
                    }
                )
                return

            backend_kwargs = {"seed_simulator": self.config.seed, "device": _AER_GPU_DEVICE}
            if self._noise_model is not None:
                backend_kwargs["noise_model"] = self._noise_model

            sim = AerSimulator(**backend_kwargs)
            isa_circuit = transpile(circuit, sim, optimization_level=1, seed_transpiler=self.config.seed)

            sim_counts = sim.run(isa_circuit, shots=self.config.shots).result().get_counts()

            sampler_options: dict[str, Any] = {"backend_options": {"device": _AER_GPU_DEVICE}}
            if self._noise_model is not None:
                sampler_options["backend_options"]["noise_model"] = self._noise_model

            sampler = SamplerV2(
                default_shots=self.config.shots,
                seed=self.config.seed,
                options=sampler_options,
            )
            sampler_result = sampler.run([isa_circuit]).result()[0]
            sampler_counts = _extract_sampler_counts(sampler_result)

            sim_probs = _normalize_counts(sim_counts)
            sampler_probs = _normalize_counts(sampler_counts)
            tvd = _total_variation_distance(sim_probs, sampler_probs)
            sim_top = max(sim_counts, key=sim_counts.get) if sim_counts else "n/a"
            sampler_top = max(sampler_counts, key=sampler_counts.get) if sampler_counts else "n/a"

            print(
                f"  [{index}] {record.source}: qubits={circuit.num_qubits}, depth={isa_circuit.depth()}, "
                f"sim_top={sim_top}, sampler_top={sampler_top}, tvd={tvd:.6f}"
            )
            if entanglement_status == "ok":
                print(
                    "       Entanglement trace recorded "
                    f"(mode={entanglement_payload.get('entanglement_simulation_mode')}, "
                    f"peak_mean={float(entanglement_payload.get('entanglement_peak_mean_single_qubit_entropy', 0.0)):.4f}, "
                    f"peak_total={float(entanglement_payload.get('entanglement_peak_total_single_qubit_entropy', 0.0)):.4f}, "
                    f"snapshots={int(entanglement_payload.get('entanglement_snapshots_recorded', 0))}, "
                    f"every_step={bool(entanglement_payload.get('entanglement_entanglement_every_step', False))})."
                )
            elif entanglement_status == "skipped_too_many_qubits":
                print(
                    "       Entanglement trace deferred "
                    f"(qubits={circuit.num_qubits} > entanglement_limit={ent_limit_label})."
                )
            elif entanglement_status not in {"disabled", "no_quantum_evolution", "no_snapshot_points"}:
                print(f"       Entanglement trace status: {entanglement_status}.")
            if state_status == "ok":
                print(
                    "       State snapshots recorded "
                    f"(top_k={int(entanglement_payload.get('entanglement_state_top_k', 0) or 0)}, "
                    f"full={bool(entanglement_payload.get('entanglement_state_include_full', False))}, "
                    f"every_step={bool(entanglement_payload.get('entanglement_state_every_step', False))})."
                )
            elif state_status == "skipped_too_many_qubits":
                print(
                    "       State snapshots deferred "
                    f"(qubits={circuit.num_qubits} > state_limit={state_limit_label})."
                )
            elif state_status not in {"disabled", "no_quantum_evolution", "no_snapshot_points"}:
                print(f"       State snapshot status: {state_status}.")
            self._append_log(
                {
                    **base_record,
                    "aer_device": _AER_GPU_DEVICE,
                    "aer_method": str(backend_kwargs.get("method", "automatic")),
                    "qubits": circuit.num_qubits,
                    "depth": isa_circuit.depth(),
                    "sim_top": sim_top,
                    "sampler_top": sampler_top,
                    "tvd": tvd,
                    "sim_counts": sim_counts,
                    "sampler_counts": sampler_counts,
                    **entanglement_payload,
                }
            )
        except Exception as exc:
            print(f"  [{index}] {record.source}: backend validation failed ({exc}).")
            self._append_log({**base_record, "status": "error", "error": str(exc), **entanglement_payload})

    def emit_report(self) -> None:
        print(
            f"\n[Backend Validation] Scenario {self.scenario_label}: "
            f"AerSimulator / SamplerV2 cross-validation (shots={self.config.shots}, "
            f"validation_limit={self.config.max_qubits}, noise={self.config.noise_level})"
        )
        if self.config.structured_log_path:
            print(f"  Validation log: {self.config.structured_log_path}")
        if not self._records:
            print("  No executable circuits were captured for backend validation in this scenario.")
            self._append_log(self._base_log_record(status="no_circuits"))
            return
        workers = max(1, int(self.config.validation_workers or 1))
        if workers <= 1 or len(self._records) <= 1:
            for idx, record in enumerate(self._records, start=1):
                self._validate_one(record, idx)
        else:
            print(f"  Parallel validation enabled (workers={workers}).")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._validate_one, record, idx): idx
                    for idx, record in enumerate(self._records, start=1)
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover
                        idx = futures.get(future, "?")
                        print(f"  [validation-{idx}] worker failed: {exc}")
        self._append_log(
            {
                **self._base_log_record(status="scenario_complete"),
                "captured_circuits": len(self._records),
            }
        )


def make_publishable_runner(
    label: str,
    fn: Any,
    *,
    module_globals: MutableMapping[str, Any],
    extra_patch_objects: Iterable[Any] = (),
    config: PublishabilityConfig | None = None,
    shots: int = 1024,
    seed: int = 42,
    max_qubits: int = 20,
    log_dir: str | None = None,
):
    resolved_config = config or PublishabilityConfig(
        shots=shots,
        seed=seed,
        max_qubits=max_qubits,
        log_dir=log_dir,
    )

    if not resolved_config.enabled:
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        validator = ScenarioAerPublishability(str(label), config=resolved_config)
        validator.patch_namespace(module_globals)
        validator.patch_qiskit_global()
        validator.patch_statevector()
        for obj in extra_patch_objects:
            validator.patch_object(obj)
        try:
            return fn(*args, **kwargs)
        finally:
            validator.restore()
            validator.emit_report()

    return wrapped


def wrap_scenarios(
    scenarios: Sequence[tuple[str, Any]],
    *,
    module_globals: MutableMapping[str, Any],
    extra_patch_objects: Iterable[Any] = (),
    config: PublishabilityConfig | None = None,
    shots: int = 1024,
    seed: int = 42,
    max_qubits: int = 20,
    log_dir: str | None = None,
) -> list[tuple[str, Any]]:
    resolved_config = config or PublishabilityConfig(
        shots=shots,
        seed=seed,
        max_qubits=max_qubits,
        log_dir=log_dir,
    )
    if not resolved_config.enabled:
        return list(scenarios)
    return [
        (
            label,
            make_publishable_runner(
                label,
                fn,
                module_globals=module_globals,
                extra_patch_objects=extra_patch_objects,
                config=resolved_config,
            ),
        )
        for label, fn in scenarios
    ]
