import os
import sys
import numpy as np
from qiskit import QuantumCircuit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ampamp.oblivious import ObliviousEngine
from _shared_gpu_library import Logger, run_interactive_scenario_repl, transpile_for_hardware

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.join(_HERE, f"[RESULT]{os.path.splitext(os.path.basename(__file__))[0]}")
os.makedirs(_RESULT_DIR, exist_ok=True)


def _build(iterations=8):
    eng = ObliviousEngine(2,1,0.6)
    block = eng.construct_block_encoding(np.eye(4, dtype=complex))
    refl = eng.get_reflections()
    qc = QuantumCircuit(3)
    for _ in range(int(iterations)):
        qc.compose(block, inplace=True)
        qc.compose(refl, inplace=True)
        qc.compose(block.inverse(), inplace=True)
        qc.compose(refl, inplace=True)
    return qc


def run_scenario_a(iterations=8):
    qc = _build(iterations)
    _, d, ops = transpile_for_hardware(qc, None, ['cx','id','rz','sx','x'], 3)
    print(f"OAA A iter={iterations} depth={d} cx={int(ops.get('cx',0))}")


def run_scenario_b(iterations=8):
    qc = _build(iterations)
    linear = [[0,1],[1,0],[1,2],[2,1]]
    _, d, ops = transpile_for_hardware(qc, linear, ['cx','id','rz','sx','x'], 3)
    print(f"OAA B linear iter={iterations} depth={d} cx={int(ops.get('cx',0))}")


def run_scenario_p(iterations=12):
    qc = _build(iterations)
    _, d0, _ = transpile_for_hardware(qc, None, ['cx','id','rz','sx','x'], 0)
    _, d3, _ = transpile_for_hardware(qc, None, ['cx','id','rz','sx','x'], 3)
    print(f"OAA P optimize depth0={d0} depth3={d3}")


if __name__ == '__main__':
    output_filepath = os.path.join(_RESULT_DIR, "terminal_output.log")
    logger = Logger(output_filepath)
    sys.stdout = logger
    scenarios = [("A", lambda: run_scenario_a(8)), ("B", lambda: run_scenario_b(8)), ("P", lambda: run_scenario_p(12))]
    interactive = [("A", run_scenario_a), ("B", run_scenario_b), ("P", run_scenario_p)]
    try:
        for _, fn in scenarios:
            fn()
        run_interactive_scenario_repl(interactive, sep="="*70)
    finally:
        logger.log.close()
        sys.stdout = logger.terminal
        print("OAA library GPU suite complete.")
