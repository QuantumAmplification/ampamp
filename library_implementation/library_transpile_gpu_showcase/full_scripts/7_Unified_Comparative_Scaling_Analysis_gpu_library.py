import importlib.util
from pathlib import Path


def _load(name):
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(name.replace('.py',''), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_scenario_a():
    files = [
        "1_Grover's_Search_Algorithm_transpile_gpu_library.py",
        "1.1_QAOA_Grover_Laboratory_gpu_library.py",
        "2_Fixed_Point_Ammplitude_Amplification_transpile_gpu_library.py",
        "3_Oblivious_Ampltude_Amplification_transpile_gpu_library.py",
        "3.25_Controlled_Quantum_Amplification_transpile_gpu_library.py",
        "3.5_FOAA_transpile_gpu_library.py",
        "4_Distributed _Quantum_Amplitude_Amplification_transpile_gpu_library.py",
        "5_Variable_Time_Amplitude_Amplification_transpile_gpu_library.py",
        "6_Quantum_Singular_Variable_Transformation_transpile_gpu_library.py",
    ]
    for f in files:
        print('\n===', f, '===')
        _load(f).run_scenario_a()


if __name__ == '__main__':
    run_scenario_a()
