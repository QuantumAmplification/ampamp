from . import _01_1_qaoa_grover
from . import _01_grover
from . import _02_fixed_point
from . import _03_25_controlled
from . import _03_5_foqa
from . import _03_oblivious
from . import _04_distributed
from . import _05_variable_time
from . import _06_qsvt
from . import _07_unified_comparative

ALL_BUILDERS = {
    "1_grover": _01_grover.build_circuit,
    "1.1_qaoa_grover": _01_1_qaoa_grover.build_circuit,
    "2_fixed_point": _02_fixed_point.build_circuit,
    "3_oblivious": _03_oblivious.build_circuit,
    "3.25_controlled": _03_25_controlled.build_circuit,
    "3.5_foqa": _03_5_foqa.build_circuit,
    "4_distributed": _04_distributed.build_circuit,
    "5_variable_time": _05_variable_time.build_circuit,
    "6_qsvt": _06_qsvt.build_circuit,
    "7_unified_comparative": _07_unified_comparative.build_circuit,
}
