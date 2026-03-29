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

ALL_ALGORITHM_RUNNERS = {
    "1_grover": _01_grover.run,
    "1.1_qaoa_grover": _01_1_qaoa_grover.run,
    "2_fixed_point": _02_fixed_point.run,
    "3_oblivious": _03_oblivious.run,
    "3.25_controlled": _03_25_controlled.run,
    "3.5_foqa": _03_5_foqa.run,
    "4_distributed": _04_distributed.run,
    "5_variable_time": _05_variable_time.run,
    "6_qsvt": _06_qsvt.run,
    "7_unified_comparative": _07_unified_comparative.run,
}
