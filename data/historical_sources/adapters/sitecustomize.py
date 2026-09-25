"""Reject unselected historical imports without redistributing their kernels.

Old generic benchmark workers import every algorithm before choosing a method.
These placeholders make the selected conversions runnable. They implement no
algorithm and raise if an excluded branch tries to call one.
"""

import sys
import types


class _Excluded(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(f"{self.__name__} is outside this selected archive")


for _name in ("anschuetz_tensor_network", "spencer", "georges"):
    _full = "pauli_algorithm_comparison." + _name
    sys.modules[_full] = _Excluded(_full)
