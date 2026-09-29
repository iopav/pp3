"""Gate-ZO variants of the original two-stage BDNN implementation.

Model classes are imported lazily so standard-library-only utilities such as
the result summarizer can run with a cluster's bare system Python.
"""

__all__ = ["BDNN1GateZO", "BDNN2GateZO"]


def __getattr__(name):
    if name == "BDNN1GateZO":
        from .bdnn1_gatezo import BDNN1GateZO
        return BDNN1GateZO
    if name == "BDNN2GateZO":
        from .bdnn2_gatezo import BDNN2GateZO
        return BDNN2GateZO
    raise AttributeError(name)
