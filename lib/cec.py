from opfunu.cec_based import CecBenchmark, cec2017


def get_cec2017_for_dim(idx: int, dim: int) -> CecBenchmark:
    """Get an opfunu.CecBenchmark corresponding to the given function in dim dimensions"""
    if idx < 1 or idx > 29:
        raise ValueError("invalid idx for cec2017 fun")

    fname = f"F{idx}2017"
    return getattr(cec2017, fname)(dim)


def cec_range(start: int, end: int, dim: int):
    return (get_cec2017_for_dim(idx, dim) for idx in range(start, end))
