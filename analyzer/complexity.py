# analyzer/complexity.py
from dataclasses import dataclass

from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.visitors import Function


@dataclass
class ComplexityMetrics:
    cyclomatic_complexity: float   # mean across functions/methods
    max_complexity: float          # worst single function/method
    maintainability_index: float
    grade: str


def _complexity_to_grade(avg_cc: float) -> str:
    if avg_cc <= 5:   return 'A'
    if avg_cc <= 10:  return 'B'
    if avg_cc <= 15:  return 'C'
    if avg_cc <= 20:  return 'D'
    if avg_cc <= 25:  return 'E'
    return 'F'


def analyze_complexity(code: str) -> ComplexityMetrics:
    """
    Compute cyclomatic complexity and maintainability index via radon.
    Raises SyntaxError if the code is not valid Python — callers are
    expected to have parsed the code already (api/main.py runs
    extract_features first, which raises SyntaxError on bad input).
    """
    blocks = cc_visit(code)
    # cc_visit returns a flat list where a Class entry aggregates its own
    # methods, which also appear as separate Function entries. Keep only
    # Function blocks (methods included) to avoid double-counting.
    functions = [b for b in blocks if isinstance(b, Function)]

    if functions:
        avg_cc = sum(f.complexity for f in functions) / len(functions)
        max_cc = float(max(f.complexity for f in functions))
    else:
        # No functions/classes at all (empty file or top-level script);
        # radon cannot measure module-level code.
        avg_cc = 1.0
        max_cc = 1.0

    mi = mi_visit(code, multi=True)
    mi = max(0.0, min(100.0, float(mi)))

    # Grade the rounded value so the reported number and the grade
    # can never disagree at threshold boundaries (e.g. 5.004).
    avg_cc = round(avg_cc, 2)

    return ComplexityMetrics(
        cyclomatic_complexity=avg_cc,
        max_complexity=max_cc,
        maintainability_index=round(mi, 1),
        grade=_complexity_to_grade(avg_cc),
    )
