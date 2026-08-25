# analyzer/ast_features.py
import ast
from dataclasses import dataclass
from typing import List


@dataclass
class CodeFeatures:
    num_functions: int = 0
    avg_function_length: float = 0.0
    max_nesting_depth: int = 0
    num_classes: int = 0
    num_imports: int = 0
    naming_entropy: float = 0.0   # 0.0 = all single-char names, 1.0 = all meaningful names
    has_docstrings: bool = False
    num_try_except: int = 0
    num_magic_numbers: int = 0    # numeric literals not assigned to named constants


def _calc_naming_entropy(names: List[str]) -> float:
    """
    Higher entropy = more varied/meaningful names.
    Low entropy = lots of names like x, y, i, a (single chars or very short).
    Returns a float between 0.0 and 1.0.
    """
    if not names:
        return 0.0
    short_count = sum(1 for n in names if len(n) <= 2)
    return round(1.0 - (short_count / len(names)), 3)


class FeatureExtractor(ast.NodeVisitor):
    """
    Walks the AST and collects structural metrics.
    ast.NodeVisitor calls visit_<NodeType> methods automatically
    as it traverses the tree.
    """

    def __init__(self):
        self.functions = []
        self.current_depth = 0
        self.max_depth = 0
        self.variable_names = []
        self.features = CodeFeatures()

    def visit_FunctionDef(self, node):
        self.features.num_functions += 1
        if ast.get_docstring(node):
            self.features.has_docstrings = True
        self.functions.append(len(node.body))
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.features.num_classes += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.features.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.features.num_imports += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.features.num_try_except += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.variable_names.append(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and abs(node.value) > 1:
            self.features.num_magic_numbers += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_If(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1


def extract_features(code: str) -> CodeFeatures:
    """
    Main entry point. Takes a Python code string, returns CodeFeatures.
    Raises SyntaxError if the code is not valid Python.
    """
    tree = ast.parse(code)
    extractor = FeatureExtractor()
    extractor.visit(tree)
    features = extractor.features
    features.max_nesting_depth = extractor.max_depth
    features.avg_function_length = (
        sum(extractor.functions) / len(extractor.functions)
        if extractor.functions else 0.0
    )
    features.naming_entropy = _calc_naming_entropy(extractor.variable_names)
    return features