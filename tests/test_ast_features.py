# tests/test_ast_features.py
import pytest
from analyzer.ast_features import extract_features


def test_bad_code_has_low_naming_entropy():
    with open('tests/sample_code/bad_code.py') as f:
        code = f.read()
    features = extract_features(code)
    assert features.naming_entropy < 0.4, "Single-char names should give low entropy"


def test_good_code_has_docstrings():
    with open('tests/sample_code/good_code.py') as f:
        code = f.read()
    features = extract_features(code)
    assert features.has_docstrings is True


def test_good_code_has_no_magic_numbers():
    with open('tests/sample_code/good_code.py') as f:
        code = f.read()
    features = extract_features(code)
    assert features.num_magic_numbers == 0


def test_syntax_error_raises():
    with pytest.raises(SyntaxError):
        extract_features("def broken(:\n    pass")


def test_empty_code_returns_zero_functions():
    features = extract_features("")
    assert features.num_functions == 0
    assert features.naming_entropy == 0.0