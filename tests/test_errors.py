from qtk.errors import QtkError, QtkTypeError, QtkValueError


def test_hierarchy():
    assert issubclass(QtkValueError, QtkError)
    assert issubclass(QtkValueError, ValueError)
    assert issubclass(QtkTypeError, QtkError)
    assert issubclass(QtkTypeError, TypeError)


def test_raise_value_error():
    with pytest.raises(QtkValueError):
        raise QtkValueError("test")


def test_raise_type_error():
    with pytest.raises(QtkTypeError):
        raise QtkTypeError("test")


import pytest
