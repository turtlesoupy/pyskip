class PySkipError(RuntimeError):
    pass


class PySkipInvariantViolation(PySkipError):
    pass


class InvalidTensorError(PySkipError):
    pass


class IncompatibleTensorError(PySkipError):
    pass


class UnimplementedOperationError(PySkipError):
    pass


class TypeConversionError(PySkipError):
    pass
