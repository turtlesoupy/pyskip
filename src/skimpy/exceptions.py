class SkimpyError(RuntimeError):
    pass


class SkimpyInvariantViolation(SkimpyError):
    pass


class InvalidTensorError(SkimpyError):
    pass


class IncompatibleTensorError(SkimpyError):
    pass


class UnimplementedOperationError(SkimpyError):
    pass


class TypeConversionError(SkimpyError):
    pass
