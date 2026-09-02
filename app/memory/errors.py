class MemoryValidationError(ValueError):
    pass


class MemoryNotFoundError(LookupError):
    pass


class ConsolidationBusyError(RuntimeError):
    pass