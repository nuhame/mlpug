
class MLPugException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"{self.message} :\n\n{self.__cause__}"


class TrainerInvalidException(MLPugException):
    def __init__(self, message=None):
        err_msg = "Trainer is invalid"
        if message:
            err_msg += f" : {message}"

        super().__init__(err_msg)


class BatchNotChunkableException(MLPugException):
    def __init__(self, message=None):
        err_msg = "Given batch is not chunkable, ensure that the batch object implements the " \
                  "`__len__` and `__getitem__` methods, and that the `__getitem__` method can handle slices."
        if message:
            err_msg += f" : {message}"

        super().__init__(err_msg)


class StateInvalidException(MLPugException):
    def __init__(self, message=None):
        err_msg = "State invalid, unable to set state."
        if message:
            err_msg += f" : {message}"

        super().__init__(err_msg)


class InvalidParametersException(MLPugException):
    def __init__(self, message=None):
        err_msg = "Invalid parameter(s)."
        if message:
            err_msg += f" : {message}"

        super().__init__(err_msg)
