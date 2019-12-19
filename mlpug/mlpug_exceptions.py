
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
