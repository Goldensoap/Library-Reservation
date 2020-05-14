class MyError(Exception):
    pass

class LoginError(MyError):
    r"""Exception raised for errors in the input.

    Attributes:
        location -- the error loacation
        detial -- error detial
    """
    def __init__(self, location, detial):
        super().__init__(location, detial)
        self.location = location
        self.detial = detial
