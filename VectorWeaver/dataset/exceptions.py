class SvgPrepareException(Exception):
    pass


class SvgParseException(SvgPrepareException):
    pass


class SvgNoColorException(SvgPrepareException):
    pass


class SvgUnknownElement(SvgPrepareException):
    pass


class SvgToManyPaths(SvgPrepareException):
    pass


class SvgToManySegments(SvgPrepareException):
    pass
