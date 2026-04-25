class PlatformError(Exception):
    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ConfigurationError(PlatformError):
    pass


class ToolExecutionError(PlatformError):
    pass


class ToolNotFoundError(PlatformError):
    pass


class PolicyViolationError(PlatformError):
    def __init__(self, message: str, tool_name: str, environment: str):
        self.tool_name = tool_name
        self.environment = environment
        super().__init__(message, code="POLICY_VIOLATION")


class RadarBlockedError(PlatformError):
    def __init__(self, message: str, tool_name: str, status: str):
        self.tool_name = tool_name
        self.status = status
        super().__init__(message, code="RADAR_BLOCKED")


class KairosError(PlatformError):
    pass


class RAGRetrievalError(PlatformError):
    pass
