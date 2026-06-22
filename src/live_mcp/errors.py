"""Normalized Live MCP error taxonomy."""

PARSE_ERROR = "parse_error"
UNKNOWN_TOOL = "unknown_tool"
SCHEMA_INVALID = "schema_invalid"
ARGUMENT_INVALID = "argument_invalid"
PARALLEL_NOT_SUPPORTED = "parallel_not_supported"
PRECONDITION_FAILED = "precondition_failed"
EXECUTION_ERROR = "execution_error"
TIMEOUT = "timeout"
PERMISSION_DENIED = "permission_denied"
STATE_CONFLICT = "state_conflict"
SERVER_UNAVAILABLE = "server_unavailable"

ERROR_TYPES = {
    PARSE_ERROR,
    UNKNOWN_TOOL,
    SCHEMA_INVALID,
    ARGUMENT_INVALID,
    PARALLEL_NOT_SUPPORTED,
    PRECONDITION_FAILED,
    EXECUTION_ERROR,
    TIMEOUT,
    PERMISSION_DENIED,
    STATE_CONFLICT,
    SERVER_UNAVAILABLE,
}
