from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SessionEventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):  # ty: ignore[conflicting-metaclass]  # generated .pyi: EnumTypeWrapper+int base is standard proto3 pattern
    __slots__ = ()
    SESSION_EVENT_UNKNOWN: _ClassVar[SessionEventType]
    SESSION_EVENT_ATTACHED: _ClassVar[SessionEventType]
    SESSION_EVENT_KEEPALIVE: _ClassVar[SessionEventType]
    SESSION_EVENT_SHUTDOWN: _ClassVar[SessionEventType]

class IndexKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):  # ty: ignore[conflicting-metaclass]  # generated .pyi: EnumTypeWrapper+int base is standard proto3 pattern
    __slots__ = ()
    INDEX_KIND_UNKNOWN: _ClassVar[IndexKind]
    INDEX_KIND_PROJECT: _ClassVar[IndexKind]
    INDEX_KIND_HISTORY: _ClassVar[IndexKind]
SESSION_EVENT_UNKNOWN: SessionEventType
SESSION_EVENT_ATTACHED: SessionEventType
SESSION_EVENT_KEEPALIVE: SessionEventType
SESSION_EVENT_SHUTDOWN: SessionEventType
INDEX_KIND_UNKNOWN: IndexKind
INDEX_KIND_PROJECT: IndexKind
INDEX_KIND_HISTORY: IndexKind

class ToolResult(_message.Message):
    __slots__ = ("content", "is_error", "error_message")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    content: str
    is_error: bool
    error_message: str
    def __init__(self, content: _Optional[str] = ..., is_error: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthReply(_message.Message):
    __slots__ = ("ready", "protocol_version", "service_version", "uptime_seconds", "attached_clients", "model_loaded")
    READY_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VERSION_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ATTACHED_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    protocol_version: int
    service_version: str
    uptime_seconds: int
    attached_clients: int
    model_loaded: bool
    def __init__(self, ready: bool = ..., protocol_version: _Optional[int] = ..., service_version: _Optional[str] = ..., uptime_seconds: _Optional[int] = ..., attached_clients: _Optional[int] = ..., model_loaded: bool = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusReply(_message.Message):
    __slots__ = ("pid", "transport_address", "uptime_seconds", "attached_clients", "model_loaded", "embedding_model", "searches_total", "index_ops_total", "last_activity_unix", "warm_tenants")
    PID_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ATTACHED_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    SEARCHES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    INDEX_OPS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVITY_UNIX_FIELD_NUMBER: _ClassVar[int]
    WARM_TENANTS_FIELD_NUMBER: _ClassVar[int]
    pid: int
    transport_address: str
    uptime_seconds: int
    attached_clients: int
    model_loaded: bool
    embedding_model: str
    searches_total: int
    index_ops_total: int
    last_activity_unix: int
    warm_tenants: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, pid: _Optional[int] = ..., transport_address: _Optional[str] = ..., uptime_seconds: _Optional[int] = ..., attached_clients: _Optional[int] = ..., model_loaded: bool = ..., embedding_model: _Optional[str] = ..., searches_total: _Optional[int] = ..., index_ops_total: _Optional[int] = ..., last_activity_unix: _Optional[int] = ..., warm_tenants: _Optional[_Iterable[str]] = ...) -> None: ...

class AttachRequest(_message.Message):
    __slots__ = ("client_id",)
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    def __init__(self, client_id: _Optional[str] = ...) -> None: ...

class SessionEvent(_message.Message):
    __slots__ = ("event_type", "message", "timestamp_ns")
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    event_type: SessionEventType
    message: str
    timestamp_ns: int
    def __init__(self, event_type: _Optional[_Union[SessionEventType, str]] = ..., message: _Optional[str] = ..., timestamp_ns: _Optional[int] = ...) -> None: ...

class ProjectSearchRequest(_message.Message):
    __slots__ = ("repo_path", "query", "top_k")
    REPO_PATH_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    repo_path: str
    query: str
    top_k: int
    def __init__(self, repo_path: _Optional[str] = ..., query: _Optional[str] = ..., top_k: _Optional[int] = ...) -> None: ...

class ProjectSearchReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class HistorySearchRequest(_message.Message):
    __slots__ = ("summaries_dir", "query", "top_k")
    SUMMARIES_DIR_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    summaries_dir: str
    query: str
    top_k: int
    def __init__(self, summaries_dir: _Optional[str] = ..., query: _Optional[str] = ..., top_k: _Optional[int] = ...) -> None: ...

class HistorySearchReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class IndexHistoryRequest(_message.Message):
    __slots__ = ("summaries_dir", "clear_first")
    SUMMARIES_DIR_FIELD_NUMBER: _ClassVar[int]
    CLEAR_FIRST_FIELD_NUMBER: _ClassVar[int]
    summaries_dir: str
    clear_first: bool
    def __init__(self, summaries_dir: _Optional[str] = ..., clear_first: bool = ...) -> None: ...

class IndexHistoryReply(_message.Message):
    __slots__ = ("result", "files_indexed", "chunks_added", "lines_read", "errors")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    FILES_INDEXED_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_ADDED_FIELD_NUMBER: _ClassVar[int]
    LINES_READ_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    files_indexed: int
    chunks_added: int
    lines_read: int
    errors: int
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ..., files_indexed: _Optional[int] = ..., chunks_added: _Optional[int] = ..., lines_read: _Optional[int] = ..., errors: _Optional[int] = ...) -> None: ...

class IndexStatusRequest(_message.Message):
    __slots__ = ("kind", "path")
    KIND_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    kind: IndexKind
    path: str
    def __init__(self, kind: _Optional[_Union[IndexKind, str]] = ..., path: _Optional[str] = ...) -> None: ...

class IndexStatusReply(_message.Message):
    __slots__ = ("exists", "complete", "pending_lines", "problem")
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    PENDING_LINES_FIELD_NUMBER: _ClassVar[int]
    PROBLEM_FIELD_NUMBER: _ClassVar[int]
    exists: bool
    complete: bool
    pending_lines: int
    problem: str
    def __init__(self, exists: bool = ..., complete: bool = ..., pending_lines: _Optional[int] = ..., problem: _Optional[str] = ...) -> None: ...

class ListTenantsRequest(_message.Message):
    __slots__ = ("kind", "limit")
    KIND_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    kind: IndexKind
    limit: int
    def __init__(self, kind: _Optional[_Union[IndexKind, str]] = ..., limit: _Optional[int] = ...) -> None: ...

class ListTenantsReply(_message.Message):
    __slots__ = ("tenants", "total")
    TENANTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    tenants: _containers.RepeatedScalarFieldContainer[str]
    total: int
    def __init__(self, tenants: _Optional[_Iterable[str]] = ..., total: _Optional[int] = ...) -> None: ...
