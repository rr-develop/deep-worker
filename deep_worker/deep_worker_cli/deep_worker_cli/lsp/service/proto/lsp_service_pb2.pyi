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
    SESSION_EVENT_REFRESH_STARTED: _ClassVar[SessionEventType]
    SESSION_EVENT_REFRESH_FINISHED: _ClassVar[SessionEventType]
    SESSION_EVENT_SHUTDOWN: _ClassVar[SessionEventType]
    SESSION_EVENT_DIAGNOSTICS_UPDATED: _ClassVar[SessionEventType]
SESSION_EVENT_UNKNOWN: SessionEventType
SESSION_EVENT_ATTACHED: SessionEventType
SESSION_EVENT_KEEPALIVE: SessionEventType
SESSION_EVENT_REFRESH_STARTED: SessionEventType
SESSION_EVENT_REFRESH_FINISHED: SessionEventType
SESSION_EVENT_SHUTDOWN: SessionEventType
SESSION_EVENT_DIAGNOSTICS_UPDATED: SessionEventType

class Position(_message.Message):
    __slots__ = ("line", "character")
    LINE_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_FIELD_NUMBER: _ClassVar[int]
    line: int
    character: int
    def __init__(self, line: _Optional[int] = ..., character: _Optional[int] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: Position
    end: Position
    def __init__(self, start: _Optional[_Union[Position, _Mapping]] = ..., end: _Optional[_Union[Position, _Mapping]] = ...) -> None: ...

class Location(_message.Message):
    __slots__ = ("uri", "range")
    URI_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    uri: str
    range: Range
    def __init__(self, uri: _Optional[str] = ..., range: _Optional[_Union[Range, _Mapping]] = ...) -> None: ...

class Symbol(_message.Message):
    __slots__ = ("name", "kind", "location", "container", "detail")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: int
    location: Location
    container: str
    detail: str
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[int] = ..., location: _Optional[_Union[Location, _Mapping]] = ..., container: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class ToolResult(_message.Message):
    __slots__ = ("content_json", "is_error", "error_message")
    CONTENT_JSON_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    content_json: str
    is_error: bool
    error_message: str
    def __init__(self, content_json: _Optional[str] = ..., is_error: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthReply(_message.Message):
    __slots__ = ("alive", "service_version", "serena_version", "project_root", "uptime_seconds", "attached_clients")
    ALIVE_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERENA_VERSION_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ATTACHED_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    alive: bool
    service_version: str
    serena_version: str
    project_root: str
    uptime_seconds: int
    attached_clients: int
    def __init__(self, alive: bool = ..., service_version: _Optional[str] = ..., serena_version: _Optional[str] = ..., project_root: _Optional[str] = ..., uptime_seconds: _Optional[int] = ..., attached_clients: _Optional[int] = ...) -> None: ...

class AttachRequest(_message.Message):
    __slots__ = ("client_id", "project_root")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    project_root: str
    def __init__(self, client_id: _Optional[str] = ..., project_root: _Optional[str] = ...) -> None: ...

class SessionEvent(_message.Message):
    __slots__ = ("event_type", "message", "uri", "timestamp_ns")
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    event_type: SessionEventType
    message: str
    uri: str
    timestamp_ns: int
    def __init__(self, event_type: _Optional[_Union[SessionEventType, str]] = ..., message: _Optional[str] = ..., uri: _Optional[str] = ..., timestamp_ns: _Optional[int] = ...) -> None: ...

class RefreshRequest(_message.Message):
    __slots__ = ("force",)
    FORCE_FIELD_NUMBER: _ClassVar[int]
    force: bool
    def __init__(self, force: bool = ...) -> None: ...

class RefreshReply(_message.Message):
    __slots__ = ("triggered", "reason")
    TRIGGERED_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    triggered: bool
    reason: str
    def __init__(self, triggered: bool = ..., reason: _Optional[str] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusReply(_message.Message):
    __slots__ = ("project_root", "slug", "transport", "transport_address", "pid", "uptime_seconds", "attached_clients", "last_refresh_unix", "last_refresh_reason", "active_providers", "watched_paths")
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    SLUG_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ATTACHED_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    LAST_REFRESH_UNIX_FIELD_NUMBER: _ClassVar[int]
    LAST_REFRESH_REASON_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_PROVIDERS_FIELD_NUMBER: _ClassVar[int]
    WATCHED_PATHS_FIELD_NUMBER: _ClassVar[int]
    project_root: str
    slug: str
    transport: str
    transport_address: str
    pid: int
    uptime_seconds: int
    attached_clients: int
    last_refresh_unix: int
    last_refresh_reason: str
    active_providers: _containers.RepeatedScalarFieldContainer[str]
    watched_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, project_root: _Optional[str] = ..., slug: _Optional[str] = ..., transport: _Optional[str] = ..., transport_address: _Optional[str] = ..., pid: _Optional[int] = ..., uptime_seconds: _Optional[int] = ..., attached_clients: _Optional[int] = ..., last_refresh_unix: _Optional[int] = ..., last_refresh_reason: _Optional[str] = ..., active_providers: _Optional[_Iterable[str]] = ..., watched_paths: _Optional[_Iterable[str]] = ...) -> None: ...

class FindSymbolRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class FindSymbolReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class RefRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class RefReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class DeclRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class DeclReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class ImplRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class ImplReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class OverviewRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class OverviewReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class DiagRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class DiagReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class RenameRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class RenameReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class ReplaceRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class ReplaceReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class InsertRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class InsertReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("args_json",)
    ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    args_json: str
    def __init__(self, args_json: _Optional[str] = ...) -> None: ...

class DeleteReply(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ToolResult
    def __init__(self, result: _Optional[_Union[ToolResult, _Mapping]] = ...) -> None: ...
