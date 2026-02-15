# ARPyC - Async RPyC Adapter

Asyncio integration layer for RPyC (Remote Python Call).

## Features

- ✅ **Asyncio Support**: Convert RPyC `AsyncResult` to `awaitable`
- ✅ **Thread-safe Callbacks**: Seamless integration between RPyC threads and asyncio event loop
- ✅ **Transparent Proxies**: `await proxy.method()` instead of verbose API
- ✅ **Heartbeat & Auto-reconnect**: Production-ready connection management
- ✅ **Real-time Streaming**: Stream data from RPyC to asyncio
- ✅ **Type-safe**: Full type hints with pyright strict mode

## Installation

```bash
# From source (development mode)
cd arpyc
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### Server

```python
import rpyc
from rpyc.utils.server import ThreadedServer

class MyService(rpyc.Service):
    def exposed_greet(self, name: str) -> str:
        return f"Hello, {name}!"

server = ThreadedServer(MyService, port=18861)
server.start()
```

### Client

```python
import asyncio
from arpyc import RpycAsyncSession

async def main():
    async with RpycAsyncSession("localhost", 18861) as session:
        # Transparent async proxy
        result = await session.root.greet("World")
        print(result)  # "Hello, World!"

asyncio.run(main())
```

## Documentation

See [DESIGN.md](DESIGN.md) for detailed architecture and API reference.

## Examples

- [Basic Usage](examples/basic/) - Simple client-server example
- [Callbacks](examples/callbacks/) - Event subscription with callbacks
- [Streaming](examples/streaming/) - Real-time data streaming

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arpyc --cov-report=html

# Type checking
pyright
```

## License

MIT
