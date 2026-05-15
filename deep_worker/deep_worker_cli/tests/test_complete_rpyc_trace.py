"""Complete RPyC protocol message trace for ONE get_status() call.

This test captures BOTH send and receive messages by patching protocol layer.
"""

import asyncio
from typing import List, Dict, Any
import time

import pytest


class CompleteRPyCTracer:
    """Captures ALL RPyC protocol messages (send + receive)."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.original_send = None
        self.original_dispatch = None

    def start_tracing(self, connection):
        """Start capturing all protocol messages."""
        self.messages = []

        # Patch _send for outgoing messages
        self.original_send = connection._send

        def traced_send(msg_type, seq, data):
            self.messages.append({
                'timestamp': time.time(),
                'direction': 'SEND',
                'msg_type': msg_type,
                'seq': seq,
                'data_repr': repr(data)[:300]
            })
            return self.original_send(msg_type, seq, data)

        connection._send = traced_send

        # Patch _dispatch for incoming messages
        # This is called for EVERY incoming message
        self.original_dispatch = connection._dispatch

        def traced_dispatch(data):
            # Parse message to log it
            # RPyC message format: msg_type(1 byte) + pickled(seq, args)
            try:
                import brine
                from rpyc.core import consts
                msg_type, = brine.I1.unpack(data[:1])
                seq, args = brine.load(data[1:])

                self.messages.append({
                    'timestamp': time.time(),
                    'direction': 'RECV',
                    'msg_type': msg_type,
                    'seq': seq,
                    'data_repr': repr(args)[:300]
                })
            except:
                # If parsing fails, just log raw
                self.messages.append({
                    'timestamp': time.time(),
                    'direction': 'RECV',
                    'msg_type': '?',
                    'seq': '?',
                    'data_repr': repr(data)[:300]
                })

            return self.original_dispatch(data)

        connection._dispatch = traced_dispatch

    def stop_tracing(self, connection):
        """Stop tracing and restore original methods."""
        if self.original_send:
            connection._send = self.original_send
        if self.original_dispatch:
            connection._dispatch = self.original_dispatch

    def get_analysis(self) -> Dict[str, Any]:
        """Analyze captured messages."""
        if not self.messages:
            return {
                'total': 0,
                'send': 0,
                'recv': 0,
                'messages': []
            }

        total = len(self.messages)
        sends = sum(1 for m in self.messages if m['direction'] == 'SEND')
        recvs = sum(1 for m in self.messages if m['direction'] == 'RECV')

        # Calculate duration
        if len(self.messages) >= 2:
            start_time = self.messages[0]['timestamp']
            end_time = self.messages[-1]['timestamp']
            duration = end_time - start_time
        else:
            duration = 0

        # Count by message type
        type_counts = {}
        for msg in self.messages:
            key = f"{msg['direction']}:{msg['msg_type']}"
            type_counts[key] = type_counts.get(key, 0) + 1

        # Look for patterns in data
        data_patterns = {}
        for msg in self.messages:
            data = msg['data_repr']
            # Extract key patterns
            if 'dict' in data and 'get' in data:
                data_patterns['dict.get'] = data_patterns.get('dict.get', 0) + 1
            elif 'method' in data:
                data_patterns['method_call'] = data_patterns.get('method_call', 0) + 1
            elif 'builtin' in data:
                data_patterns['builtin'] = data_patterns.get('builtin', 0) + 1

        return {
            'total': total,
            'send': sends,
            'recv': recvs,
            'duration': duration,
            'type_counts': type_counts,
            'data_patterns': data_patterns,
            'messages': self.messages
        }


@pytest.mark.skip(
    reason=(
        "Integration trace — requires a live deep-worker agent named "
        "'test_dw' already running and registered at the moment the "
        "test executes. This is a diagnostic/perf harness, not a unit "
        "test. Run manually against a real agent when you need the "
        "RPyC message trace. The message-count optimization it was "
        "originally written to verify is captured by the automated "
        "unit tests in `tests/test_obtain_netref_fix.py` (which mock "
        "the RPyC connection)."
    )
)
@pytest.mark.asyncio
async def test_complete_rpyc_trace_single_get_status():
    """Capture COMPLETE RPyC trace (SEND + RECV) for ONE get_status() call."""

    agent_name = "test_dw"

    print(f"\n{'='*80}")
    print(f"COMPLETE RPyC PROTOCOL TRACE - ONE get_status() CALL")
    print(f"{'='*80}\n")

    from deep_worker_cli.client.agent_client import AgentClient

    # Create client and connect
    client = AgentClient(agent_name)
    await client.connect()
    print("✅ Client connected to agent")

    # Get RPyC connection
    rpyc_conn = client._conn
    print(f"✅ Got RPyC connection: {rpyc_conn}\n")

    # Create tracer
    tracer = CompleteRPyCTracer()

    # Start tracing
    print("🔍 Starting COMPLETE protocol trace (SEND + RECV)...")
    tracer.start_tracing(rpyc_conn)

    # Make ONE get_status() call
    print("📞 Making SINGLE get_status() call...\n")
    start_time = time.time()
    status = await client.get_status()
    duration = time.time() - start_time

    print(f"✅ get_status() completed in {duration:.3f}s")
    print(f"   Status: {status.status}\n")

    # Stop tracing
    tracer.stop_tracing(rpyc_conn)
    print("✅ Trace complete\n")

    # Disconnect
    await client.disconnect()

    # Analyze
    analysis = tracer.get_analysis()

    print(f"{'='*80}")
    print(f"COMPLETE RPyC PROTOCOL MESSAGE COUNT")
    print(f"{'='*80}\n")

    print(f"📊 Total RPyC protocol messages: {analysis['total']}")
    print(f"   - SEND (client → agent): {analysis['send']}")
    print(f"   - RECV (agent → client): {analysis['recv']}")
    print(f"   - Duration: {analysis['duration']:.3f}s\n")

    print(f"📋 Messages by type:")
    for msg_type, count in sorted(analysis['type_counts'].items(),
                                   key=lambda x: x[1], reverse=True):
        print(f"   {msg_type}: {count}")

    print(f"\n📋 Data patterns detected:")
    for pattern, count in sorted(analysis['data_patterns'].items(),
                                  key=lambda x: x[1], reverse=True):
        print(f"   {pattern}: {count}")

    # Print all messages for detailed analysis
    print(f"\n📄 Complete message sequence:")
    for i, msg in enumerate(analysis['messages'], 1):
        elapsed = msg['timestamp'] - analysis['messages'][0]['timestamp']
        # Handle msg_type and seq that might be strings ('?') or ints
        msg_type_str = str(msg['msg_type']).rjust(3)
        seq_str = str(msg['seq']).rjust(4)
        print(f"   {i:3d}. [{msg['direction']:4s}] type={msg_type_str}, seq={seq_str}, t={elapsed:.4f}s")
        print(f"        {msg['data_repr'][:150]}")

    print(f"\n{'='*80}")
    print(f"💡 ANALYSIS")
    print(f"{'='*80}\n")

    total = analysis['total']

    # Expected for efficient RPC:
    # 1 SEND: call get_status
    # 1 RECV: response
    # Total: 2 messages

    expected_efficient = 2
    overhead_ratio = total / expected_efficient if expected_efficient else 0

    print(f"Expected (efficient RPC): {expected_efficient} messages")
    print(f"Actual (RPyC):           {total} messages")
    print(f"Overhead:                {overhead_ratio:.1f}x\n")

    if total < 10:
        print(f"✅ EXCELLENT: Only {total} messages - very efficient")
    elif total < 50:
        print(f"✅ GOOD: {total} messages - acceptable overhead")
    elif total < 200:
        print(f"⚠️  MODERATE: {total} messages - noticeable overhead")
    elif total < 1000:
        print(f"❌ PROBLEM: {total} messages - SIGNIFICANT overhead!")
    else:
        print(f"❌ CRITICAL: {total} messages - EXTREME overhead!")
        print(f"   This explains the high CPU usage!")

    # Check for dict.get pattern (netref access)
    if 'dict.get' in analysis['data_patterns']:
        dict_get_count = analysis['data_patterns']['dict.get']
        print(f"\n⚠️  Detected {dict_get_count} dict.get() operations")
        print(f"   This indicates netref proxy overhead - each field access is a separate RPC call")

    print(f"\n{'='*80}\n")

    # Assert basic correctness
    assert status.status in ["ready", "working", "error"], f"Unexpected status: {status.status}"
    assert total > 0, "Should have captured some messages"

    # TDD: After JSON serialization fix, expect EXACTLY 2 messages (1 SEND + 1 RECV)
    # This is the target after eliminating netref overhead
    assert total == 2, (
        f"Expected EXACTLY 2 messages after JSON serialization fix "
        f"(1 SEND request + 1 RECV response), but got {total}. "
        f"This test enforces the performance target from RPYC_SERIALIZATION_DESIGN.md"
    )

    # After fix, there should be NO dict.get() operations (no netref access)
    assert 'dict.get' not in analysis['data_patterns'], (
        f"Detected dict.get() operations indicating netref proxy overhead. "
        f"After JSON serialization fix, response should be a plain string, not netref."
    )

    # Save detailed trace
    trace_file = "/tmp/complete_rpyc_trace.txt"
    with open(trace_file, 'w') as f:
        f.write(f"Complete RPyC trace for ONE get_status() call\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total messages: {total} (SEND: {analysis['send']}, RECV: {analysis['recv']})\n")
        f.write(f"Duration: {analysis['duration']:.3f}s\n\n")
        f.write(f"Message sequence:\n\n")
        for i, msg in enumerate(analysis['messages'], 1):
            elapsed = msg['timestamp'] - analysis['messages'][0]['timestamp']
            msg_type_str = str(msg['msg_type']).rjust(3)
            seq_str = str(msg['seq']).rjust(4)
            f.write(f"{i:3d}. [{msg['direction']:4s}] type={msg_type_str}, seq={seq_str}, t={elapsed:.6f}s\n")
            f.write(f"     {msg['data_repr']}\n\n")

    print(f"📁 Complete trace saved to: {trace_file}\n")

    return analysis


if __name__ == "__main__":
    result = asyncio.run(test_complete_rpyc_trace_single_get_status())
    print(f"\n✅ Test completed")
    print(f"   Total messages: {result['total']}")
    print(f"   SEND: {result['send']}, RECV: {result['recv']}")
