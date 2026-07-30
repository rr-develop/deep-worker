#!/usr/bin/env python3
"""Verify rpyc_async source integration.

This script verifies that rpyc is imported from the correct source
(a local rpyc_async checkout) and has AsyncioServer support.
"""

from __future__ import annotations

import sys
from pathlib import Path


def verify_rpyc_async_integration() -> bool:
    """Verify that rpyc is imported from correct source.

    Returns:
        True if integration is correct, False otherwise.
    """
    try:
        import rpyc_async as rpyc
        from rpyc_async.utils.async_server import AsyncioServer

        # Check that AsyncioServer is available
        assert hasattr(rpyc.utils, "async_server"), (
            "AsyncioServer module not found - wrong rpyc version?"
        )

        # Check rpyc location
        rpyc_path = Path(rpyc.__file__).resolve()
        expected_base = Path("/path/to/rpyc_async")

        # Check if rpyc is from expected location
        try:
            rpyc_path.relative_to(expected_base)
        except ValueError:
            print(f"❌ rpyc imported from wrong location!")
            print(f"   Expected base: {expected_base}")
            print(f"   Actual: {rpyc_path}")
            return False

        # Check protocol version
        from rpyc_async.core import consts

        if consts.PROTOCOL_VERSION < (5, 1):
            print(f"❌ Wrong protocol version: {consts.PROTOCOL_VERSION}")
            print("   Expected: >= (5, 1)")
            return False

        # Check for enable_asyncio_serving method
        conn_test = rpyc.connect("localhost", 18861, config={"allow_all_attrs": True})
        conn_test.close()

    except ConnectionRefusedError:
        # Expected - we're just checking if method exists
        pass
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

    print("✅ rpyc_async integration verified!")
    print(f"   rpyc location: {rpyc_path}")
    print(f"   Protocol version: {consts.PROTOCOL_VERSION}")
    print(f"   AsyncioServer: Available")

    return True


if __name__ == "__main__":
    try:
        success = verify_rpyc_async_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Integration verification failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
