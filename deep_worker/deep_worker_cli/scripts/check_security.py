#!/usr/bin/env python3
"""Security checker for deep-worker-cli.

This script verifies that all network services are properly configured
to bind only to localhost (127.0.0.1) and not to external interfaces.

Usage:
    python scripts/check_security.py
"""

import re
import sys
from pathlib import Path


def check_rpyc_binding():
    """Check that RPyC server binds to localhost only."""
    errors = []

    # Check agent_process.py
    agent_process_file = Path("deep_worker_cli/process/agent_process.py")
    if agent_process_file.exists():
        content = agent_process_file.read_text()

        # Check for ThreadedServer instantiation
        if "ThreadedServer(" in content:
            # Verify hostname is set to 127.0.0.1 or SAFE_HOSTNAME
            if 'hostname="127.0.0.1"' in content or 'hostname=SAFE_HOSTNAME' in content:
                print("✅ RPyC server correctly binds to localhost (127.0.0.1)")
            else:
                errors.append("❌ RPyC server does not explicitly bind to 127.0.0.1")

            # Check for dangerous patterns
            if 'hostname="0.0.0.0"' in content:
                errors.append("❌ CRITICAL: Found binding to 0.0.0.0 (external interfaces)")
            if 'hostname=""' in content:
                errors.append("❌ CRITICAL: Found empty hostname (binds to all interfaces)")
    else:
        errors.append("❌ agent_process.py not found")

    return errors


def check_interactive_binding():
    """Check that interactive.py connects to localhost only."""
    errors = []

    interactive_file = Path("deep_worker_cli/cli/interactive.py")
    if interactive_file.exists():
        content = interactive_file.read_text()

        # Check for RpycAsyncSession connections
        if "RpycAsyncSession(" in content:
            # Verify connecting to localhost or 127.0.0.1
            if '"127.0.0.1"' in content or '"localhost"' in content:
                print("✅ Interactive app connects to localhost only")
            else:
                errors.append("❌ Interactive app may connect to non-localhost addresses")

            # Check for dangerous patterns
            external_ip_pattern = r'RpycAsyncSession\(["\'](?!127\.0\.0\.1|localhost)'
            if re.search(external_ip_pattern, content):
                errors.append("❌ CRITICAL: Found connection to non-localhost address")
    else:
        errors.append("❌ interactive.py not found")

    return errors


def check_security_documentation():
    """Check that security documentation exists."""
    errors = []

    security_md = Path("SECURITY.md")
    if security_md.exists():
        content = security_md.read_text()

        # Check for key security topics
        required_topics = [
            "localhost",
            "127.0.0.1",
            "external network",
            "critical",
        ]

        missing_topics = [topic for topic in required_topics if topic not in content.lower()]

        if missing_topics:
            errors.append(f"❌ SECURITY.md missing topics: {', '.join(missing_topics)}")
        else:
            print("✅ SECURITY.md contains required security documentation")
    else:
        errors.append("❌ SECURITY.md file not found")

    return errors


def check_test_configurations():
    """Check that tests use localhost addresses."""
    errors = []

    test_files = [
        Path("tests/integration_ai/test_rpyc_integration.py"),
        Path("tests/integration_ai/test_interactive_rpyc.py"),
    ]

    for test_file in test_files:
        if test_file.exists():
            content = test_file.read_text()

            # Check for external IP addresses (not localhost/127.0.0.1)
            external_ip_pattern = r'RpycAsyncSession\(["\'](?!127\.0\.0\.1|localhost)[\d.]+'
            if re.search(external_ip_pattern, content):
                errors.append(f"❌ {test_file}: Found non-localhost connection")

            # Check for 0.0.0.0
            if '"0.0.0.0"' in content or "'0.0.0.0'" in content:
                errors.append(f"❌ {test_file}: Found binding to 0.0.0.0")
        else:
            print(f"⚠️  Test file not found: {test_file}")

    if not errors:
        print("✅ All test files use localhost addresses")

    return errors


def main():
    """Run all security checks."""
    print("=" * 60)
    print("Deep-Worker Security Configuration Check")
    print("=" * 60)
    print()

    all_errors = []

    # Run all checks
    all_errors.extend(check_rpyc_binding())
    all_errors.extend(check_interactive_binding())
    all_errors.extend(check_security_documentation())
    all_errors.extend(check_test_configurations())

    print()
    print("=" * 60)

    if all_errors:
        print("❌ SECURITY ISSUES FOUND:")
        print()
        for error in all_errors:
            print(f"  {error}")
        print()
        print("=" * 60)
        print("CRITICAL: Fix all security issues before deploying!")
        sys.exit(1)
    else:
        print("✅ All security checks passed!")
        print()
        print("Configuration is safe for localhost-only operation.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
