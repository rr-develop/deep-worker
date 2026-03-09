"""Tests for deep_worker_cli.

CRITICAL: ALWAYS TEST LOCAL REPOSITORY CODE
============================================
ALL tests in this package MUST use code from:
    the deepagents directory in this repository

The conftest.py in this directory ensures sys.path is configured correctly
to FORCE local repository code to be imported FIRST, before any installed
packages from site-packages.

WHY THIS IS CRITICAL:
---------------------
- Tests MUST validate the actual code in THIS repository
- Testing installed packages is COMPLETELY UNACCEPTABLE
- Using site-packages versions leads to:
  * Testing outdated/different code
  * False test results (tests pass but production code is broken)
  * Wasted debugging time
  * Production bugs that tests don't catch

NEVER REMOVE OR BYPASS conftest.py PATH CONFIGURATION!
-------------------------------------------------------
If you think you need to disable path configuration, you are WRONG.
The correct solution is to fix the code, not the test configuration.

Test directories:
- unit_tests/    - Unit tests for individual components
- integration_ai/ - Integration tests with AI validation
- task_manager/  - Task manager specific tests
- rpyc_service/  - RPyC service tests

All directories benefit from conftest.py path configuration.
"""
