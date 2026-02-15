# Deep Worker Integration Tests

This document describes all integration tests for the `dwcli`, `deep-worker` and `deepagents` components in the project.

## Contents

- [1. Terminal Emulator Tests (AI)](#1-terminal-emulator-tests-ai)
  - [1.1 TerminalEmulator Unit Tests](#11-terminalemulator-unit-tests)
  - [1.2 DwcliTerminalSession Unit Tests](#12-dwcliterminalsession-unit-tests)
  - [1.3 dwcli Integration Tests](#13-dwcli-integration-tests)
  - [1.4 deepagents Integration Tests](#14-deepagents-integration-tests)
- [2. Agent Tests (AI Validation)](#2-agent-tests-ai-validation)
- [3. Sandbox Operations Tests](#3-sandbox-operations-tests)
  - [3.1 Write Operations](#31-write-operations)
  - [3.2 Read Operations](#32-read-operations)
  - [3.3 Edit Operations](#33-edit-operations)
  - [3.4 LS Info Operations](#34-ls-info-operations)
  - [3.5 Grep Operations](#35-grep-operations)
  - [3.6 Glob Operations](#36-glob-operations)
  - [3.7 Complex Scenarios](#37-complex-scenarios)
- [4. Sandbox Factory Tests](#4-sandbox-factory-tests)
- [5. IPC Integration Tests](#5-ipc-integration-tests)

---

## 1. Terminal Emulator Tests (AI)

**File**: `deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py`

These tests verify the terminal emulator used for automated testing of the `dwcli` and `deepagents` TUI interfaces.

### 1.1 TerminalEmulator Unit Tests

Basic terminal emulator tests that do not launch real applications:

| Test | Description |
|------|----------|
| `test_terminal_emulator_init` | Verifies terminal emulator initialization |
| `test_terminal_emulator_feed_simple_text` | Handling plain text without escape sequences |
| `test_terminal_emulator_feed_with_newlines` | Handling text with line breaks |
| `test_terminal_emulator_ansi_colors` | Handling ANSI escape sequences for colors |
| `test_terminal_emulator_cursor_movement` | Handling cursor movement escape sequences |
| `test_terminal_emulator_get_line` | Retrieving a specific line from the terminal |
| `test_terminal_emulator_find_text` | Searching for text on the terminal screen |
| `test_terminal_emulator_find_text_case_insensitive` | Case-insensitive text search |
| `test_terminal_emulator_get_cursor_position` | Retrieving the cursor position |
| `test_terminal_emulator_clear` | Clearing the terminal screen |
| `test_terminal_emulator_resize` | Resizing the terminal |
| `test_terminal_emulator_large_output` | Handling large volumes of output |
| `test_terminal_emulator_rapid_updates` | Handling rapid screen updates |
| `test_terminal_emulator_unicode` | Handling Unicode characters |

### 1.2 DwcliTerminalSession Unit Tests

Tests of the `DwcliTerminalSession` class for working with dwcli through a PTY:

| Test | Description |
|------|----------|
| `test_create_terminal_snapshot_basic` | Creating a terminal snapshot without the cursor |
| `test_create_terminal_snapshot_with_title` | Creating a terminal snapshot with a title |
| `test_create_terminal_snapshot_with_cursor` | Creating a terminal snapshot with the cursor position |
| `test_dwcli_terminal_session_init` | Initializing a DwcliTerminalSession session |
| `test_dwcli_terminal_session_send_key_validation` | Validating key names in send_key() |
| `test_dwcli_terminal_session_start_stop` | Starting and stopping a dwcli terminal session |
| `test_dwcli_terminal_session_context_manager` | Using DwcliTerminalSession as a context manager |
| `test_dwcli_terminal_session_send_input` | Sending text input to dwcli |
| `test_dwcli_terminal_session_send_keys` | Sending special keys to dwcli |
| `test_dwcli_terminal_session_wait_for_text` | Waiting for specific text to appear |
| `test_dwcli_terminal_session_wait_for_prompt` | Waiting for the prompt to appear |
| `test_dwcli_terminal_session_get_cursor_position` | Retrieving the cursor position from a running session |
| `test_dwcli_terminal_session_send_input_not_running` | Verifies that send_input() raises an exception if the session is not running |
| `test_dwcli_terminal_session_send_key_not_running` | Verifies that send_key() raises an exception if the session is not running |
| `test_dwcli_terminal_session_double_start` | Verifies that starting again causes no problems |
| `test_dwcli_terminal_session_close_not_started` | Verifies that close() is safe for a session that was not started |

### 1.3 dwcli Integration Tests

Integration tests with real dwcli and agents:

| Test | Description | Criticality |
|------|----------|-------------|
| `test_dwcli_query_visibility` | **🔒 SYNCHRONIZED**: Verifies the visibility of queries and responses in the dwcli terminal. Synchronized with `test_deepagents_query_does_not_disappear`. Uses AI validation to verify that the entered query and the agent's response remain visible on screen. | **CRITICAL** |
| `test_dwcli_reconnect_preserves_history` | Verifies that dwcli shows the conversation history when reconnecting to an agent. The test creates an agent, performs 3 queries, closes dwcli, opens it again and verifies that the entire history is visible. | **CRITICAL** |
| `test_dwcli_input_visible_with_long_history` | Verifies that the input area is visible when connecting to an agent with a long history. The test creates 40 queries in the first session, then opens a second session and verifies the visibility of the input area without user interaction. | **CRITICAL** |
| `test_dwcli_scroll_pagination` | Verifies that dwcli loads additional pages of history when scrolling up. The test checks pagination functionality: it creates a long history, scrolls up and verifies that older messages are loaded. | **CRITICAL** |
| `test_dwcli_input_area_immediately_visible` | Verifies that the input area is IMMEDIATELY visible when opening the TUI with history. Uses AI vision analysis to verify that the input area (with the `>` character) is visible right after opening, without user interaction. | **CRITICAL** |

### 1.4 deepagents Integration Tests

| Test | Description | Criticality |
|------|----------|-------------|
| `test_deepagents_query_does_not_disappear` | **🔒 SYNCHRONIZED**: Verifies the visibility of queries and responses in the deepagents terminal. Synchronized with `test_dwcli_query_visibility`. Uses AI validation to verify that the entered query and the agent's response remain visible on screen. | **CRITICAL** |
| `test_deepagents_multiple_queries` | Verifies several consecutive queries in deepagents. Checks that multiple query-response cycles work correctly. | HIGH |

---

## 2. Agent Tests (AI Validation)

**File**: `deep_worker/deep_worker_cli/tests/integration_ai/test_agent_basic_example.py`

These tests demonstrate the use of AI validation for checking agent responses instead of exact string comparison.

| Test | Description |
|------|----------|
| `test_simple_arithmetic` | Verifies that the agent can perform simple arithmetic. Demonstrates basic AI validation of a response. |
| `test_factual_question` | Verifies that the agent can answer factual questions. |
| `test_with_context` | Verifies validation with additional context. |
| `test_batch_validation_example` | Batch validation example (skipped). Demonstrates how to validate several responses at once. |
| `test_model_fallback_example` | Example of switching to a better model when validation fails. Demonstrates conditional model selection based on the initial result. |
| `test_check_reply_correctness` | Verifies that check_reply correctly identifies valid/invalid responses. Tests the validation infrastructure itself. |
| `test_check_reply_variations` | Verifies that validation handles various response formats. Demonstrates the robustness of AI validation vs. string comparison. |

---

## 3. Sandbox Operations Tests

**File**: `deep_worker/deep_worker_cli/tests/integration_tests/test_sandbox_operations.py`

Tests verifying the core file operations in BaseSandbox. All tests run on a single sandbox instance (class-scoped fixture) to avoid the overhead of creating many containers.

### 3.1 Write Operations

Tests of the `write()` method for creating new files:

| Test | Description |
|------|----------|
| `test_write_new_file` | Creating a new file with basic content |
| `test_write_creates_parent_dirs` | write() automatically creates parent directories |
| `test_write_existing_file_fails` | Writing to an existing file returns an error |
| `test_write_special_characters` | Writing content with special characters and escape sequences |
| `test_write_empty_file` | Creating an empty file |
| `test_write_path_with_spaces` | Creating a file with spaces in the path |
| `test_write_unicode_content` | Writing Unicode content |
| `test_write_consecutive_slashes_in_path` | Handling consecutive slashes in the path |
| `test_write_very_long_content` | Writing very long content |
| `test_write_content_with_only_newlines` | Writing content consisting only of line breaks |

### 3.2 Read Operations

Tests of the `read()` method for reading files with line numbering:

| Test | Description |
|------|----------|
| `test_read_basic_file` | Reading a basic file |
| `test_read_nonexistent_file` | Reading a nonexistent file |
| `test_read_empty_file` | Reading an empty file |
| `test_read_with_offset` | Reading a file with an offset (starting from a specific line) |
| `test_read_with_limit` | Reading a file with a limit (restricting the number of lines) |
| `test_read_with_offset_and_limit` | Reading a file with offset and limit at the same time |
| `test_read_unicode_content` | Reading Unicode content |
| `test_read_file_with_very_long_lines` | Reading a file with very long lines |
| `test_read_with_zero_limit` | Reading a file with limit=0 |
| `test_read_offset_beyond_file_length` | Reading with an offset beyond the file length |
| `test_read_offset_at_exact_file_length` | Reading with an offset at exactly the file length |
| `test_read_very_large_file_in_chunks` | Reading a very large file in chunks |

### 3.3 Edit Operations

Tests of the `edit()` method for replacing strings in files:

| Test | Description |
|------|----------|
| `test_edit_single_occurrence` | Replacing a single occurrence of a string |
| `test_edit_multiple_occurrences_without_replace_all` | Error on multiple occurrences without replace_all |
| `test_edit_multiple_occurrences_with_replace_all` | Replacing all occurrences of a string with replace_all=True |
| `test_edit_string_not_found` | Error when the string is not found |
| `test_edit_nonexistent_file` | Editing a nonexistent file |
| `test_edit_special_characters` | Editing with special characters |
| `test_edit_multiline_support` | Support for multiline replacements |
| `test_edit_with_empty_new_string` | Replacing with an empty string (deletion) |
| `test_edit_identical_strings` | Error when old_string and new_string are identical |
| `test_edit_unicode_content` | Editing Unicode content |
| `test_edit_whitespace_only_strings` | Editing strings consisting only of whitespace |
| `test_edit_with_very_long_strings` | Editing very long strings |
| `test_edit_line_ending_preservation` | Preserving line endings when editing |
| `test_edit_partial_line_match` | Partial line match when editing |

### 3.4 LS Info Operations

Tests of the `ls_info()` method for retrieving information about directory contents:

| Test | Description |
|------|----------|
| `test_ls_info_path_is_absolute` | Verifies that the path is absolute |
| `test_ls_info_basic_directory` | Listing a basic directory |
| `test_ls_info_empty_directory` | Listing an empty directory |
| `test_ls_info_nonexistent_directory` | Listing a nonexistent directory |
| `test_ls_info_hidden_files` | Listing hidden files |
| `test_ls_info_directory_with_spaces` | Listing a directory with spaces in the names |
| `test_ls_info_unicode_filenames` | Listing files with Unicode names |
| `test_ls_info_large_directory` | Listing a large directory |
| `test_ls_info_path_with_trailing_slash` | Listing with a trailing slash in the path |
| `test_ls_info_special_characters_in_filenames` | Listing files with special characters in the names |

### 3.5 Grep Operations

Tests of the `grep_raw()` method for searching patterns in files:

| Test | Description |
|------|----------|
| `test_grep_basic_search` | Basic string search |
| `test_grep_with_glob_pattern` | Search with a glob pattern for filtering files |
| `test_grep_no_matches` | Search with no matches |
| `test_grep_multiple_matches_per_file` | Several matches in a single file |
| `test_grep_literal_string_matching` | Searching for a literal string |
| `test_grep_unicode_pattern` | Searching for a Unicode pattern |
| `test_grep_case_sensitivity` | Case sensitivity when searching |
| `test_grep_with_special_characters` | Searching with special characters |
| `test_grep_empty_directory` | Searching in an empty directory |
| `test_grep_across_nested_directories` | Searching in nested directories |
| `test_grep_with_multiline_matches` | Searching with multiline matches |

### 3.6 Glob Operations

Tests of the `glob_info()` method for file pattern matching:

| Test | Description |
|------|----------|
| `test_glob_basic_pattern` | Basic glob pattern |
| `test_glob_recursive_pattern` | Recursive glob pattern (with `**`) |
| `test_glob_no_matches` | Glob pattern with no matches |
| `test_glob_with_directories` | Glob with directories |
| `test_glob_specific_extension` | Glob for a specific file extension |
| `test_glob_hidden_files_explicitly` | Explicit glob for hidden files |
| `test_glob_with_character_class` | Glob with a character class ([]) |
| `test_glob_with_question_mark` | Glob with a question mark (?) for a single character |
| `test_glob_multiple_extensions` | Glob for several extensions |
| `test_glob_deeply_nested_pattern` | Glob for deeply nested patterns |
| `test_glob_with_no_path_argument` | Glob without a path argument |

### 3.7 Complex Scenarios

Integration tests verifying combinations of operations:

| Test | Description |
|------|----------|
| `test_write_read_edit_workflow` | Full workflow: creating a file → reading → editing |
| `test_complex_directory_operations` | Complex directory operations |

---

## 4. Sandbox Factory Tests

**File**: `deep_worker/deep_worker_cli/tests/integration_tests/test_sandbox_factory.py`

Tests verifying the factory that creates sandbox containers.

| Test | Description |
|------|----------|
| `test_sandbox_creation` | Creating sandboxes of various types (Daytona) |
| `test_upload_single_file` | Uploading a single file to the sandbox |
| `test_download_single_file` | Downloading a single file from the sandbox |
| `test_upload_download_roundtrip` | Uploading a file → downloading → verifying the content |
| `test_upload_multiple_files` | Uploading several files at once |
| `test_download_multiple_files` | Downloading several files at once |
| `test_download_nonexistent_file` | Downloading a nonexistent file |
| `test_upload_binary_content` | Uploading binary content |
| `test_partial_success_upload` | Partial success on upload (some files uploaded) |
| `test_partial_success_download` | Partial success on download (some files downloaded) |
| `test_download_error_file_not_found` | Download error: file not found |
| `test_download_error_is_directory` | Download error: the path is a directory |
| `test_upload_error_parent_not_found` | Upload error: parent directory not found |
| `test_upload_error_invalid_path` | Upload error: invalid path |
| `test_download_error_invalid_path` | Download error: invalid path |
| `test_upload_to_existing_directory_path` | Uploading to the path of an existing directory |

---

## 5. IPC Integration Tests

**File**: `deep_worker/deep_worker_cli/tests/integration_tests/test_ipc_integration.py`

Tests verifying inter-process communication (IPC) between deep-worker components.

| Test | Description |
|------|----------|
| `test_process_registry_registration` | Registering a process in ProcessRegistry |
| `test_process_registry_lookup` | Looking up a registered process |
| `test_process_registry_unregister` | Removing a process from the registry |
| `test_agent_client_connection` | Connecting to an agent through AgentClient |
| `test_agent_client_query` | Sending a query to an agent through AgentClient |
| `test_websocket_connection` | WebSocket connection to an agent |
| `test_websocket_message_exchange` | Exchanging messages over WebSocket |

---

## Running the Tests

### Running all integration tests

```bash
# All Terminal Emulator integration tests (AI)
pytest deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py -v

# All Agent tests (AI validation)
pytest deep_worker/deep_worker_cli/tests/integration_ai/test_agent_basic_example.py -v

# All Sandbox Operations tests
pytest deep_worker/deep_worker_cli/tests/integration_tests/test_sandbox_operations.py -v

# All Sandbox Factory tests
pytest deep_worker/deep_worker_cli/tests/integration_tests/test_sandbox_factory.py -v

# All IPC Integration tests
pytest deep_worker/deep_worker_cli/tests/integration_tests/test_ipc_integration.py -v
```

### Running a specific test

```bash
# Example: running the query visibility test in dwcli
pytest deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_query_visibility -xvs

# Example: running the test for preserving history on reconnect
pytest deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_reconnect_preserves_history -xvs
```

### Running only the critical tests

```bash
# Critical dwcli tests
pytest deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_query_visibility \
       deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_reconnect_preserves_history \
       deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_input_visible_with_long_history \
       deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_scroll_pagination \
       deep_worker/deep_worker_cli/tests/integration_ai/test_terminal_emulator.py::test_dwcli_input_area_immediately_visible \
       -v
```

---

## Requirements

- Python 3.12+
- pytest 8.4.2+
- deep-worker installed in the environment
- dwcli available in PATH
- For sandbox tests: Daytona installed and configured
- For AI validation: access to a model API (Anthropic Claude)

---

## Known Issues

1. **Sandbox tests can be slow**: The sandbox operations tests use real containers, so they may take several minutes to run.

2. **AI validation requires an API key**: Tests with AI validation require a configured API key for access to the Claude model.

3. **TUI tests are timing-sensitive**: Some TUI tests may be unstable under high system load. Increasing the timeouts is recommended in such cases.

4. **🔒 Synchronized tests**: The `test_dwcli_query_visibility` and `test_deepagents_query_does_not_disappear` tests must remain synchronized. Changes in one test must be reflected in the other.

---

## Additional Documentation

- **AI validation**: `deep_worker/deep_worker_cli/tests/integration_ai/README_AI_VALIDATION.md`
- **Terminal Emulator**: `deep_worker/deep_worker_cli/tests/integration_ai/README.md`
- **Response checking guide**: `deep_worker/deep_worker_cli/tests/integration_ai/CHECK_REPLY_GUIDE.md`
- **Usage guide**: `deep_worker/deep_worker_cli/tests/integration_ai/USAGE_GUIDE.md`

---

**Last updated**: 2026-02-12
