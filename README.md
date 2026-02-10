# LangChain Integration with Security

Complete LangChain agent integration for the web application with a Claude Code-compatible security model and `skip_permissions` support.

## Overview

This project provides a comprehensive LangChain agent framework integrated with the web application, featuring:

- **Security Framework**: STRICT (default) and PERMISSIVE (--skip-permissions) modes matching Claude Code behavior
- **16 Tools**: Shell, File, Python, Git, and Web tools for agent interaction
- **Session Management**: Full lifecycle management with message history
- **WebSocket Support**: Real-time streaming of task execution
- **UI Integration**: Form-based session creation with working directory and permission controls

## Quick Start

### Installation

```bash
cd <repository-root>
pip install -r requirements.txt
```

### Running Examples

```bash
# Basic LangChain usage
python examples/example_gemini.py

# Check API quota
python examples/check_quota.py

# Full comparison of providers
python examples/comparison.py
```

## Documentation

- **[docs/](docs/)** - Complete documentation including:
  - API guides
  - Security policies
  - Deployment instructions
  - Phase overviews
  - Integration analysis

- **[examples/](examples/)** - Working examples:
  - Gemini provider setup
  - OpenAI integration
  - DeepSeek integration
  - Quota checking
  - Provider comparison

- **[scripts/](scripts/)** - Setup and utility scripts

## Security Model

### STRICT Mode (Default)
- Blocks dangerous shell commands (rm, mkfs, kill, chmod, etc.)
- Blocks dangerous Python patterns (subprocess, exec, eval, os.system)
- Prevents path traversal attacks
- Protects system paths (/etc, /sys, /proc, /dev, /root, ~/.ssh)

### PERMISSIVE Mode (--skip-permissions)
- Allows all dangerous operations
- Maintains output and size limits
- Maintains working directory isolation
- Equivalent to `claude --skip-permissions`

## Key Features

### Resource Limits (Claude Code Compatible)
- Command timeout: 60 seconds
- Max output: 100KB
- Max file read: 10MB
- Max file write: 50MB
- Python timeout: 60 seconds
- Web timeout: 30 seconds

### Tool Categories

**Shell Tools**
- `execute_command`: Run shell commands
- `get_cwd`: Get current working directory
- `list_dir`: List directory contents

**File Tools**
- `read_file`: Read file contents
- `write_file`: Write to file
- `append_file`: Append to file
- `delete_file`: Delete file
- `search_files`: Search with glob patterns

**Python Tools**
- `execute_python`: Execute Python code
- `eval_expr`: Evaluate Python expressions

**Git Tools**
- `git_status`: Get repository status
- `git_log`: View commit history
- `git_diff`: Show differences
- `git_branch`: Get current branch

**Web Tools**
- `fetch_url`: Fetch URL content
- `search_web`: Web search (placeholder)

## Architecture

```
.
├── docs/              # Documentation
├── examples/          # Code examples
├── scripts/           # Setup scripts
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## API Integration

The system integrates with the web application through REST API and WebSocket:

- Create sessions with `skip_permissions` flag
- Execute tasks with real-time streaming
- Monitor session status
- Access message history

## Dependencies

- `langchain-core`: Core LangChain framework
- `langchain-google-genai`: Google Gemini integration
- `langchain-openai`: OpenAI integration
- `google-generativeai`: Direct Gemini API access

See `requirements.txt` for full dependency list.

## Environment Variables

```bash
# Google Gemini API
export GOOGLE_API_KEY="your-api-key"

# OpenAI API
export OPENAI_API_KEY="your-api-key"

# DeepSeek API (if using)
export DEEPSEEK_API_KEY="your-api-key"
```

## License

MIT

## Contributing

This is part of the main web application project. See the main repository for contribution guidelines.
