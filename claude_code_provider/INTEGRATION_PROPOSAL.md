# Proposals for Integrating the Claude Code Provider

## Analysis of the current architecture

### Key findings

1. **The agent command vs the CLI client**:
   - `deep-worker [name]` - starts the agent itself (`cli/agent_command.py`)
   - The CLI client - connects to an already running agent as a client
   - The source of truth is the AGENT, not the client

2. **How provider configuration works**:
   ```python
   # agent_command.py creates the AgentConfig
   config = AgentConfig(
       model=args.model,        # From the --model flag
       provider=args.provider,  # From the --provider flag
       working_dir=working_dir
   )

   # agent_process.py starts the agent
   model = create_model()  # ❌ PROBLEM: the config is NOT PASSED!
   ```

3. **Critical problem in the current code**:
   - `AgentConfig` stores `provider` and `model`
   - But `create_model()` is called WITHOUT parameters!
   - `create_model()` determines the provider from the presence of API keys (priority: deepseek → openai → anthropic → google)
   - This means the `--provider` and `--model` flags are IGNORED!

4. **The Claude Code SDK supports simple model names**:
   ```python
   # In types.py:
   model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None

   # The Claude CLI resolves automatically:
   # "sonnet" -> "claude-sonnet-4-5-20250929" (the latest version)
   # "haiku" -> "claude-3-5-haiku-20241022" (the latest version)
   # "opus" -> "claude-opus-4-1-20250805" (the latest version)
   ```

## Integration proposals

### IMPORTANT: Fixes for CRITICAL PROBLEMS

Before adding Claude Code, the current code must be fixed:

**File: `process/agent_process.py`**

Line 173, CHANGE:
```python
# BEFORE (DOES NOT USE config.provider and config.model!):
model = create_model()

# SHOULD BE:
model = create_model(
    provider=self.config.provider,
    model_name=self.config.model
)
```

**File: `config.py`**

Line 1061, CHANGE the function signature:
```python
# BEFORE:
def create_model(model_name_override: str | None = None) -> BaseChatModel:

# SHOULD BE:
def create_model(
    provider: str | None = None,
    model_name: str | None = None,
    model_name_override: str | None = None  # Deprecated, for backward compatibility
) -> BaseChatModel:
    """Create the appropriate model based on provider and model name.

    Args:
        provider: Provider name ("deepseek", "openai", "anthropic", "google", "vertexai", "claude-code")
        model_name: Model name (e.g., "deepseek-chat", "gpt-4o", "sonnet", "haiku", "opus")
        model_name_override: DEPRECATED - use model_name instead

    Returns:
        ChatModel instance
    """
    # Support legacy parameter name
    if model_name_override and not model_name:
        model_name = model_name_override

    # If provider and model are explicitly provided, use them
    if provider and model_name:
        # Validate provider has credentials (except claude-code)
        if provider == "claude-code":
            # Claude Code doesn't need API key
            pass
        elif provider == "deepseek" and not settings.has_deepseek:
            console.print(f"[bold red]Error:[/bold red] Provider '{provider}' requires DEEPSEEK_API_KEY")
            sys.exit(1)
        # ... similar checks for other providers

    # Fallback to auto-detection if not provided
    elif not provider or not model_name:
        # EXISTING auto-detection logic (lines 1152-1182)
        if settings.has_deepseek:
            provider = "deepseek"
            model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        # ... etc
```

### CLAUDE CODE INTEGRATION

After fixing the critical problems, add Claude Code:

#### 1. File: `agent_command.py` (CLI arguments)

Lines 42-47, CHANGE:
```python
parser.add_argument(
    "--provider",
    default="deepseek",  # We do not change the default!
    choices=["deepseek", "gemini", "openai", "claude-code"],  # Add claude-code
    help="AI provider to use (default: deepseek)",
)
```

Lines 36-40, CHANGE (optional):
```python
parser.add_argument(
    "--model",
    default=None,  # None = the provider's default model is used
    help="Model to use. Examples: 'deepseek-chat', 'gpt-4o', 'sonnet', 'haiku', 'opus'",
)
```

#### 2. File: `agent_config.py` (Validation)

Lines 51-56, CHANGE:
```python
# Validate provider
valid_providers = ["deepseek", "gemini", "openai", "claude-code"]  # Add claude-code
if self.provider not in valid_providers:
    raise ValueError(
        f"provider must be one of {valid_providers}, got '{self.provider}'"
    )
```

Lines 58-80, CHANGE (remove the API key requirement for claude-code):
```python
# Auto-load API key from environment if not provided
if self.api_key is None:
    if self.provider == "claude-code":
        # Claude Code does not require an API key!
        self.api_key = None
    elif self.provider == "deepseek":
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment variables. "
                "Add it to .env file: DEEPSEEK_API_KEY=sk-..."
            )
    elif self.provider == "gemini":
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Add it to .env file: GOOGLE_API_KEY=AIza..."
            )
    elif self.provider == "openai":
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Add it to .env file: OPENAI_API_KEY=sk-..."
            )
```

Lines 31-39, CHANGE the model default:
```python
@dataclass
class AgentConfig:
    model: str | None = None  # None = the provider's default model is used
    working_dir: Path
    provider: str = "deepseek"
    system_prompt: str | None = None
    tools: list[str] = field(default_factory=list)
    backend_type: str = "state"
    max_tokens: int = 20000
    temperature: float = 1.0
    api_key: str | None = None
```

Lines 47-49, CHANGE the validation:
```python
# Validate model (allow None - will use provider's default)
if self.model is not None and not self.model.strip():
    raise ValueError("model must be a non-empty string or None")
```

#### 3. File: `config.py` (Model creation)

After line 1258, ADD a new branch for claude-code:
```python
    elif provider == "claude-code":
        # Import ClaudeCodeChatModel from source
        import sys
        from pathlib import Path as PathLib

        _provider_path = PathLib(__file__).parent.parent.parent / "claude_code_provider"
        if _provider_path.exists() and str(_provider_path) not in sys.path:
            sys.path.insert(0, str(_provider_path))

        from claude_agent_sdk import ClaudeAgentOptions
        from claude_code_provider import ClaudeCodeChatModel

        # Map simple names to Claude SDK format
        model_mapping = {
            "sonnet": "sonnet",      # The Claude SDK automatically resolves
            "haiku": "haiku",        # to the latest version
            "opus": "opus",
            # We also support full names for compatibility:
            "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
            "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
            "claude-opus-4-1-20250805": "claude-opus-4-1-20250805",
        }

        # Set default model if not specified
        if not model_name:
            model_name = "sonnet"  # Default: Sonnet

        # Map model name (if it's a simple name, use directly; if full name, also use)
        resolved_model = model_mapping.get(model_name, model_name)

        # Create options with model
        options = ClaudeAgentOptions(model=resolved_model)

        # Create ClaudeCodeChatModel
        model = ClaudeCodeChatModel(base_options=options)

        logger.info(f"✓ Created Claude Code model: {resolved_model}")
        logger.info("  Note: Claude Code uses local installation, no API key required")
```

Also at lines 1185-1186, UPDATE:
```python
# Store model info in settings for display
settings.model_name = model_name
settings.model_provider = provider
```

### USAGE

#### Running with Claude Code:

```bash
# Default model (sonnet):
deep-worker myagent --provider claude-code

# A specific model (haiku - fast):
deep-worker myagent --provider claude-code --model haiku

# A specific model (opus - powerful):
deep-worker myagent --provider claude-code --model opus

# A full model name (for a specific version):
deep-worker myagent --provider claude-code --model claude-sonnet-4-5-20250929
```

#### Verification:

```bash
# Connect to the running agent with the CLI client:
dwcli myagent

# Or via the API:
from client import DeepWorkerClient  # the CLI client package

client = DeepWorkerClient(agent_name="myagent")
response = await client.send_message("Hello!")
```

## Claude Code models

| Simple name | Purpose | Speed | Quality |
|------------|-----------|----------|----------|
| `sonnet` | Balanced (DEFAULT) | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `haiku` | Fast/cheap | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| `opus` | Maximum quality | ⚡⚡ | ⭐⭐⭐⭐⭐ |

The Claude SDK automatically resolves simple names to the latest model versions.

## Advantages of this approach

1. **No hardcoded model versions** - the Claude SDK itself resolves "sonnet" → the latest version
2. **The source of truth is AgentConfig** - provider and model are passed explicitly
3. **No API key required** - Claude Code uses a local installation
4. **Backward compatibility** - old code continues to work with auto-detection
5. **Default provider = deepseek** - unchanged
6. **Simple model names** - "sonnet", "haiku", "opus" instead of long strings

## Files to be changed (Total: 3 files)

1. `cli/agent_command.py`
   - Add "claude-code" to choices

2. `agent_config_pkg/agent_config.py`
   - Add "claude-code" to valid_providers
   - Do not require an API key for claude-code
   - Allow model=None (the provider's default model is used)

3. `config.py`
   - Add the `provider` and `model_name` parameters to `create_model()`
   - Add a branch for the claude-code provider
   - Use simple model names

4. `process/agent_process.py`
   - Pass `config.provider` and `config.model` into `create_model()`

## Critical problems in the current code (require fixing)

1. **agent_process.py line 173**: `create_model()` is called without parameters - the config is ignored!
2. **The --provider and --model flags do not work**: an AgentConfig is created, but is not used afterwards
3. **provider/model are not passed into create_model()**: only auto-detection by API keys

These problems MUST be fixed before or together with adding Claude Code.
