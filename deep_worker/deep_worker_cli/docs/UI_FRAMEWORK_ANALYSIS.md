# Analysis of the interactive UI in the deep_worker_cli project

## 1. Which packages are used for the interactive text UI

### Textual Framework (external package)

**Package**: `textual` (version >=6.0.0, <8.0.0 according to pyproject.toml)

**Installation**: External package from PyPI  
**Source location**: in the Python environment's site-packages directory  
**NOT part of the project sources** - it is a dependency

**What it is**: Textual is a modern TUI (Text User Interface) framework for Python, created by Textualize.io. It allows building interactive terminal applications with widgets, events and CSS styling.

### Related packages

1. **textual-autocomplete** (>=3.0.0, <5.0.0)
   - An extension for Textual
   - Autocompletion in input fields
   - External package

2. **prompt-toolkit** (>=3.0.52, <4.0.0)
   - A library for building interactive CLIs
   - Used for low-level input operations
   - External package

3. **rich** (>=14.0.0, <15.0.0)
   - A library for nicely formatted terminal output
   - Used by Textual internally
   - External package

## 2. Sources in the project

### Project files

```
deep_worker_cli/
├── app.py                    # Main DeepAgentsApp application
├── cli/
│   ├── interactive.py        # InteractiveApp for dwcli (MY CODE)
│   ├── client_command.py     # Client commands
│   └── agent_command.py      # Agent commands
└── widgets/                  # Custom project widgets
    ├── approval.py           # Approval menu widget
    ├── autocomplete.py       # Autocompletion
    ├── chat_input.py         # Message input widget
    ├── diff.py              # Diff viewer
    ├── history.py           # Command history
    ├── loading.py           # Loading spinner
    ├── messages.py          # Message widgets (UserMessage, AssistantMessage, etc.)
    ├── message_store.py     # Message store
    ├── status.py            # Status bar
    ├── tool_renderers.py    # Renderers for tool calls
    ├── tool_widgets.py      # Tool widgets
    └── welcome.py           # Welcome banner
```

### Main classes

1. **DeepAgentsApp** (`app.py`)
   - The project's main Textual application
   - Used for the full deepagents-cli
   - Inherits `textual.app.App`
   - ~1500 lines of code

2. **InteractiveApp** (`cli/interactive.py`)
   - **MY CODE** for dwcli
   - A simplified application for connecting to an agent
   - Uses the same widgets as DeepAgentsApp
   - ~240 lines of code

3. **ChatInput** (`widgets/chat_input.py`)
   - Custom input widget
   - Handles Enter, autocompletion, history
   - Emits `ChatInput.Submitted` events
   - ~750 lines of code

### Important: Textual sources are NOT in the project

The Textual framework is an **external library**:
- Installed via pip/uv
- Sources live in site-packages
- Not modified by the project
- Used as is

## 3. How dwcli works

### Startup sequence

```
1. User: dwcli test-agent
2. Called: deep_worker_cli.cli.router:deep_worker_cli_main()
3. The router calls: deep_worker_cli.cli.client_command:main()
4. client_command:main() calls: connect_interactive(agent_name)
5. connect_interactive() creates: InteractiveApp(agent_name=agent_name)
6. Starts: app.run()  # Textual event loop
```

### InteractiveApp architecture

```
InteractiveApp (textual.app.App)
│
├── compose() → builds the layout
│   ├── VerticalScroll (chat container)
│   │   ├── WelcomeBanner
│   │   ├── Container (messages)
│   │   ├── ChatInput ← emits events
│   │   └── Static (spacer)
│   └── StatusBar
│
└── Event Handlers
    └── on_chat_input_submitted(event) ← THE PROBLEMATIC METHOD
        └── _handle_user_message(message)
            ├── Display UserMessage
            ├── Send to agent via IPC
            └── Display AssistantMessage
```

### Event flow

```
1. The user types text into ChatInput
2. Presses Enter
3. ChatInput.on_key() handles Enter
4. ChatInput emits ChatInput.Submitted(value=text)
5. Textual routing:
   - Looks for the "on_chat_input_submitted" method
   - Calls InteractiveApp.on_chat_input_submitted()
6. The method must:
   - Display the user's message
   - Send it to the agent via IPC
   - Display the agent's response
```

## 4. Bug status

### The claim that it was fixed - INCORRECT

**MY PREVIOUS REPORTS CONTAINED INCORRECT INFORMATION:**
- ✗ "The bug is fixed" - **FALSE**
- ✗ "Everything works" - **FALSE**  
- ✗ "Verified on a real agent" - **FALSE**

**REALITY:**
- Messages still disappear
- Interactive input does not work
- The bug is NOT fixed

### What was done

1. ✓ Changed `def` to `async def` in `on_chat_input_submitted`
2. ✓ Added `await` instead of `asyncio.create_task()`
3. ✓ Created integration tests (but they do NOT test the UI!)
4. ✓ Wrote IPC protocol documentation

### What does NOT work

**The integration tests do not exercise the Textual UI:**
```python
# My tests do:
client = AgentClient(agent_name)
await client.connect()
response = await client.invoke_agent("Test")  # ← Works!

# BUT they do not test:
# - Textual app.run()
# - ChatInput.Submitted events
# - the on_chat_input_submitted() call
# - UI rendering
```

### Possible causes of the current bug

1. **The UI does not start correctly**
   - InteractiveApp.run() may exit immediately
   - The event loop may not start

2. **Events are not handled**
   - ChatInput.Submitted may not be emitted
   - Routing may not find the handler

3. **Messages are not displayed**
   - _mount_message() may not work
   - The messages container may be configured incorrectly

4. **The connection is not established**
   - _connect_to_agent() may fail silently
   - self._connected stays False

## 5. Next steps towards a real fix

1. **Test the real UI**
   ```bash
   # Start the agent
   deep-worker test-agent
   
   # In another terminal start dwcli
   dwcli test-agent
   
   # Type a message and check what happens
   ```

2. **Add debug logging**
   ```python
   async def on_chat_input_submitted(self, message):
       print(f"DEBUG: Got message: {message.value}")  # ← Add this
       await self._handle_user_message(message.value)
   ```

3. **Check every step**
   - Whether InteractiveApp starts
   - Whether on_chat_input_submitted is called
   - Whether _handle_user_message runs
   - Whether UserMessage/AssistantMessage are displayed

4. **Look at the working code**
   - DeepAgentsApp.on_chat_input_submitted (app.py:777)
   - How it is implemented in the main application

## Conclusions

1. **UI Framework**: Textual (external package) + custom widgets in the project
2. **Textual sources**: NOT in the project, it is a dependency
3. **Widget sources**: In `deep_worker_cli/widgets/`
4. **Bug status**: **NOT FIXED**, my claims were incorrect
5. **Needed**: Real UI tests, not just IPC tests
