#!/bin/bash

# ============================================================================
# Setup deepagents-cli with Deepseek Provider
# ============================================================================
# This script configures deepagents-cli to use Deepseek API
# instead of default OpenAI/Anthropic
# ============================================================================

set -e

# Get the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
DEEPAGENTS_HOME="${HOME}/.deepagents"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Setting up deepagents-cli with Deepseek provider${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_DIR${NC}"
    exit 1
fi

# Load .env to get Deepseek API key
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo -e "${GREEN}✓ Loaded environment variables from .env${NC}"
else
    echo -e "${YELLOW}⚠ .env file not found${NC}"
fi

# Check if Deepseek API key is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo -e "${RED}Error: DEEPSEEK_API_KEY not set in .env${NC}"
    echo "Please add your Deepseek API key to .env file"
    exit 1
fi

echo -e "${GREEN}✓ Deepseek API key found${NC}\n"

# Create deepagents configuration directory
mkdir -p "$DEEPAGENTS_HOME"
echo -e "${GREEN}✓ Created deepagents config directory${NC}"

# Create a wrapper script that uses Deepseek through OpenAI-compatible API
cat > "$VENV_DIR/bin/deepagents-deepseek" << 'WRAPPER_EOF'
#!/bin/bash
# Deepseek wrapper for deepagents-cli
# This script uses Deepseek API through OpenAI-compatible endpoint

export OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

exec deepagents "$@"
WRAPPER_EOF

chmod +x "$VENV_DIR/bin/deepagents-deepseek"
echo -e "${GREEN}✓ Created deepagents-deepseek wrapper script${NC}"

# Create .env file for deepagents
cat > "$DEEPAGENTS_HOME/.env" << EOF_ENV
# Deepseek Configuration for deepagents-cli
OPENAI_API_KEY=${DEEPSEEK_API_KEY}
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=${DEEPSEEK_MODEL:-deepseek-chat}
EOF_ENV

echo -e "${GREEN}✓ Created deepagents configuration (.deepagents/.env)${NC}"

# Create default agent if not exists
if [ ! -d "$DEEPAGENTS_HOME/agents" ]; then
    mkdir -p "$DEEPAGENTS_HOME/agents"
    echo -e "${GREEN}✓ Created agents directory${NC}"
fi

# Create a sample agent with Deepseek configuration
cat > "$DEEPAGENTS_HOME/AGENTS.md" << 'AGENTS_EOF'
# Deepseek Agent Configuration

## Default Agent for Deepseek

This configuration file sets up agents for deepagents-cli using Deepseek API.

### Environment Variables Used
- `OPENAI_API_KEY`: Your Deepseek API key
- `OPENAI_BASE_URL`: Deepseek API endpoint (https://api.deepseek.com)
- `OPENAI_MODEL`: Model to use (deepseek-chat, deepseek-coder)

### Using Deepseek with deepagents-cli

```bash
# Using the wrapper script (recommended)
deepagents-deepseek

# Or using standard deepagents with model specification
deepagents --model deepseek-chat
```

AGENTS_EOF

echo -e "${GREEN}✓ Created default AGENTS.md${NC}"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

echo -e "You can now use deepagents-cli with Deepseek in two ways:\n"
echo -e "${BLUE}Option 1: Using the wrapper script (recommended)${NC}"
echo -e "  ${GREEN}$VENV_DIR/bin/deepagents-deepseek${NC}\n"

echo -e "${BLUE}Option 2: Using deepagents directly with model name${NC}"
echo -e "  ${GREEN}$VENV_DIR/bin/deepagents --model deepseek-chat${NC}\n"

echo -e "${BLUE}Option 3: Using the Python environment${NC}"
echo -e "  ${GREEN}source $VENV_DIR/bin/activate${NC}"
echo -e "  ${GREEN}deepagents-deepseek${NC}\n"

echo -e "Configuration saved to: ${GREEN}$DEEPAGENTS_HOME${NC}"
