#!/usr/bin/env bash
# Semantic Search CLI wrapper
#
# Usage:
#   sem_search -q "your search query"
#   sem_search -qf query.txt
#   sem_search -q "query" -d /path/to/repo
#   sem_search -q "query" -k 10

set -e

# Resolve symlinks to get the real script path
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do
    SCRIPT_DIR="$( cd -P "$( dirname "$SCRIPT_PATH" )" && pwd )"
    SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
    [[ $SCRIPT_PATH != /* ]] && SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SCRIPT_PATH" )" && pwd )"

# Add semantic_search directory to PYTHONPATH so Python can find the module
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -m semantic_search.search_cli "$@"
