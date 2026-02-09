"""
Custom Tools for LangChain Agents

Tools that can be used by agents to perform various tasks.
"""

from langchain_core.tools import tool
import os


@tool
def calculator(expression: str) -> float:
    """
    Calculates mathematical expression

    Args:
        expression: Mathematical expression (e.g., '2 + 2 * 3')

    Returns:
        Result of calculation
    """
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for information

    Args:
        query: Search query

    Returns:
        Search results (stub - returns placeholder)
    """
    # This is a stub - in real implementation, use requests + BeautifulSoup
    # or integrate with actual search API (Google, Bing, etc)
    return f"Web search for '{query}' would return results here (stub implementation)"


@tool
def read_file(filepath: str) -> str:
    """
    Read file content

    Args:
        filepath: Path to file

    Returns:
        File content
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write content to file

    Args:
        filepath: Path to file
        content: Content to write

    Returns:
        Success message or error
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# List of all available tools
AVAILABLE_TOOLS = [calculator, search_web, read_file, write_file]
