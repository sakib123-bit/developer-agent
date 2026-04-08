from mcp.server.fastmcp import FastMCP
from indexer.searcher import Searcher
from api.git_ops import GitOps

# Initialize the MCP server
mcp = FastMCP("Codebase Agent")

# Initialize the underlying services
# These classes are expected to handle their own configuration (e.g., via environment variables)
searcher = Searcher()
git_ops = GitOps()

@mcp.tool()
def search_code(query: str, n_results: int = 5) -> str:
    """
    Perform a semantic search over the codebase to find relevant code snippets.
    
    Args:
        query: The natural language query or code snippet to search for.
        n_results: The number of results to return.
    """
    results = searcher.search(query, n_results=n_results)
    return str(results)

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the content of a file from the repository.
    
    Args:
        path: The relative path to the file.
    """
    return git_ops.read_file(path)

@mcp.tool()
def apply_diff(path: str, diff: str) -> str:
    """
    Apply a unified diff to a file in the repository.
    
    Args:
        path: The relative path to the file.
        diff: The unified diff string to apply.
    """
    return git_ops.apply_diff(path, diff)

@mcp.tool()
def create_pull_request(title: str, body: str, head: str, base: str = "main") -> str:
    """
    Create a new pull request on GitHub.
    
    Args:
        title: The title of the pull request.
        body: The description of the pull request.
        head: The name of the branch where your changes are implemented.
        base: The name of the branch you want the changes pulled into.
    """
    return git_ops.create_pull_request(title, body, head, base)

if __name__ == "__main__":
    mcp.run()