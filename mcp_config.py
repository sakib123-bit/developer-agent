import os

"""
Centralized configuration for Model Context Protocol (MCP) servers.
This file defines the connection parameters for various MCP servers used by the application.
"""

MCP_SERVERS = {
    "jira": {
        "command": "npx",
        "args": ["-y", "@atlassian/jira-mcp-server"],
        "env": {
            "JIRA_URL": os.environ.get("JIRA_URL"),
            "JIRA_API_TOKEN": os.environ.get("JIRA_API_TOKEN"),
            "JIRA_USERNAME": os.environ.get("JIRA_USERNAME"),
        },
    },
    # Future servers can be added here following the same structure:
    # "example-server": {
    #     "command": "npx",
    #     "args": ["-y", "@org/server-name"],
    #     "env": {
    #         "API_KEY": os.environ.get("EXAMPLE_API_KEY"),
    #     },
    # },
}