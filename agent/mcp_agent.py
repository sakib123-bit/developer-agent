import asyncio
import json
import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage


class MCPAgent:
    def __init__(self, server_configs: List[Dict[str, Any]]):
        """
        Initialize the MCPAgent with a list of server configurations.
        Each config should have 'command', 'args', and optionally 'env'.
        """
        self.server_configs = server_configs
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tools: List[StructuredTool] = []
        self.agent_executor = None

    async def _connect_to_server(self, config: Dict[str, Any]):
        """Connect to a single MCP server and retrieve its tools."""
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env={**os.environ, **config.get("env", {})}
        )
        
        transport_ctx = stdio_client(server_params)
        read, write = await self.exit_stack.enter_async_context(transport_ctx)
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        
        await session.initialize()
        self.sessions.append(session)
        
        # List tools from the server
        response = await session.list_tools()
        
        for mcp_tool in response.tools:
            # Create a wrapper function for the MCP tool
            def create_tool_func(s: ClientSession, name: str):
                async def tool_func(**kwargs):
                    result = await s.call_tool(name, arguments=kwargs)
                    return result.content
                return tool_func

            # Convert MCP tool schema to LangChain StructuredTool
            # Note: MCP uses JSON Schema for inputSchema
            lc_tool = StructuredTool.from_function(
                coroutine=create_tool_func(session, mcp_tool.name),
                name=mcp_tool.name,
                description=mcp_tool.description or f"Tool {mcp_tool.name} from MCP server",
                # We pass the schema directly if possible, or let StructuredTool infer
                # For simplicity in this generic implementation, we rely on the dynamic function
            )
            self.tools.append(lc_tool)

    async def initialize(self):
        """Initialize connections to all servers and set up the agent."""
        for config in self.server_configs:
            try:
                await self._connect_to_server(config)
            except Exception as e:
                print(f"Failed to connect to server {config.get('command')}: {e}", file=sys.stderr)

        if not self.tools:
            print("Warning: No tools were loaded from MCP servers.")

        # Initialize the LLM and ReAct agent
        # Ensure OPENAI_API_KEY is set in environment
        llm = ChatOpenAI(model="gpt-4o")
        self.agent_executor = create_react_agent(llm, self.tools)

    async def run(self, prompt: str):
        """Execute a prompt using the aggregated tools."""
        if not self.agent_executor:
            await self.initialize()

        inputs = {"messages": [HumanMessage(content=prompt)]}
        result = await self.agent_executor.ainvoke(inputs)
        
        # Return the last message content (the agent's final response)
        return result["messages"][-1].content

    async def cleanup(self):
        """Close all server connections."""
        await self.exit_stack.aclose()


async def main():
    """CLI entry point for testing the MCPAgent."""
    if len(sys.argv) < 2:
        print("Usage: python mcp_agent.py <config_json_path> or provide a prompt directly")
        # Default config for testing if none provided
        configs = []
    else:
        # Try to load config from first arg if it's a file
        config_path = sys.argv[1]
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
        else:
            print(f"Config file {config_path} not found. Using empty config.")
            configs = []

    agent = MCPAgent(configs)
    try:
        await agent.initialize()
        print("MCP Agent initialized. Type 'exit' to quit.")
        
        while True:
            try:
                prompt = input("\nUser: ").strip()
                if prompt.lower() in ["exit", "quit"]:
                    break
                if not prompt:
                    continue
                
                response = await agent.run(prompt)
                print(f"\nAgent: {response}")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())