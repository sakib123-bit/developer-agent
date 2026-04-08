import asyncio
import os
import sys
from contextlib import AsyncExitStack

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPAgent:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.agent_executor = None

    async def initialize(self):
        """Initialize connections to MCP servers and set up the agent."""
        # Determine the base directory (project root)
        # agent/mcp_agent.py -> project_root/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        server_configs = [
            {
                "name": "jira",
                "command": sys.executable,
                "args": [os.path.join(base_dir, "jira_mcp_server.py")]
            },
            {
                "name": "codebase",
                "command": sys.executable,
                "args": [os.path.join(base_dir, "mcp_server.py")]
            }
        ]

        for config in server_configs:
            try:
                # load_mcp_tools is an async context manager that yields a list of tools
                server_tools = await self.exit_stack.enter_async_context(
                    load_mcp_tools(
                        config["command"],
                        config["args"],
                        env=os.environ.copy()
                    )
                )
                self.tools.extend(server_tools)
            except Exception as e:
                print(f"Failed to connect to server {config['name']}: {e}", file=sys.stderr)

        if not self.tools:
            raise RuntimeError("No tools were loaded from MCP servers. Check if servers are running correctly.")

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
    agent = MCPAgent()
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