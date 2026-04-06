"""
Multi-Agent Framework
Collaborative AI agents with role-based specialization
"""
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class AgentRole(Enum):
    """Agent roles"""
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    EXECUTOR = "executor"


@dataclass
class Message:
    """Agent message"""
    sender: str
    receiver: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    role: AgentRole
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""


class BaseAgent:
    """Base agent class"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.role = config.role
        self.memory: List[Message] = []
        self.tools: Dict[str, Callable] = {}
    
    def add_tool(self, name: str, func: Callable):
        """Add a tool to the agent"""
        self.tools[name] = func
    
    def receive_message(self, message: Message):
        """Receive a message"""
        self.memory.append(message)
    
    def send_message(self, receiver: 'BaseAgent', content: str):
        """Send a message to another agent"""
        message = Message(
            sender=self.name,
            receiver=receiver.name,
            content=content
        )
        receiver.receive_message(message)
    
    def think(self, prompt: str) -> str:
        """
        Think (generate response)
        
        In real implementation, call LLM API
        """
        # Placeholder - in production, call OpenAI/Anthropic API
        return f"[{self.name}] Processing: {prompt[:50]}..."
    
    def act(self, action: str, **kwargs) -> str:
        """Execute an action"""
        if action in self.tools:
            return self.tools[action](**kwargs)
        return f"No tool found: {action}"
    
    def get_context(self) -> str:
        """Get conversation context"""
        context = f"## {self.name} ({self.role.value})\n"
        context += f"System: {self.config.system_prompt}\n\n"
        
        for msg in self.memory[-5:]:  # Last 5 messages
            context += f"{msg.sender}: {msg.content[:100]}...\n"
        
        return context


class ResearcherAgent(BaseAgent):
    """Research agent - gathers information"""
    
    def __init__(self, name: str = "Researcher"):
        config = AgentConfig(
            name=name,
            role=AgentRole.RESEARCHER,
            system_prompt="You are a research agent. Gather relevant information and provide summaries."
        )
        super().__init__(config)
        
        # Add research tools
        self.add_tool("search", self._search)
        self.add_tool("summarize", self._summarize)
    
    def _search(self, query: str) -> str:
        """Search for information"""
        return f"Search results for: {query}"
    
    def _summarize(self, text: str) -> str:
        """Summarize text"""
        return f"Summary of: {text[:50]}..."


class CoderAgent(BaseAgent):
    """Coding agent - writes code"""
    
    def __init__(self, name: str = "Coder"):
        config = AgentConfig(
            name=name,
            role=AgentRole.CODER,
            system_prompt="You are a coding agent. Write clean, efficient code."
        )
        super().__init__(config)
        
        self.add_tool("write_code", self._write_code)
        self.add_tool("debug", self._debug)
    
    def _write_code(self, task: str, language: str = "python") -> str:
        """Write code"""
        return f"# Code for: {task}\nprint('Hello, World!')"
    
    def _debug(self, code: str, error: str) -> str:
        """Debug code"""
        return f"Debugging: {error}"


class ReviewerAgent(BaseAgent):
    """Code review agent"""
    
    def __init__(self, name: str = "Reviewer"):
        config = AgentConfig(
            name=name,
            role=AgentRole.REVIEWER,
            system_prompt="You are a code review agent. Provide constructive feedback."
        )
        super().__init__(config)
        
        self.add_tool("review_code", self._review_code)
    
    def _review_code(self, code: str) -> str:
        """Review code"""
        return "Code review: Looks good! Minor suggestions..."


class AgentTeam:
    """
    Agent Team - manages multiple agents
    """
    
    def __init__(self, name: str = "Team"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List[Message] = []
        self.task_queue: List[Dict] = []
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the team"""
        self.agents[agent.name] = agent
        print(f"Added {agent.name} to team")
    
    def remove_agent(self, name: str):
        """Remove an agent"""
        if name in self.agents:
            del self.agents[name]
    
    def broadcast(self, sender: str, message: str):
        """Broadcast message to all agents"""
        for name, agent in self.agents.items():
            if name != sender:
                msg = Message(sender=sender, receiver=name, content=message)
                agent.receive_message(msg)
                self.message_history.append(msg)
    
    def delegate_task(self, task: str, role: AgentRole = None) -> str:
        """
        Delegate task to appropriate agent
        
        Args:
            task: Task description
            role: Specific role (optional)
        
        Returns:
            Task result
        """
        if role:
            # Find agent with specific role
            for agent in self.agents.values():
                if agent.role == role:
                    return agent.think(task)
        
        # Default: use coder
        for agent in self.agents.values():
            if agent.role == AgentRole.CODER:
                return agent.think(task)
        
        return "No suitable agent found"
    
    def run_workflow(self, workflow: List[Dict]) -> List[str]:
        """
        Run a multi-agent workflow
        
        Args:
            workflow: List of steps [{"agent": "name", "action": "task"}]
        
        Returns:
            List of results
        """
        results = []
        
        for step in workflow:
            agent_name = step.get("agent")
            action = step.get("action")
            params = step.get("params", {})
            
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                
                if action in agent.tools:
                    result = agent.act(action, **params)
                else:
                    result = agent.think(action)
                
                results.append(result)
                
                # Broadcast result to other agents
                self.broadcast(agent_name, result)
        
        return results
    
    def status(self) -> str:
        """Get team status"""
        status = f"## {self.name} Status\n"
        status += f"Agents: {len(self.agents)}\n\n"
        
        for name, agent in self.agents.items():
            status += f"- {name} ({agent.role.value}): {len(agent.memory)} messages\n"
        
        return status


class AgentManager:
    """
    Agent Manager - creates and manages agent teams
    """
    
    def __init__(self):
        self.teams: Dict[str, AgentTeam] = {}
    
    def create_team(self, name: str, roles: List[AgentRole] = None) -> AgentTeam:
        """Create a new team with specified roles"""
        team = AgentTeam(name)
        
        # Add default agents based on roles
        if roles is None:
            roles = [AgentRole.RESEARCHER, AgentRole.CODER, AgentRole.REVIEWER]
        
        for role in roles:
            if role == AgentRole.RESEARCHER:
                team.add_agent(ResearcherAgent())
            elif role == AgentRole.CODER:
                team.add_agent(CoderAgent())
            elif role == AgentRole.REVIEWER:
                team.add_agent(ReviewerAgent())
            elif role == AgentRole.PLANNER:
                team.add_agent(BaseAgent(AgentConfig(name="Planner", role=role)))
        
        self.teams[name] = team
        return team
    
    def get_team(self, name: str) -> Optional[AgentTeam]:
        """Get team by name"""
        return self.teams.get(name)
    
    def list_teams(self) -> List[str]:
        """List all teams"""
        return list(self.teams.keys())


# ============== Demo ==============

def demo():
    """Demo multi-agent system"""
    print("=" * 60)
    print("Multi-Agent System Demo")
    print("=" * 60)
    
    # Create manager
    manager = AgentManager()
    
    # Create team
    team = manager.create_team("DevTeam", [
        AgentRole.RESEARCHER,
        AgentRole.CODER,
        AgentRole.REVIEWER
    ])
    
    print("\n" + team.status())
    
    # Simple workflow
    print("\n--- Running Workflow ---")
    
    workflow = [
        {"agent": "Researcher", "action": "search", "params": {"query": "LoRA fine-tuning"}},
        {"agent": "Coder", "action": "write_code", "params": {"task": "Implement LoRA", "language": "python"}},
        {"agent": "Reviewer", "action": "review_code", "params": {"code": "# some code"}}
    ]
    
    results = team.run_workflow(workflow)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  Step {i+1}: {result}")
    
    print("\n" + "=" * 60)
    print("Multi-agent collaboration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
