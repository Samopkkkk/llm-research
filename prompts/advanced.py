"""
Advanced Prompt Engineering
Chain-of-Thought, ReAct, and other prompting techniques
"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Prompt template"""
    name: str
    template: str
    description: str


class ChainOfThought:
    """
    Chain-of-Thought Prompting
    
    Encourages step-by-step reasoning
    """
    
    @staticmethod
    def create_prompt(problem: str, examples: List[Dict] = None) -> str:
        """
        Create CoT prompt
        
        Args:
            problem: The problem to solve
            examples: Few-shot examples (optional)
        
        Returns:
            Formatted prompt
        """
        prompt = "Let's think step by step.\n\n"
        
        if examples:
            prompt += "Examples:\n"
            for ex in examples:
                prompt += f"Problem: {ex['problem']}\n"
                prompt += f"Solution: {ex['solution']}\n\n"
        
        prompt += f"Problem: {problem}\n"
        prompt += "Solution:"
        
        return prompt
    
    @staticmethod
    def extract_reasoning(response: str) -> str:
        """Extract reasoning from response"""
        # Look for reasoning keywords
        lines = response.split('\n')
        reasoning = []
        
        for line in lines:
            if any(word in line.lower() for word in ['first', 'then', 'therefore', 'so', 'because']):
                reasoning.append(line)
        
        return '\n'.join(reasoning) if reasoning else response


class TreeOfThoughts:
    """
    Tree of Thoughts Prompting
    
    Explores multiple reasoning paths
    """
    
    def __init__(self, num_thoughts: int = 3, depth: int = 2):
        self.num_thoughts = num_thoughts
        self.depth = depth
        self.thoughts: List[Dict] = []
    
    def create_prompt(self, problem: str) -> str:
        """Create ToT prompt"""
        prompt = f"""Consider multiple approaches to solve this problem:

Problem: {problem}

Generate {self.num_thoughts} different approaches, explore each one, then choose the best.
"""
        return prompt
    
    def evaluate_thought(self, thought: str, criteria: List[str]) -> Dict[str, float]:
        """Evaluate a thought against criteria"""
        scores = {}
        
        for criterion in criteria:
            # In real implementation, use LLM to evaluate
            scores[criterion] = 0.5  # Placeholder
        
        return scores
    
    def select_best(self) -> str:
        """Select best thought"""
        if not self.thoughts:
            return ""
        
        # Sort by average score
        best = max(self.thoughts, key=lambda t: sum(t.get('scores', {}).values()))
        return best.get('content', '')


class ReAct:
    """
    ReAct Prompting
    
    Combines reasoning and action
    """
    
    def __init__(self, tools: List[str]):
        self.tools = {t: t for t in tools}
        self.history: List[Dict] = []
    
    def create_prompt(self, task: str) -> str:
        """Create ReAct prompt"""
        tool_desc = "\n".join([f"- {t}: {t}" for t in self.tools.keys()])
        
        prompt = f"""Solve this task using reasoning and actions:

Task: {task}

Available tools:
{tool_desc}

Format your response as:
Thought: [your reasoning]
Action: [tool to use]
Observation: [result of action]
... (repeat)
Final Answer: [your final answer]
"""
        return prompt
    
    def parse_response(self, response: str) -> List[Dict]:
        """Parse ReAct response"""
        steps = []
        current_step = {}
        
        for line in response.split('\n'):
            if line.startswith('Thought:'):
                if current_step:
                    steps.append(current_step)
                current_step = {'thought': line[8:].strip()}
            elif line.startswith('Action:'):
                current_step['action'] = line[7:].strip()
            elif line.startswith('Observation:'):
                current_step['observation'] = line[12:].strip()
        
        if current_step:
            steps.append(current_step)
        
        return steps


class SelfConsistency:
    """
    Self-Consistency Prompting
    
    Multiple reasoning paths with majority voting
    """
    
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
    
    def create_prompt(self, problem: str) -> str:
        """Create self-consistency prompt"""
        prompt = f"""Solve this problem multiple ways. 
Think carefully and provide your best answer.

Problem: {problem}

Generate {self.num_samples} different solutions, then determine the most consistent answer.
"""
        return prompt
    
    def aggregate(self, answers: List[str]) -> str:
        """Aggregate multiple answers (majority voting)"""
        # Count answer frequencies
        from collections import Counter
        counts = Counter(answers)
        
        # Return most common
        return counts.most_common(1)[0][0]


class PromptOptimizer:
    """
    Automatic Prompt Optimization
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def optimize(self, initial_prompt: str, feedback: str) -> str:
        """
        Optimize prompt based on feedback
        
        Args:
            initial_prompt: Original prompt
            feedback: Feedback on prompt performance
        
        Returns:
            Optimized prompt
        """
        # Simple optimization rules
        # In production, use LLM to generate improvements
        
        optimized = initial_prompt
        
        # Add clarity instructions based on feedback
        if "unclear" in feedback.lower():
            optimized += "\nBe more specific and clear."
        
        if "too long" in feedback.lower():
            optimized = "Concisely: " + optimized
        
        if "wrong format" in feedback.lower():
            optimized += "\nFormat your response as: [Answer]"
        
        self.history.append({
            "original": initial_prompt,
            "feedback": feedback,
            "optimized": optimized
        })
        
        return optimized
    
    def get_best_prompt(self) -> str:
        """Get best performing prompt"""
        if not self.history:
            return ""
        
        # In real implementation, track scores
        return self.history[-1].get("optimized", "")


# ============== Demo ==============

def demo():
    """Demo prompt engineering"""
    print("=" * 60)
    print("Prompt Engineering Demo")
    print("=" * 60)
    
    # Chain of Thought
    print("\n1. Chain-of-Thought:")
    cot = ChainOfThought()
    problem = "If a train travels 60km in 30 minutes, how fast is it going in km/h?"
    prompt = cot.create_prompt(problem)
    print(f"Prompt: {prompt[:100]}...")
    
    # Tree of Thoughts
    print("\n2. Tree of Thoughts:")
    tot = TreeOfThoughts(num_thoughts=3)
    prompt = tot.create_prompt("What is the best investment strategy?")
    print(f"Prompt: {prompt[:100]}...")
    
    # ReAct
    print("\n3. ReAct:")
    react = ReAct(tools=["search", "calculate", "lookup"])
    prompt = react.create_prompt("What is the population of Tokyo?")
    print(f"Prompt: {prompt[:100]}...")
    
    # Self-Consistency
    print("\n4. Self-Consistency:")
    sc = SelfConsistency(num_samples=5)
    prompt = sc.create_prompt("What is 2+2?")
    print(f"Prompt: {prompt}")
    
    # Prompt Optimizer
    print("\n5. Prompt Optimizer:")
    opt = PromptOptimizer()
    optimized = opt.optimize(
        "Explain quantum computing",
        "The explanation was unclear"
    )
    print(f"Optimized: {optimized}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
