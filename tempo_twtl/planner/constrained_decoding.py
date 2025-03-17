"""
Time-aware constrained decoding for enforcing TWTL constraints during plan generation.
"""
import re
import time
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass

# Import TWTL components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twtl.automaton import TimedAutomaton


@dataclass
class ActionInfo:
    """Information about an action in the plan."""
    name: str
    props: Set[str]  # Properties that become true after this action
    duration: float  # Estimated duration to execute this action


class TWTLConstrainedDecoding:
    """
    Enforces TWTL constraints during plan generation using constrained decoding.
    
    This class modifies the token probability distribution from an LLM to ensure
    the generated plan adheres to the TWTL specification.
    """
    
    def __init__(self, automaton: TimedAutomaton, action_info: Dict[str, ActionInfo]):
        """
        Initialize the constrained decoding module.
        
        Args:
            automaton: A timed automaton representing the TWTL specification.
            action_info: Information about available actions and their properties.
        """
        self.automaton = automaton
        self.action_info = action_info
        self.current_state = self.automaton.initial_state
        self.clock_values = {clock: 0.0 for clock in self.automaton.clocks}
        self.elapsed_time = 0.0
        self.action_history = []
    
    def reset(self):
        """Reset the automaton state and clock values."""
        self.current_state = self.automaton.initial_state
        self.clock_values = {clock: 0.0 for clock in self.automaton.clocks}
        self.elapsed_time = 0.0
        self.action_history = []
    
    def get_valid_actions(self) -> Set[str]:
        """
        Get the set of valid actions from the current state.
        
        Returns:
            Set of action names that are valid from the current state.
        """
        valid_actions = set()
        
        for action_name, info in self.action_info.items():
            # Check if this action leads to a valid next state
            if self._is_valid_action(action_name):
                valid_actions.add(action_name)
        
        return valid_actions
    
    def _is_valid_action(self, action_name: str) -> bool:
        """
        Check if an action is valid from the current state.
        
        Args:
            action_name: Name of the action to check.
            
        Returns:
            True if the action is valid, False otherwise.
        """
        if action_name not in self.action_info:
            return False
        
        # Get the properties that become true after this action
        action_props = self.action_info[action_name].props
        
        # Check if there's a valid transition for these properties
        for transition in self.automaton.transitions:
            if transition.source != self.current_state:
                continue
            
            # Check if the guard matches any of the action's properties
            for prop in action_props:
                if self.automaton._evaluate_guard(transition.guard, prop):
                    # Check time constraints
                    if self._check_time_constraints(transition.time_constraints):
                        return True
        
        return False
    
    def _check_time_constraints(self, constraints: Dict[str, Tuple[float, float]]) -> bool:
        """Check if current clock values satisfy time constraints."""
        for clock, (lower, upper) in constraints.items():
            if clock not in self.clock_values:
                return False
            
            value = self.clock_values[clock]
            if value < lower or value > upper:
                return False
        
        return True
    
    def apply_action(self, action_name: str) -> bool:
        """
        Apply an action and update the automaton state.
        
        Args:
            action_name: Name of the action to apply.
            
        Returns:
            True if the action was successfully applied, False otherwise.
        """
        if action_name not in self.action_info:
            return False
        
        # Get the properties and duration of this action
        info = self.action_info[action_name]
        action_props = info.props
        duration = info.duration
        
        # Find a valid transition for these properties
        valid_transition = None
        for transition in self.automaton.transitions:
            if transition.source != self.current_state:
                continue
            
            # Check if the guard matches any of the action's properties
            for prop in action_props:
                if self.automaton._evaluate_guard(transition.guard, prop):
                    # Check time constraints
                    if self._check_time_constraints(transition.time_constraints):
                        valid_transition = transition
                        break
            
            if valid_transition:
                break
        
        if not valid_transition:
            return False
        
        # Update the automaton state
        self.current_state = valid_transition.target
        
        # Update clock values
        self.elapsed_time += duration
        for clock in self.clock_values:
            self.clock_values[clock] += duration
        
        # Reset clocks as specified by the transition
        for clock in valid_transition.reset_clocks:
            self.clock_values[clock] = 0.0
        
        # Record this action
        self.action_history.append((action_name, self.elapsed_time))
        
        return True
    
    def modify_token_probabilities(self, token_ids: List[int], probabilities: List[float], 
                                 id_to_action: Callable[[int], str]) -> List[float]:
        """
        Modify token probabilities to enforce TWTL constraints.
        
        Args:
            token_ids: List of token IDs from the LLM.
            probabilities: Corresponding probabilities for each token.
            id_to_action: Function to convert a token ID to an action name.
            
        Returns:
            Modified probability distribution.
        """
        valid_actions = self.get_valid_actions()
        
        # Create a mask for valid tokens
        mask = [0.0] * len(token_ids)
        
        for i, token_id in enumerate(token_ids):
            action_name = id_to_action(token_id)
            if action_name in valid_actions:
                mask[i] = 1.0
        
        # If no valid actions are available, allow all actions
        # (This is a fallback to prevent getting stuck)
        if sum(mask) == 0.0:
            return probabilities
        
        # Apply the mask to the probabilities
        masked_probs = [p * m for p, m in zip(probabilities, mask)]
        
        # Normalize the masked probabilities
        total = sum(masked_probs)
        if total > 0:
            normalized_probs = [p / total for p in masked_probs]
            return normalized_probs
        else:
            # Fallback to original probabilities if all are masked out
            return probabilities
    
    def is_plan_valid(self, plan: List[str]) -> bool:
        """
        Check if a complete plan satisfies the TWTL constraints.
        
        Args:
            plan: List of action names in the plan.
            
        Returns:
            True if the plan satisfies the constraints, False otherwise.
        """
        # Reset the automaton state
        self.reset()
        
        # Apply each action in the plan
        for action_name in plan:
            if not self.apply_action(action_name):
                return False
        
        # Check if we ended in an accepting state
        return self.current_state in self.automaton.accepting_states
    
    def get_plan_efficiency(self, plan: List[str]) -> float:
        """
        Calculate the efficiency (total duration) of a plan.
        
        Args:
            plan: List of action names in the plan.
            
        Returns:
            Total duration of the plan.
        """
        # Reset the automaton state
        self.reset()
        
        # Apply each action in the plan
        for action_name in plan:
            if not self.apply_action(action_name):
                return float('inf')  # Invalid plan
        
        # Return the total elapsed time
        return self.elapsed_time
    
    def get_next_state_and_progress(self, action_name: str) -> Tuple[Optional[int], Dict[str, float]]:
        """
        Get the next automaton state and updated clock values after applying an action.
        Useful for planning algorithms to simulate the effect of an action.
        
        Args:
            action_name: Name of the action to apply.
            
        Returns:
            Tuple of (next state ID, updated clock values), or (None, {}) if invalid.
        """
        if action_name not in self.action_info:
            return None, {}
        
        # Get the properties and duration of this action
        info = self.action_info[action_name]
        action_props = info.props
        duration = info.duration
        
        # Find a valid transition for these properties
        valid_transition = None
        for transition in self.automaton.transitions:
            if transition.source != self.current_state:
                continue
            
            # Check if the guard matches any of the action's properties
            for prop in action_props:
                if self.automaton._evaluate_guard(transition.guard, prop):
                    # Check time constraints
                    if self._check_time_constraints(transition.time_constraints):
                        valid_transition = transition
                        break
            
            if valid_transition:
                break
        
        if not valid_transition:
            return None, {}
        
        # Calculate next state and clock values
        next_state = valid_transition.target
        next_clocks = dict(self.clock_values)
        
        # Update clock values
        for clock in next_clocks:
            next_clocks[clock] += duration
        
        # Reset clocks as specified by the transition
        for clock in valid_transition.reset_clocks:
            next_clocks[clock] = 0.0
        
        return next_state, next_clocks


# Example of how the constrained decoding would be used with an LLM
class TWTLConstrainedDecodingWrapper:
    """
    Wrapper for integrating TWTL constrained decoding with an LLM for plan generation.
    """
    
    def __init__(self, llm, constrained_decoder: TWTLConstrainedDecoding, 
                token_to_action_map: Dict[int, str]):
        """
        Initialize the wrapper.
        
        Args:
            llm: The language model to use for generation.
            constrained_decoder: The TWTL constrained decoding module.
            token_to_action_map: Mapping from token IDs to action names.
        """
        self.llm = llm
        self.decoder = constrained_decoder
        self.token_to_action_map = token_to_action_map
    
    def _id_to_action(self, token_id: int) -> str:
        """Convert a token ID to an action name."""
        return self.token_to_action_map.get(token_id, "UNKNOWN")
    
    def generate_plan(self, prompt: str, max_steps: int = 10) -> List[str]:
        """
        Generate a plan that satisfies the TWTL constraints.
        
        Args:
            prompt: The planning prompt for the LLM.
            max_steps: Maximum number of steps in the plan.
            
        Returns:
            List of action names that form a valid plan.
        """
        plan = []
        
        # Reset the decoder state
        self.decoder.reset()
        
        # Generate the plan step by step
        for _ in range(max_steps):
            # In a real implementation, this would query the LLM and modify its output distribution
            # For this PoC, we'll simulate it with a simplified approach
            
            # 1. Get valid actions from the current state
            valid_actions = self.decoder.get_valid_actions()
            
            if not valid_actions:
                # No valid actions available - plan is complete or invalid
                break
            
            # 2. In a real implementation, we would get token probabilities from the LLM
            # and modify them using the constrained decoder
            # Here, we'll just select the first valid action
            action = next(iter(valid_actions))
            
            # 3. Apply the action and update the decoder state
            if not self.decoder.apply_action(action):
                # Should not happen if valid_actions is correct
                break
            
            # 4. Add the action to the plan
            plan.append(action)
            
            # 5. Check if we've reached an accepting state
            if self.decoder.current_state in self.decoder.automaton.accepting_states:
                # Plan is complete
                break
        
        return plan