"""
TWTL to timed automaton conversion utilities.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union


@dataclass
class State:
    """A state in a timed automaton."""
    id: int
    is_accepting: bool = False
    time_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    

@dataclass
class Transition:
    """A transition in a timed automaton."""
    source: int
    target: int
    guard: str  # Logical guard expression
    time_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    reset_clocks: List[str] = field(default_factory=list)


class TimedAutomaton:
    """
    A timed automaton representation for TWTL formulas.
    """
    def __init__(self):
        self.states: Dict[int, State] = {}
        self.transitions: List[Transition] = []
        self.initial_state: Optional[int] = None
        self.accepting_states: Set[int] = set()
        self.clocks: Set[str] = set()
        self.next_state_id = 0
    
    def add_state(self, is_accepting: bool = False, 
                 time_constraints: Dict[str, Tuple[float, float]] = None) -> int:
        """Add a state to the automaton and return its ID."""
        state_id = self.next_state_id
        self.next_state_id += 1
        
        if time_constraints:
            for clock in time_constraints:
                self.clocks.add(clock)
        
        self.states[state_id] = State(
            id=state_id,
            is_accepting=is_accepting,
            time_constraints=time_constraints or {}
        )
        
        if is_accepting:
            self.accepting_states.add(state_id)
            
        return state_id
    
    def set_initial_state(self, state_id: int):
        """Set the initial state of the automaton."""
        if state_id not in self.states:
            raise ValueError(f"State {state_id} does not exist")
        self.initial_state = state_id
    
    def add_transition(self, source: int, target: int, guard: str,
                      time_constraints: Dict[str, Tuple[float, float]] = None,
                      reset_clocks: List[str] = None):
        """Add a transition to the automaton."""
        if source not in self.states:
            raise ValueError(f"Source state {source} does not exist")
        if target not in self.states:
            raise ValueError(f"Target state {target} does not exist")
        
        if time_constraints:
            for clock in time_constraints:
                self.clocks.add(clock)
        
        if reset_clocks:
            for clock in reset_clocks:
                self.clocks.add(clock)
        
        self.transitions.append(Transition(
            source=source,
            target=target,
            guard=guard,
            time_constraints=time_constraints or {},
            reset_clocks=reset_clocks or []
        ))
    
    def check_timed_run(self, run: List[Tuple[str, float]]) -> bool:
        """
        Check if a timed run satisfies the automaton.
        
        Args:
            run: List of (proposition, time) pairs representing system states at different times.
            
        Returns:
            bool: True if the run satisfies the automaton's acceptance condition.
        """
        if self.initial_state is None:
            raise ValueError("Initial state not set")
        
        current_state = self.initial_state
        clock_values = {clock: 0.0 for clock in self.clocks}
        last_time = 0.0
        
        for prop, time in run:
            # Update clock values
            time_delta = time - last_time
            for clock in clock_values:
                clock_values[clock] += time_delta
            
            # Check available transitions
            valid_transitions = []
            for transition in self.transitions:
                if transition.source != current_state:
                    continue
                
                # Check proposition guard
                if not self._evaluate_guard(transition.guard, prop):
                    continue
                
                # Check time constraints
                if not self._check_time_constraints(transition.time_constraints, clock_values):
                    continue
                
                valid_transitions.append(transition)
            
            if not valid_transitions:
                return False  # No valid transition available
            
            # Take the first valid transition (deterministic automaton)
            transition = valid_transitions[0]
            current_state = transition.target
            
            # Reset clocks as specified by the transition
            for clock in transition.reset_clocks:
                clock_values[clock] = 0.0
            
            last_time = time
        
        # Check if we ended in an accepting state
        return current_state in self.accepting_states
    
    def _evaluate_guard(self, guard: str, proposition: str) -> bool:
        """Evaluate a guard expression against a proposition."""
        # Simple implementation - in practice, use a proper parser
        if guard == "true":
            return True
        if guard == "false":
            return False
        
        # Handle negation
        if guard.startswith("!"):
            negated_guard = guard[1:]
            return negated_guard != proposition
        
        # Direct match
        return guard == proposition
    
    def _check_time_constraints(self, constraints: Dict[str, Tuple[float, float]], 
                              clock_values: Dict[str, float]) -> bool:
        """Check if clock values satisfy time constraints."""
        for clock, (lower, upper) in constraints.items():
            if clock not in clock_values:
                return False
            
            value = clock_values[clock]
            if value < lower or value > upper:
                return False
        
        return True


# Factory functions for creating automata from TWTL formulas

def create_atomic_automaton(prop: str) -> TimedAutomaton:
    """Create an automaton for an atomic proposition."""
    automaton = TimedAutomaton()
    
    # Create two states: initial (non-accepting) and final (accepting)
    initial = automaton.add_state(is_accepting=False)
    final = automaton.add_state(is_accepting=True)
    
    # Set initial state
    automaton.set_initial_state(initial)
    
    # Add transition on the proposition
    automaton.add_transition(initial, final, guard=prop)
    
    return automaton


def within_time_automaton(sub_automaton: TimedAutomaton, lower: float, upper: float) -> TimedAutomaton:
    """
    Create an automaton for the F[lower,upper] operator.
    
    This represents the formula: "eventually within time window [lower,upper], subformula holds".
    """
    automaton = TimedAutomaton()
    
    # Create a new clock for tracking time
    clock = f"c{len(automaton.clocks)}"
    
    # Create initial state
    initial = automaton.add_state(is_accepting=False)
    automaton.set_initial_state(initial)
    
    # Create a copy of all states from sub_automaton
    state_map = {}
    for state_id, state in sub_automaton.states.items():
        new_state_id = automaton.add_state(is_accepting=state.is_accepting)
        state_map[state_id] = new_state_id
    
    # Copy all transitions
    for transition in sub_automaton.transitions:
        automaton.add_transition(
            source=state_map[transition.source],
            target=state_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    # Add transition from initial to sub-automaton initial with clock reset
    automaton.add_transition(
        source=initial,
        target=state_map[sub_automaton.initial_state],
        guard="true",
        reset_clocks=[clock]
    )
    
    # Add time constraints to accepting states
    for state_id in automaton.accepting_states:
        state = automaton.states[state_id]
        state.time_constraints[clock] = (lower, upper)
    
    return automaton


def always_during_time_automaton(sub_automaton: TimedAutomaton, lower: float, upper: float) -> TimedAutomaton:
    """
    Create an automaton for the G[lower,upper] operator.
    
    This represents the formula: "always during time window [lower,upper], subformula holds".
    """
    # This is a simplified implementation
    automaton = TimedAutomaton()
    
    # Create a new clock for tracking time
    clock = f"c{len(automaton.clocks)}"
    
    # Create states
    initial = automaton.add_state(is_accepting=False)
    monitoring = automaton.add_state(is_accepting=False)
    violation = automaton.add_state(is_accepting=False)
    accepting = automaton.add_state(is_accepting=True)
    
    automaton.set_initial_state(initial)
    
    # Transition to monitoring state at lower bound with clock reset
    automaton.add_transition(
        source=initial,
        target=monitoring,
        guard="true",
        time_constraints={clock: (lower, lower)},
        reset_clocks=[clock]
    )
    
    # Add self-loop in monitoring state for when sub-formula holds
    for transition in sub_automaton.transitions:
        if transition.source == sub_automaton.initial_state and transition.target in sub_automaton.accepting_states:
            automaton.add_transition(
                source=monitoring,
                target=monitoring,
                guard=transition.guard,
                time_constraints={clock: (0, upper - lower)}
            )
    
    # Transition to violation state if sub-formula doesn't hold during monitoring
    for state_id in sub_automaton.states:
        if state_id not in sub_automaton.accepting_states:
            automaton.add_transition(
                source=monitoring,
                target=violation,
                guard="true",
                time_constraints={clock: (0, upper - lower)}
            )
    
    # Transition to accepting state after upper bound
    automaton.add_transition(
        source=monitoring,
        target=accepting,
        guard="true",
        time_constraints={clock: (upper - lower, upper - lower)}
    )
    
    return automaton


def sequence_automaton(first_automaton: TimedAutomaton, second_automaton: TimedAutomaton) -> TimedAutomaton:
    """
    Create an automaton for sequential composition.
    
    This represents "first; second" - complete first before starting second.
    """
    automaton = TimedAutomaton()
    
    # Map states from first automaton
    first_map = {}
    for state_id, state in first_automaton.states.items():
        new_state_id = automaton.add_state(is_accepting=False)  # None are accepting yet
        first_map[state_id] = new_state_id
    
    # Map states from second automaton
    second_map = {}
    for state_id, state in second_automaton.states.items():
        new_state_id = automaton.add_state(is_accepting=state.is_accepting)
        second_map[state_id] = new_state_id
    
    # Set initial state from first automaton
    automaton.set_initial_state(first_map[first_automaton.initial_state])
    
    # Copy transitions from first automaton
    for transition in first_automaton.transitions:
        automaton.add_transition(
            source=first_map[transition.source],
            target=first_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    # Copy transitions from second automaton
    for transition in second_automaton.transitions:
        automaton.add_transition(
            source=second_map[transition.source],
            target=second_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    # Add transitions from first accepting states to second initial state
    for state_id in first_automaton.accepting_states:
        automaton.add_transition(
            source=first_map[state_id],
            target=second_map[second_automaton.initial_state],
            guard="true"
        )
    
    return automaton


# Other automaton operations (union, intersection, etc.)

def intersect_automata(a: TimedAutomaton, b: TimedAutomaton) -> TimedAutomaton:
    """Create the intersection (product) of two automata."""
    # Simplified implementation for PoC
    result = TimedAutomaton()
    
    # Create product state space
    state_map = {}
    for a_state in a.states:
        for b_state in b.states:
            product_state = (a_state, b_state)
            is_accepting = a.states[a_state].is_accepting and b.states[b_state].is_accepting
            
            # Combine time constraints
            time_constraints = {}
            time_constraints.update(a.states[a_state].time_constraints)
            time_constraints.update(b.states[b_state].time_constraints)
            
            state_id = result.add_state(is_accepting=is_accepting, time_constraints=time_constraints)
            state_map[product_state] = state_id
            
            # Set initial state
            if a_state == a.initial_state and b_state == b.initial_state:
                result.set_initial_state(state_id)
    
    # Create product transitions
    for a_trans in a.transitions:
        for b_trans in b.transitions:
            if a_trans.guard == b_trans.guard:  # Guards must match
                source = state_map[(a_trans.source, b_trans.source)]
                target = state_map[(a_trans.target, b_trans.target)]
                
                # Combine time constraints
                time_constraints = {}
                time_constraints.update(a_trans.time_constraints)
                time_constraints.update(b_trans.time_constraints)
                
                # Combine clock resets
                reset_clocks = list(set(a_trans.reset_clocks + b_trans.reset_clocks))
                
                result.add_transition(
                    source=source,
                    target=target,
                    guard=a_trans.guard,
                    time_constraints=time_constraints,
                    reset_clocks=reset_clocks
                )
    
    return result


def union_automata(a: TimedAutomaton, b: TimedAutomaton) -> TimedAutomaton:
    """Create the union of two automata."""
    # Simplified implementation for PoC
    result = TimedAutomaton()
    
    # Create new initial state
    initial = result.add_state(is_accepting=False)
    result.set_initial_state(initial)
    
    # Map states from automaton A
    a_map = {}
    for state_id, state in a.states.items():
        new_id = result.add_state(is_accepting=state.is_accepting, 
                                time_constraints=state.time_constraints)
        a_map[state_id] = new_id
    
    # Map states from automaton B
    b_map = {}
    for state_id, state in b.states.items():
        new_id = result.add_state(is_accepting=state.is_accepting,
                                time_constraints=state.time_constraints)
        b_map[state_id] = new_id
    
    # Add epsilon transitions from initial to both automata's initial states
    result.add_transition(initial, a_map[a.initial_state], guard="true")
    result.add_transition(initial, b_map[b.initial_state], guard="true")
    
    # Copy transitions from automaton A
    for transition in a.transitions:
        result.add_transition(
            source=a_map[transition.source],
            target=a_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    # Copy transitions from automaton B
    for transition in b.transitions:
        result.add_transition(
            source=b_map[transition.source],
            target=b_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    return result


def negate_automaton(a: TimedAutomaton) -> TimedAutomaton:
    """Create the negation of an automaton (complement)."""
    # This is a simplified implementation - proper complementation of timed automata is complex
    result = TimedAutomaton()
    
    # Copy all states but flip accepting status
    state_map = {}
    for state_id, state in a.states.items():
        new_id = result.add_state(
            is_accepting=not state.is_accepting,
            time_constraints=state.time_constraints
        )
        state_map[state_id] = new_id
    
    # Set initial state
    result.set_initial_state(state_map[a.initial_state])
    
    # Copy all transitions
    for transition in a.transitions:
        result.add_transition(
            source=state_map[transition.source],
            target=state_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    return result


# Factory functions for other TWTL operators
def eventually_automaton(sub_automaton: TimedAutomaton) -> TimedAutomaton:
    """Create an automaton for the 'eventually' operator (F φ)."""
    return within_time_automaton(sub_automaton, 0, float('inf'))


def always_automaton(sub_automaton: TimedAutomaton) -> TimedAutomaton:
    """Create an automaton for the 'always' operator (G φ)."""
    return always_during_time_automaton(sub_automaton, 0, float('inf'))


def until_automaton(left_automaton: TimedAutomaton, right_automaton: TimedAutomaton) -> TimedAutomaton:
    """Create an automaton for the 'until' operator (φ U ψ)."""
    # Simplified implementation
    result = TimedAutomaton()
    
    # Create states
    initial = result.add_state(is_accepting=False)
    result.set_initial_state(initial)
    
    # Add right automaton (accepting part)
    right_map = {}
    for state_id, state in right_automaton.states.items():
        new_id = result.add_state(is_accepting=state.is_accepting,
                                time_constraints=state.time_constraints)
        right_map[state_id] = new_id
    
    # Copy right automaton transitions
    for transition in right_automaton.transitions:
        result.add_transition(
            source=right_map[transition.source],
            target=right_map[transition.target],
            guard=transition.guard,
            time_constraints=transition.time_constraints,
            reset_clocks=transition.reset_clocks
        )
    
    # Add left automaton (non-accepting part with self-loops)
    left_map = {}
    for state_id, state in left_automaton.states.items():
        new_id = result.add_state(is_accepting=False,
                                time_constraints=state.time_constraints)
        left_map[state_id] = new_id
    
    # Copy left automaton transitions
    for transition in left_automaton.transitions:
        if transition.target in left_automaton.accepting_states:
            # If transition leads to accepting state in left, add self-loop
            result.add_transition(
                source=left_map[transition.source],
                target=left_map[transition.source],
                guard=transition.guard,
                time_constraints=transition.time_constraints,
                reset_clocks=transition.reset_clocks
            )
        else:
            # Regular transition
            result.add_transition(
                source=left_map[transition.source],
                target=left_map[transition.target],
                guard=transition.guard,
                time_constraints=transition.time_constraints,
                reset_clocks=transition.reset_clocks
            )
    
    # Connect initial to both automata
    result.add_transition(initial, right_map[right_automaton.initial_state], guard="true")
    result.add_transition(initial, left_map[left_automaton.initial_state], guard="true")
    
    # Connect left accepting states to right initial state
    for state_id in left_automaton.accepting_states:
        result.add_transition(
            source=left_map[state_id],
            target=right_map[right_automaton.initial_state],
            guard="true"
        )
    
    return result