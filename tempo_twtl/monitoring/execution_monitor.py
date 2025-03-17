"""
Online execution monitoring and adaptation for TWTL-constrained plans.
"""
import time
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass

# Import TWTL components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twtl.automaton import TimedAutomaton
from planner.constrained_decoding import TWTLConstrainedDecoding, ActionInfo


@dataclass
class ExecutionStatus:
    """Status of plan execution."""
    current_step: int
    current_time: float
    is_on_schedule: bool
    time_margin: float  # Positive means ahead of schedule, negative means behind
    constraint_status: str  # 'satisfied', 'at_risk', 'violated'
    critical_constraints: List[str]  # Constraints that are at risk


class TWTLExecutionMonitor:
    """
    Monitors the execution of a plan against TWTL constraints and provides adaptation.
    """
    
    def __init__(self, constrained_decoder: TWTLConstrainedDecoding, 
                original_plan: List[str], expected_durations: Dict[str, float],
                replanner: Optional[Callable[[List[str], int, float], List[str]]] = None):
        """
        Initialize the execution monitor.
        
        Args:
            constrained_decoder: The TWTL constrained decoding module.
            original_plan: The original plan to execute.
            expected_durations: Expected durations for each action.
            replanner: Optional function to replan when necessary.
        """
        self.decoder = constrained_decoder
        self.original_plan = original_plan
        self.expected_durations = expected_durations
        self.replanner = replanner
        
        # Initialize execution state
        self.current_step = 0
        self.execution_start_time = None
        self.step_start_times = {}
        self.actual_durations = {}
        self.current_plan = list(original_plan)  # Copy the original plan
        
        # Calculate expected completion times for each step
        self.expected_completion_times = self._calculate_expected_times(original_plan)
    
    def _calculate_expected_times(self, plan: List[str]) -> Dict[int, float]:
        """Calculate expected completion times for each step in the plan."""
        expected_times = {}
        cumulative_time = 0.0
        
        for i, action in enumerate(plan):
            duration = self.expected_durations.get(action, 1.0)  # Default to 1.0 if unknown
            cumulative_time += duration
            expected_times[i] = cumulative_time
        
        return expected_times
    
    def start_execution(self):
        """Start monitoring the plan execution."""
        self.execution_start_time = time.time()
        self.step_start_times[0] = self.execution_start_time
        print(f"Starting execution of plan: {self.current_plan}")
    
    def step_completed(self, step_index: int, success: bool = True) -> ExecutionStatus:
        """
        Record the completion of a step and check execution status.
        
        Args:
            step_index: Index of the completed step.
            success: Whether the step completed successfully.
            
        Returns:
            Current execution status.
        """
        if self.execution_start_time is None:
            raise ValueError("Execution not started. Call start_execution() first.")
        
        current_time = time.time()
        elapsed_time = current_time - self.execution_start_time
        
        # Record actual duration
        if step_index in self.step_start_times:
            duration = current_time - self.step_start_times[step_index]
            self.actual_durations[step_index] = duration
        
        # Check if we need to replan
        status = self._check_execution_status(step_index, elapsed_time, success)
        
        if status.constraint_status == 'violated' or (status.constraint_status == 'at_risk' and self.replanner):
            print(f"Need to replan: {status.constraint_status}, time margin: {status.time_margin}")
            self._adapt_plan(step_index, elapsed_time)
        
        # Move to the next step
        if success:
            self.current_step = step_index + 1
            if self.current_step < len(self.current_plan):
                self.step_start_times[self.current_step] = current_time
        
        return status
    
    def _check_execution_status(self, step_index: int, elapsed_time: float, success: bool) -> ExecutionStatus:
        """
        Check the execution status against TWTL constraints.
        
        Args:
            step_index: Index of the current step.
            elapsed_time: Elapsed time since execution started.
            success: Whether the step completed successfully.
            
        Returns:
            Current execution status.
        """
        # Expected time to reach this point
        expected_time = self.expected_completion_times.get(step_index, 0.0)
        
        # Calculate time margin
        time_margin = expected_time - elapsed_time  # Positive means ahead of schedule
        
        # Analyze constraint status
        constraint_status = 'satisfied'
        critical_constraints = []
        
        # Reset decoder and simulate execution up to this point
        self.decoder.reset()
        for i in range(min(step_index + 1, len(self.current_plan))):
            if not self.decoder.apply_action(self.current_plan[i]):
                # A constraint has been violated
                constraint_status = 'violated'
                critical_constraints.append(f"Action {self.current_plan[i]} at step {i}")
                break
        
        # Check if we're at risk of violating timing constraints
        if constraint_status != 'violated' and time_margin < 0:
            # We're behind schedule, check if any timing constraints are at risk
            # This requires detailed analysis of the remaining plan against TWTL constraints
            
            # Simplified check: if we're more than 20% behind schedule, mark as at risk
            if abs(time_margin) > 0.2 * expected_time:
                constraint_status = 'at_risk'
                critical_constraints.append(f"Timing constraint for plan completion")
        
        return ExecutionStatus(
            current_step=step_index,
            current_time=elapsed_time,
            is_on_schedule=time_margin >= 0,
            time_margin=time_margin,
            constraint_status=constraint_status,
            critical_constraints=critical_constraints
        )
    
    def _adapt_plan(self, current_step: int, elapsed_time: float):
        """
        Adapt the plan when constraints are at risk or violated.
        
        Args:
            current_step: Current step in the plan.
            elapsed_time: Elapsed time since execution started.
        """
        if self.replanner is None:
            print("Replanning needed but no replanner provided.")
            return
        
        # Get the remaining steps in the plan
        remaining_plan = self.current_plan[current_step+1:]
        
        # Call the replanner to get a new plan for the remaining steps
        new_remaining_plan = self.replanner(remaining_plan, current_step+1, elapsed_time)
        
        if new_remaining_plan:
            # Update the current plan
            self.current_plan = self.current_plan[:current_step+1] + new_remaining_plan
            
            # Recalculate expected completion times
            new_expected_times = self._calculate_expected_times(self.current_plan)
            
            # Adjust expected times based on actual elapsed time
            adjustment = elapsed_time - self.expected_completion_times.get(current_step, 0.0)
            for step in new_expected_times:
                if step > current_step:
                    new_expected_times[step] += adjustment
            
            self.expected_completion_times = new_expected_times
            
            print(f"Plan adapted: {self.current_plan}")
        else:
            print("Replanning failed, continuing with current plan.")
    
    def get_current_plan(self) -> List[str]:
        """Get the current (possibly adapted) plan."""
        return self.current_plan
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution so far."""
        if self.execution_start_time is None:
            return {"status": "not_started"}
        
        current_time = time.time()
        elapsed_time = current_time - self.execution_start_time
        
        # Check if execution is complete
        is_complete = self.current_step >= len(self.current_plan)
        
        # Calculate average time deviation
        time_deviations = []
        for step, actual_time in self.actual_durations.items():
            expected_time = self.expected_durations.get(self.current_plan[step], 1.0)
            time_deviations.append(actual_time - expected_time)
        
        avg_deviation = sum(time_deviations) / len(time_deviations) if time_deviations else 0
        
        return {
            "status": "complete" if is_complete else "in_progress",
            "current_step": self.current_step,
            "total_steps": len(self.current_plan),
            "elapsed_time": elapsed_time,
            "expected_duration": self.expected_completion_times.get(len(self.current_plan) - 1, 0),
            "time_deviation": avg_deviation,
            "adapted": self.current_plan != self.original_plan,
            "constraint_violations": self.decoder.current_state not in self.decoder.automaton.accepting_states
        }