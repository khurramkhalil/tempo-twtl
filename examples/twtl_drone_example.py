import sys
import os
import time
import random
from typing import List, Dict, Tuple, Set

# Import from selp_twtl package
import time
import random
from typing import List, Dict, Tuple, Set

from selp_twtl.twtl.grammar import Atomic, Conjunction, WithinTime, AlwaysDuringTime, Sequence
from selp_twtl.twtl.automaton import TimedAutomaton
from selp_twtl.translator.nl_to_twtl import NLtoTWTL
from selp_twtl.planner.constrained_decoding import TWTLConstrainedDecoding, ActionInfo
from selp_twtl.monitoring.execution_monitor import TWTLExecutionMonitor


def create_drone_action_info() -> Dict[str, ActionInfo]:
    """Create action information for a drone navigation task."""
    action_info = {}
    
    # Define rooms and landmarks
    rooms = ["room_A", "room_B", "room_C", "room_D", "room_E"]
    landmarks = ["landmark_1", "landmark_2", "landmark_3"]
    
    # Create navigation actions
    for room in rooms:
        action_info[f"goto_{room}"] = ActionInfo(
            name=f"goto_{room}",
            props={f"visit_{room}", "drone_moving"},
            duration=random.uniform(3.0, 5.0)  # Random duration between 3-5 seconds
        )
    
    for landmark in landmarks:
        action_info[f"goto_{landmark}"] = ActionInfo(
            name=f"goto_{landmark}",
            props={f"visit_{landmark}", "drone_moving"},
            duration=random.uniform(2.0, 4.0)  # Random duration between 2-4 seconds
        )
    
    # Special actions
    action_info["take_photo"] = ActionInfo(
        name="take_photo",
        props={"photo_taken", "drone_stationary"},
        duration=1.0
    )
    
    action_info["scan_area"] = ActionInfo(
        name="scan_area",
        props={"area_scanned", "drone_moving"},
        duration=random.uniform(4.0, 6.0)  # Random duration between 4-6 seconds
    )
    
    return action_info


def create_example_twtl_formula():
    """Create an example TWTL formula programmatically."""
    # Visit room A within 10-20 seconds
    visit_a = WithinTime(Atomic("visit_room_A"), 10, 20)
    
    # Then visit room B within 15-30 seconds
    visit_b = WithinTime(Atomic("visit_room_B"), 15, 30)
    
    # Then stay in room C for 5-10 seconds
    stay_c = AlwaysDuringTime(Atomic("visit_room_C"), 5, 10)
    
    # The complete formula is a sequence of these three requirements
    formula = Sequence(visit_a, Sequence(visit_b, stay_c))
    
    return formula


def create_example_automaton(formula) -> TimedAutomaton:
    """Create a timed automaton from a TWTL formula."""
    return formula.to_automaton()


def simulate_execution(plan: List[str], action_info: Dict[str, ActionInfo], execution_monitor: TWTLExecutionMonitor):
    """Simulate the execution of a plan."""
    print(f"\nStarting execution of plan: {plan}")
    execution_monitor.start_execution()
    
    for i, action in enumerate(plan):
        # Get the actual action info
        info = action_info.get(action)
        if not info:
            print(f"Unknown action: {action}")
            continue
        
        # Simulate the action execution
        print(f"Executing {action} (expected duration: {info.duration:.2f}s)")
        
        # Simulate some variability in execution time
        variation = random.uniform(0.8, 1.2)  # 80% to 120% of expected time
        actual_duration = info.duration * variation
        
        # Add a failure case with 10% probability
        success = random.random() > 0.1
        
        # Sleep to simulate execution
        time.sleep(actual_duration / 10)  # Divide by 10 to speed up the example
        
        # Report completion
        print(f"  {'Completed' if success else 'Failed'} {action} in {actual_duration:.2f}s")
        
        # Update the execution monitor
        status = execution_monitor.step_completed(i, success)
        
        # Print execution status
        print(f"  Status: {status.constraint_status}, {'ahead' if status.time_margin > 0 else 'behind'} by {abs(status.time_margin):.2f}s")
        
        if not success:
            print(f"  Action failed, replanning...")
            # The execution monitor should handle replanning
        
        # If the plan was adapted, get the new plan
        current_plan = execution_monitor.get_current_plan()
        if current_plan != plan:
            print(f"  Plan adapted: {current_plan}")
            plan = current_plan
    
    # Get execution summary
    summary = execution_monitor.get_execution_summary()
    print("\nExecution Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def replan(remaining_plan: List[str], current_step: int, elapsed_time: float) -> List[str]:
    """
    Simple replanning function.
    
    In a real implementation, this would use the constrained decoder to generate a new plan.
    For this example, we just return a modified version of the remaining plan.
    """
    print(f"Replanning from step {current_step} at time {elapsed_time:.2f}s")
    print(f"Original remaining plan: {remaining_plan}")
    
    # For this example, just swap the order of the first two remaining actions if possible
    if len(remaining_plan) >= 2:
        new_plan = [remaining_plan[1], remaining_plan[0]] + remaining_plan[2:]
    else:
        new_plan = list(remaining_plan)  # Just copy the original
    
    print(f"New plan: {new_plan}")
    return new_plan


def main():
    """Run the example."""
    print("TWTL-Enhanced SELP Example for Drone Navigation")
    print("===============================================\n")
    
    # 1. Create action information
    action_info = create_drone_action_info()
    print(f"Available actions: {list(action_info.keys())}")
    
    # 2. Parse a natural language command to TWTL
    nl_command = "Visit room A within 10-20 seconds, then visit room B within 15-30 seconds, then stay in room C for 5-10 seconds."
    translator = NLtoTWTL()
    twtl_formula_str = translator.translate(nl_command)
    print(f"\nNL Command: {nl_command}")
    print(f"Translated TWTL: {twtl_formula_str}")
    
    # 3. Create a TWTL formula and automaton (using the programmatic approach for the example)
    formula = create_example_twtl_formula()
    automaton = create_example_automaton(formula)
    print(f"\nTWTL Formula: {formula}")
    print(f"Automaton created with {len(automaton.states)} states and {len(automaton.transitions)} transitions")
    
    # 4. Create a constrained decoder
    decoder = TWTLConstrainedDecoding(automaton, action_info)
    
    # 5. Generate a plan using constrained decoding
    # In a real implementation, this would use an LLM with the constrained decoder
    # For this example, we'll create a simple predefined plan
    plan = [
        "goto_room_A",  # Visit room A first (within 10-20 seconds)
        "take_photo",
        "goto_room_B",  # Then visit room B (within 15-30 seconds)
        "scan_area",
        "goto_room_C",  # Finally stay in room C (for 5-10 seconds)
        "take_photo"
    ]
    
    # Check if the plan is valid
    is_valid = decoder.is_plan_valid(plan)
    efficiency = decoder.get_plan_efficiency(plan)
    print(f"\nGenerated Plan: {plan}")
    print(f"Plan validity: {is_valid}")
    print(f"Plan efficiency (total duration): {efficiency:.2f}s")
    
    # 6. Create execution monitor with replanning
    expected_durations = {action: info.duration for action, info in action_info.items()}
    execution_monitor = TWTLExecutionMonitor(
        constrained_decoder=decoder,
        original_plan=plan,
        expected_durations=expected_durations,
        replanner=replan
    )
    
    # 7. Simulate plan execution
    simulate_execution(plan, action_info, execution_monitor)


if __name__ == "__main__":
    main()