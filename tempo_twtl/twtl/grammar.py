"""
Time Window Temporal Logic (TWTL) grammar definition.
Extends Linear Temporal Logic (LTL) with time window operators.
"""

class TWTLFormula:
    """Base class for all TWTL formulas."""
    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__")
    
    def to_automaton(self):
        """Convert the formula to a timed automaton."""
        raise NotImplementedError("Subclasses must implement to_automaton")


class Atomic(TWTLFormula):
    """Atomic proposition (e.g., 'in_room_A')."""
    def __init__(self, prop):
        self.prop = prop
    
    def __str__(self):
        return self.prop
    
    def to_automaton(self):
        from .automaton import create_atomic_automaton
        return create_atomic_automaton(self.prop)


class Negation(TWTLFormula):
    """Negation operator (e.g., '!in_room_A')."""
    def __init__(self, subformula):
        self.subformula = subformula
    
    def __str__(self):
        return f"!({self.subformula})"
    
    def to_automaton(self):
        from .automaton import negate_automaton
        sub_auto = self.subformula.to_automaton()
        return negate_automaton(sub_auto)


class Conjunction(TWTLFormula):
    """Conjunction operator (e.g., 'in_room_A & in_room_B')."""
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"({self.left}) & ({self.right})"
    
    def to_automaton(self):
        from .automaton import intersect_automata
        left_auto = self.left.to_automaton()
        right_auto = self.right.to_automaton()
        return intersect_automata(left_auto, right_auto)


class Disjunction(TWTLFormula):
    """Disjunction operator (e.g., 'in_room_A | in_room_B')."""
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"({self.left}) | ({self.right})"
    
    def to_automaton(self):
        from .automaton import union_automata
        left_auto = self.left.to_automaton()
        right_auto = self.right.to_automaton()
        return union_automata(left_auto, right_auto)


class Eventually(TWTLFormula):
    """Eventually operator (e.g., 'F in_room_A')."""
    def __init__(self, subformula):
        self.subformula = subformula
    
    def __str__(self):
        return f"F({self.subformula})"
    
    def to_automaton(self):
        from .automaton import eventually_automaton
        sub_auto = self.subformula.to_automaton()
        return eventually_automaton(sub_auto)


class Always(TWTLFormula):
    """Always operator (e.g., 'G in_room_A')."""
    def __init__(self, subformula):
        self.subformula = subformula
    
    def __str__(self):
        return f"G({self.subformula})"
    
    def to_automaton(self):
        from .automaton import always_automaton
        sub_auto = self.subformula.to_automaton()
        return always_automaton(sub_auto)


class Until(TWTLFormula):
    """Until operator (e.g., 'in_room_A U in_room_B')."""
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"({self.left}) U ({self.right})"
    
    def to_automaton(self):
        from .automaton import until_automaton
        left_auto = self.left.to_automaton()
        right_auto = self.right.to_automaton()
        return until_automaton(left_auto, right_auto)


# TWTL-specific operators with time windows

class WithinTime(TWTLFormula):
    """
    Within time window operator.
    F[a,b] φ means φ must hold at least once between time a and b.
    """
    def __init__(self, subformula, lower_bound, upper_bound):
        self.subformula = subformula
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def __str__(self):
        return f"F[{self.lower_bound},{self.upper_bound}]({self.subformula})"
    
    def to_automaton(self):
        from .automaton import within_time_automaton
        sub_auto = self.subformula.to_automaton()
        return within_time_automaton(sub_auto, self.lower_bound, self.upper_bound)


class AlwaysDuringTime(TWTLFormula):
    """
    Always during time window operator.
    G[a,b] φ means φ must hold continuously between time a and b.
    """
    def __init__(self, subformula, lower_bound, upper_bound):
        self.subformula = subformula
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def __str__(self):
        return f"G[{self.lower_bound},{self.upper_bound}]({self.subformula})"
    
    def to_automaton(self):
        from .automaton import always_during_time_automaton
        sub_auto = self.subformula.to_automaton()
        return always_during_time_automaton(sub_auto, self.lower_bound, self.upper_bound)


class Sequence(TWTLFormula):
    """
    Sequential composition with time windows.
    e.g., visit_A:[0,10]; visit_B:[5,15] means first visit_A within [0,10], then visit_B within [5,15].
    """
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    def __str__(self):
        return f"{self.first}; {self.second}"
    
    def to_automaton(self):
        from .automaton import sequence_automaton
        first_auto = self.first.to_automaton()
        second_auto = self.second.to_automaton()
        return sequence_automaton(first_auto, second_auto)