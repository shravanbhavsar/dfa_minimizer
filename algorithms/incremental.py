from algorithms.hopcroft import hopcroft_minimize

def compute_partitions(states, alphabet, final_states, transitions):
    """
    Compute the current partitioning of states (equivalence classes)
    using the same refinement logic as Hopcroft’s algorithm, but
    return only the list of sets P (the partitions).
    """
    # 1. Remove unreachable states
    reachable = set()
    stack = [states[0]]
    while stack:
        s = stack.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for sym in alphabet:
            if s in transitions and sym in transitions[s]:
                stack.append(transitions[s][sym])
    states = [s for s in states if s in reachable]
    final_states = {s for s in final_states if s in reachable}

    # 2. Initial partition: final vs non-final
    P = []
    if final_states:
        P.append(set(final_states))
    nonfinal = set(states) - final_states
    if nonfinal:
        P.append(nonfinal)
    W = P.copy()

    # 3. Refinement loop
    while W:
        A = W.pop()
        for sym in alphabet:
            # X = states that transition on sym into A
            X = {
                q for q in states
                if q in transitions
                   and sym in transitions[q]
                   and transitions[q][sym] in A
            }
            if not X:
                continue
            new_P = []
            for Y in P:
                if not (X & Y) or not (Y - X):
                    new_P.append(Y)
                else:
                    Y1 = Y & X
                    Y2 = Y - X
                    new_P.append(Y1)
                    new_P.append(Y2)
                    # Update worklist
                    if Y in W:
                        W.remove(Y)
                        W.append(Y1 if len(Y1) < len(Y2) else Y2)
                    else:
                        W.append(Y1 if len(Y1) < len(Y2) else Y2)
            P = new_P

    return P

class IncrementalMinimizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        # Minimal DFA state set, start, finals, transitions
        self.states = [0]
        self.start_state = 0
        self.final_states = set()
        self.transitions = {0: {sym: 0 for sym in alphabet}}
        # Compute initial partition: only one state
        self.partitions = [{0}]

    def add_state(self, is_final=False):
        """
        Add a new state (unconnected initially) to the DFA.
        """
        new_st = len(self.states)
        self.states.append(new_st)
        if is_final:
            self.final_states.add(new_st)
        # Initialize transitions dict for the new state
        self.transitions[new_st] = {}
        # In the un-minimized view, it starts in its own block
        self.partitions.append({new_st})
        return new_st

    def add_transition(self, frm, sym, to):
        """
        Add/update a transition and then re-minimize and recompute partitions.
        """
        # 1. Update transition
        self.transitions[frm][sym] = to

        # 2. Re-minimize the DFA from scratch for correctness
        sts, st0, fins, trans = hopcroft_minimize(
            self.states,
            self.alphabet,
            self.start_state,
            self.final_states,
            self.transitions
        )
        self.states = sts
        self.start_state = st0
        self.final_states = fins
        self.transitions = trans

        # 3. Fully recompute the partition list on the new minimal DFA
        self.partitions = compute_partitions(
            self.states,
            self.alphabet,
            self.final_states,
            self.transitions
        )
