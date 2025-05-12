def hopcroft_minimize(states, alphabet, start_state, final_states, transitions):
    # Remove unreachable states for efficiency
    reachable = set()
    stack = [start_state]
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

    # Initial partition: final vs non-final states
    P = []
    if final_states:
        P.append(set(final_states))
    nonfinal = set(states) - final_states
    if nonfinal:
        P.append(nonfinal)
    W = P.copy()

    # Refinement loop
    while W:
        A = W.pop()
        for sym in alphabet:
            # X = states that transition on sym into A
            X = {q for q in states
                 if q in transitions
                 and sym in transitions[q]
                 and transitions[q][sym] in A}
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
                    # update worklist
                    if Y in W:
                        W.remove(Y)
                        W.append(Y1 if len(Y1) < len(Y2) else Y2)
                    else:
                        W.append(Y1 if len(Y1) < len(Y2) else Y2)
            P = new_P

    # Construct minimized DFA
    rep_to_new = {}
    new_states = []
    new_final_states = set()
    new_transitions = {}
    for i, group in enumerate(P):
        new_states.append(i)
        for st in group:
            rep_to_new[st] = i
        if group & final_states:
            new_final_states.add(i)
    for i, group in enumerate(P):
        rep = next(iter(group))
        new_transitions[i] = {
            sym: rep_to_new[transitions[rep][sym]]
            for sym in alphabet
        }
    new_start_state = rep_to_new[start_state]
    return new_states, new_start_state, new_final_states, new_transitions
