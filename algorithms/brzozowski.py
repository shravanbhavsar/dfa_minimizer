def brzozowski_minimize(states, alphabet, start_state, final_states, transitions):
    # Reverse DFA to NFA
    rev_trans = {}
    for s in states:
        for sym in alphabet:
            t = transitions[s][sym]
            rev_trans.setdefault(t, {}).setdefault(sym, set()).add(s)
    rev_initial_states = set(final_states)
    rev_final_states = {start_state}

    def determinize_nfa(nfa_trans, init_states, nfa_finals):
        from collections import deque
        init_set = frozenset(init_states)
        dfa_states = {init_set: 0}
        dfa_finals = set()
        dfa_trans = {}
        queue = deque([init_set])
        if init_set & nfa_finals:
            dfa_finals.add(0)
        while queue:
            curr = queue.popleft()
            cid = dfa_states[curr]
            dfa_trans[cid] = {}
            for sym in alphabet:
                nxt = set()
                for st in curr:
                    if st in nfa_trans and sym in nfa_trans[st]:
                        nxt |= nfa_trans[st][sym]
                nxt = frozenset(nxt)
                if not nxt:
                    continue
                if nxt not in dfa_states:
                    dfa_states[nxt] = len(dfa_states)
                    queue.append(nxt)
                    if nxt & nfa_finals:
                        dfa_finals.add(dfa_states[nxt])
                dfa_trans[cid][sym] = dfa_states[nxt]
        n = len(dfa_states)
        return list(range(n)), 0, dfa_finals, dfa_trans

    # First reverse + determinize
    s1, st1, f1, t1 = determinize_nfa(rev_trans, rev_initial_states, rev_final_states)
    # Second reverse
    rev2 = {}
    for s in s1:
        for sym in alphabet:
            if s in t1 and sym in t1[s]:
                d = t1[s][sym]
                rev2.setdefault(d, {}).setdefault(sym, set()).add(s)
    rev2_init = set(f1)
    rev2_finals = {st1}
    # Second determinize = minimal DFA
    return determinize_nfa(rev2, rev2_init, rev2_finals)
