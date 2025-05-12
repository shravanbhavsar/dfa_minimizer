import random

def generate_random_dfa(n_states, alphabet_size):
    alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]
    states = list(range(n_states))
    transitions = {s: {} for s in states}
    for s in states:
        for sym in alphabet:
            transitions[s][sym] = random.choice(states)
    start_state = 0
    final_states = set()
    if n_states > 0:
        # pick at least one final
        finals = random.randint(1, n_states)
        final_states = set(random.sample(states, finals))
    # Trim unreachable
    reachable = set()
    stack = [start_state]
    while stack:
        s = stack.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for sym in alphabet:
            t = transitions[s][sym]
            if t not in reachable:
                stack.append(t)
    states = [s for s in states if s in reachable]
    final_states = {s for s in final_states if s in reachable}
    transitions = {s: transitions[s] for s in states}
    return states, alphabet, start_state, final_states, transitions
