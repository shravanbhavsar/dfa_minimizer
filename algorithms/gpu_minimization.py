import cupy as cp

def gpu_minimize_dfa(states, alphabet, start_state, final_states, transitions):
    """
    GPU‐parallel partition refinement. Expects:
      - states: list of state labels (not necessarily 0..n-1)
      - alphabet: list of symbols
      - transitions: dict mapping state -> symbol -> state label
      - final_states: set of accepting state labels

    We first map labels to 0..n-1 so we can index CuPy arrays correctly.
    """
    # 1) Build a label->index map
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    k = len(alphabet)

    # 2) Build the GPU transition table of shape (n, k) holding indices
    T = cp.empty((n, k), dtype=cp.int32)
    for i, s in enumerate(states):
        for j, sym in enumerate(alphabet):
            dst_label = transitions[s][sym]
            T[i, j] = state_to_idx[dst_label]

    # 3) Initialize partition ID array: 0 for accepting, 1 for non‐accepting
    part = cp.zeros(n, dtype=cp.int32)
    for i, s in enumerate(states):
        if s not in final_states:
            part[i] = 1

    # 4) Worklist of partition‐IDs to refine
    worklist = [0, 1]

    # 5) Parallel partition refinement loop
    while worklist:
        pid = worklist.pop()
        mask = (part == pid)  # boolean mask of size n

        for sym_index in range(k):
            # For each state i, look at its transition target index T[i, sym_index]
            # and check if that target was in partition pid
            into = mask[T[:, sym_index]]      # boolean array of length n
            X = cp.nonzero(into)[0]           # states that transition into pid
            if X.size == 0:
                continue

            # Check each existing partition q for a split
            unique_parts = cp.unique(part)
            for q in unique_parts.tolist():
                Y_mask = (part == q)
                # Which states in q also have transitions into pid?
                inter = Y_mask & cp.isin(cp.arange(n, dtype=cp.int32), X)
                if not cp.any(inter) or cp.all(inter):
                    continue

                # We need to split q into Y1 (inter) and Y2 (Y_mask & ~inter)
                Y1 = inter
                Y2 = Y_mask & ~inter
                new_id = int(cp.max(part)) + 1
                part[Y1] = new_id

                # Update worklist: replace q with the smaller piece
                if q in worklist:
                    worklist.remove(q)
                    worklist.append(new_id if Y1.sum() < Y2.sum() else q)
                else:
                    worklist.append(new_id if Y1.sum() < Y2.sum() else q)
