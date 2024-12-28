# Abstraction functions.

def ngram_abstraction_fn(k):
    def ngram_abstractor(state):
        last_k = state[-k:]
        return "->".join([str(e) for e in last_k])
    return ngram_abstractor


def sequence_ngram_fn(k):
    return lambda state: ngram_abstraction_fn(k)(state)


def parent_state_ngram_fn(k, max_depth):
    return lambda state: ngram_abstraction_fn(k)(parent_state(state, max_depth))


def index_parent_state_ngram_fn(k, max_depth):
    return lambda state: ngram_abstraction_fn(k)(index_parent_state(state, max_depth))


def parent_state(state, max_depth):
    """
    Masks indices of the result in index_parent_state().
    >>> s = ["a", 3, "b", 0, "c", 2, "d"]
    >>> index_parent_state(s, 3)
    ["a", (3, -1), "c", (2, -1)]
    """
    if len(state) == 0:
        return state
    return list(map(lambda x: (x[0], -1) if isinstance(x, tuple) else x, index_parent_state(state, max_depth)))


def index_parent_state(state, max_depth):
    """
    >>> s = ["a", 3, "b", 0, "c", 2, "d"]
    >>> index_parent_state(s, 3)
    ["a", (3, 2), "c", (2, 2)]
    """
    stack = []
    depth = 0

    for idx in range(len(state)):
        choice = state[idx]
        if isinstance(choice, str):
            if depth + 1 == max_depth:
                while len(stack) > 1 and stack[-1][0] < stack[-1][1]:
                    stack.pop()
                    stack.pop()
                    depth -= 1
            else:
                depth += 1
                stack.append(choice)

        elif isinstance(choice, int):
            stack.append((choice, 1))
            while len(stack) > 1 and stack[-1][0] < stack[-1][1]:
                stack.pop()
                stack.pop()
                depth -= 1

    return stack[1:]


def get_trees_max_depth(state, max_depth):
    """
    >>> s = [6, True, -5, True, 0, True, 9,  True]
    >>> get_trees_max_depth(s, 2)
    ([(0, 1, 7)], [(4, False, False), (6, False, False), (2, 3, 5)])
    >>> s = [2, True, 1]
    >>> get_trees_max_depth(s, 2)
    ([(0, 1, -1), (2, -1, -1)], [])
    >>> s = [2, True, 1, True, 0]
    >>> get_trees_max_depth(s, 2)
    ([(0, 1, -1), (2, 3, -1)], [(4, False, False)])
    >>> s = [2, True, 1, True, 0, False, True, 3, False, False]
    >>> get_trees_max_depth(s, 2)
    ([], [(4, False, False), (2, 3, 5), (7, 8, 9), (0, 1, 6)])
    >>> s = [2, True, 1, True, 0, False, True, 3, False, True]
    >>> get_trees_max_depth(s, 2)
    ([(0, 1, 6), (7, 8, 9)], [(4, False, False), (2, 3, 5)])
    >>> get_trees_max_depth([-10, True, 9, False, True, -8, True], 2)
    ([(0, 1, 6)], [(5, False, False), (2, 3, 4)])
    """
    parent_stack = []
    full_trees = []

    # Utility function which puts parents in full_trees if all their children are in full_trees
    def pop_finished_parents():
        for i in reversed(range(len(parent_stack))):
            positions = parent_stack[i]
            if -1 in positions:
                break
            children_positions = [pos[0] for pos in full_trees]
            left_child_pos = positions[1] + 1
            right_child_pos = positions[2] + 1
            if state[positions[1]] and left_child_pos not in children_positions:
                break
            if state[positions[2]] and right_child_pos not in children_positions:
                break
            full_trees.append(positions)
            parent_stack.pop(i)

    for i in range(len(state)):
        e = state[i]
        if 'int' in str(type(e)):
            if len(parent_stack) < max_depth:
                parent_stack.append((i, -1, -1))
            else:
                full_trees.append((i, False, False))
                pop_finished_parents()
        else:
            parent = parent_stack[-1]
            if parent[1] == -1:
                parent_stack[-1] = (parent[0], i, -1)
            elif parent[2] == -1:
                parent_stack[-1] = (parent[0], parent[1], i)
                if not e:
                    full_trees.append(parent_stack.pop())
                    pop_finished_parents()
    parent_stack = [posns for posns in parent_stack if posns not in full_trees]
    return parent_stack, full_trees
