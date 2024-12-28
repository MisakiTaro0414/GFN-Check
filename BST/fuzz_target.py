import sys
import random
from collections import Counter
from bst import BinarySearchTree

MAX_DEPTH = 4
VALUES = range(10)


def generate_tree(depth=0, min_value=-float('inf'), max_value=float('inf')):
    """
    A white-box fuzzer that choose random sequence by keeping a current choice sequence always valid.
    """
    if len(range(int(min_value) + 1, int(max_value))) == 0:
        return None
    value = random.choice(range(int(min_value) + 1, int(max_value)))
    tree = BinarySearchTree(value)
    if depth < MAX_DEPTH and random.choice([True, False]):
        # For left child, restrict max value to current node's value
        tree.left = generate_tree(depth + 1, min_value, value)
    if depth < MAX_DEPTH and random.choice([True, False]):
        # For right child, restrict min value to current node's value
        tree.right = generate_tree(depth + 1, value, max_value)
    return tree


def fuzz(valid=0, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
    sizes = list()
    valid_set = set()
    trials = 10000
    for i in range(trials):
        print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), (len(valid_set) * 100 / valids) if valids != 0 else 0), end='\r')
        # Node values range from 0 to 9
        tree = generate_tree(min_value=min(VALUES) - 1, max_value=max(VALUES))
        if tree.valid():
            valids += 1
            if tree.__repr__() not in valid_set:
                valid_set.add(tree.__repr__())
                sizes.append(tree.depth())
            else:
                valid += 1  # Simulate a reward mechanism if needed
        else:
            invalid += 1  # Simulate a reward mechanism if needed
    print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
        trials, valids, len(valid_set), len(valid_set) * 100 / valids), end='\r')
    print("\ndone!", file=sys.stderr)
    print(Counter(sizes))


if __name__ == '__main__':
    print("====Target====")
    fuzz()
