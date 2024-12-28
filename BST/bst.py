class BinarySearchTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def valid(self):
        left_ok = True
        if self.left:
            left_ok = self.left.all_values_less_than(
                self.value) and self.left.valid()
        right_ok = True
        if self.right:
            right_ok = self.right.all_values_geq_than(
                self.value) and self.right.valid()
        return right_ok and left_ok

    def insert(self, to_add):
        if self.value > to_add:
            if self.left:
                self.left.insert(to_add)
            else:
                self.left = BinarySearchTree(to_add)
        else:
            if self.right:
                self.right.insert(to_add)
            else:
                self.right = BinarySearchTree(to_add)

    def all_values_less_than(self, value):
        if self.value >= value:
            return False
        left_less_than = True
        if self.left:
            left_less_than = self.left.all_values_less_than(value)

        right_less_than = True
        if self.right:
            right_less_than = self.right.all_values_less_than(value)
        return left_less_than and right_less_than

    def all_values_geq_than(self, value):
        if self.value <= value:
            return False
        left_geq_than = True
        if self.left:
            left_geq_than = self.left.all_values_geq_than(value)

        right_geq_than = True
        if self.right:
            right_geq_than = self.right.all_values_geq_than(value)
        return left_geq_than and right_geq_than

    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1

    def __repr__(self):
        return "({} L{} R{})".format(self.value, self.left, self.right)


def generate_tree(oracle, MAX_DEPTH, depth=0, pruning=False):
    num_nodes = 0
    value = oracle.select(1)
    tree = BinarySearchTree(value)
    num_nodes += 1

    if pruning and not tree.valid():
        return tree, num_nodes, False

    if depth < MAX_DEPTH and oracle.select(2):
        tree.left, l_num_nodes, validity = generate_tree(oracle, MAX_DEPTH, depth + 1)
        num_nodes += l_num_nodes

        if pruning and not validity:
            return tree, num_nodes, False

    if depth < MAX_DEPTH and oracle.select(3):
        tree.right, r_num_nodes, validity = generate_tree(oracle, MAX_DEPTH, depth + 1)
        num_nodes += r_num_nodes

        if pruning and not validity:
            return tree, num_nodes, False

    return tree, num_nodes, tree.valid()
