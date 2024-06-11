import operator

from tree_utils.node import Node

class Tree:
    """
    Although a Node is a tree by itself, this class provides more iterators and quick access to the different depths of
    the tree, and keeps track of the root node
    """

    def __init__(self, root_data):
        root = Node(root_data)
        root.id = 0
        self.next_id = 1
        self.new_root(root, keep_subtree=True)

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def str_tree(self, str_fn=lambda node: str(node.data)):
        return (self.root.str_node(i=0, str_fn=str_fn))

    def iter_insertion_order(self):  # To make it explicit
        return iter(self._nodes)

    def iter_breadth_first(self):
        return self.root.breadth_first()

    def iter_breadth_first_reverse(self):
        for n in reversed(list(self.root.breadth_first())):
            yield n

    def new_root(self, node: Node, keep_subtree):
        node.make_root()
        self.root = node
        if not keep_subtree:
            node.children = list()  # remove children

        self._nodes = list(self.root.breadth_first())
        self._nodes.sort(key=operator.attrgetter("id"))  # Sort nodes by id so that we can iterate over them
        assert self._nodes[0] is self.root

    def add(self, parent_node, data):
        child = parent_node.add(data)
        child.id = self.next_id
        self.next_id += 1
        self._nodes.append(child)
        return child

    def remove_subtree(self, node):
        node.parent.children.remove(node)
        node.parent = None
        for n in node.breadth_first():
            self._nodes.remove(n)

    def max_depth(self) -> int:
        return max([node.depth for node in self._nodes])

    def detatch_subtree(self, node, copy_and_keep_node):
        if copy_and_keep_node:
            new_tree = Tree(root_data=node.data)
            new_tree.root.id = node.id
            new_tree.root.depth = node.depth
            for c in node.children:
                new_tree.root.children.append(c)
                c.parent = new_tree.root
            node.children = list()
        else:
            new_tree = Tree()
            new_tree.root = node

        new_tree.next_id = self.next_id
        new_tree.new_root(new_tree.root, keep_subtree=True)
        return new_tree