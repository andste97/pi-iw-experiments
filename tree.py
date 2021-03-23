import numpy as np
import operator
from utils import cstr


class Node:
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.children = []

    def size(self):
        return np.sum([c.size() for c in self.children]) + 1

    def breadth_first(self):
        current_nodes = [self]
        while len(current_nodes) > 0:
            children = []
            for node in current_nodes:
                yield node
                children.extend(node.children)
            current_nodes = children

    def depth_first(self):
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def add(self, data):
        return Node(data, parent=self)

    def make_root(self):
        if not self.is_root():
            self.parent.children.remove(self)  # just to be consistent
            self.parent = None
        if self.depth > 0:
            self.update_depths(depth=0)

    def update_depths(self, depth):
        delta = depth - self.depth
        for node in self.breadth_first():
            node.depth += delta

    def str_node(self, i=0, str_fn=lambda n: str(n.data)):  # TODO: remove, use str_tree
        tab = '   '
        s = f"{i} | {str_fn(self)}" + '\n'
        cnt = 1
        for node in self.depth_first():
            d = node.depth - self.depth
            if d > 0:
                s += "".join([tab] * d + [f"{i+cnt} | {str_fn(node)}", '\n'])
                cnt += 1
        return s

    def str_tree(self, str_fn=lambda n: 'o', selected_child=None, all_children_new_line=False, max_depth=None):
        str_root = str_fn(self)
        node_characters = len(str_root)
        lines = [" " + str_root]
        if all_children_new_line:
            lines.append("")  # start a new line
        q = list(reversed(self.children))
        depth_nodes = [len(self.children)]
        print_blue = False
        while q:
            n = q.pop()

            if selected_child is not None and n.depth == 1:
                print_blue = n is selected_child

            if len(lines[-1]) == 0:  # new branch
                for i, d in enumerate(depth_nodes):
                    if all_children_new_line:
                        lines[-1] += " "
                    else:
                        lines[-1] += " "*(node_characters+1)

                    if i == len(depth_nodes)-1:  # last one
                        assert d != 0
                        if d == 1:
                            lines[-1] += '\u2570'  # L
                        else:
                            lines[-1] += '\u251c'  # |-
                    else:
                        if d == 0:
                            lines[-1] += ' '
                        else:
                            lines[-1] += '\u2502' # |
            else: # same branch, this is never the case with all_children_new_line=True
                if depth_nodes[-1] == 1:
                    lines[-1] += '\u2500' # --
                else:
                    lines[-1] += '\u252c' # T

            str_node = str_fn(n)
            if not all_children_new_line:
                assert len(str_node) == len(str_root), "Use all_children_new_line for str_node of variable size."
            lines[-1] += '\u2500' + str_node # remove '--' for shorter tree
            depth_nodes[-1] -= 1
            assert depth_nodes[-1] >= 0

            if n.children and (max_depth is None or n.depth < max_depth + 1):
                # continue branch with first children
                q.extend(reversed(n.children))  # add them reversed, since it's a stack (LIFO)
                depth_nodes.append(len(n.children))
                if all_children_new_line:
                    lines.append("")  # start a new line
            else:
                # end branch
                if print_blue:
                    lines[-1] = lines[-1][:3] + cstr(lines[-1][3:], 'blue')  # TODO: this 3 should be dependant on str_node
                lines.append("")

            # shrink depth list to the current depth
            while depth_nodes and depth_nodes[-1] == 0:
                depth_nodes.pop()

        return "\n".join(lines)

    def ascendants(self):
        aux=self
        while not aux.is_root():
            aux = aux.parent
            yield aux


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

    def new_root(self, node, keep_subtree):
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