import numpy as np

from piiw.utils.utils import cstr


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
        """Count the number of children of this node, including this node."""
        return np.sum([c.size() for c in self.children]) + 1

    def breadth_first(self):
        """Yield all nodes in this tree using breadth first search."""
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    def depth_first(self):
        """Yield all nodes in this tree using depth first search."""
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def add(self, data):
        """Add a new child node containing data to this node."""
        return Node(data, parent=self)

    def make_root(self):
        """Make this node the root node."""
        if not self.is_root():
            self.parent.children.remove(self)  # just to be consistent
            self.parent = None
        if self.depth > 0:
            self.update_depths(depth=0)

    def update_depths(self, depth):
        """Update the depth for each node of the tree. Should be called after changes to tree
        structure such as after setting a new root."""
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
