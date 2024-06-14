import hashlib

from tree_utils.node import Node
import networkx as nx
import matplotlib.pyplot as plt

from utils.utils import create_folders_in_path

h = hashlib.new('sha1')

def visualize_tree_with_observations(node: Node, fname):
    create_folders_in_path(fname)
    G = nx.DiGraph()
    add_node_to_graph(G, node)
    create_networkx_graph(node, G)
    create_tree_layout_with_observations(G)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_tree_no_observations(node: Node, path):
    G = nx.DiGraph()
    G.add_node(node, obs=node.data["obs"][-1])
    create_networkx_graph(node, G)
    plt.figure(figsize=(25, 15))
    create_tree_layout(G)
    plt.show()

def create_networkx_graph(node: Node, G: nx.DiGraph):
    for child in node.children:
        add_node_to_graph(G, child)
        if child.pruned:
            edge_color="red"
        else:
            edge_color="black"

        G.add_edge(node, child, a=child.data["a"], color=edge_color)
        create_networkx_graph(child, G)


def add_node_to_graph(G, node):
    features = [x[1] for x in node.data["features"]]
    h.update(bytearray(features))
    features = h.hexdigest()[-16:]
    G.add_node(node, obs=node.data["obs"][-1], R=int(node.data["R"]), features=features)


def create_tree_layout(G: nx.DiGraph):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    node_labels = nx.get_node_attributes(G, "R")
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=50)

    draw_edge_labels(G, pos, 0.3)


def draw_edge_labels(G, pos, label_pos=0.5):
    edge_labels = nx.get_edge_attributes(G, 'a')
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        bbox={"alpha": 0},
        label_pos=label_pos
    )


def create_tree_layout_with_observations(G: nx.DiGraph):
    fig, ax = plt.subplots(figsize=(100, 40))
    plt.box(False)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    pos = nx.rescale_layout_dict(pos)

    # get edge colors
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]

    # Note: the min_source/target_margin kwargs only work with FancyArrowPatch objects.
    # Force the use of FancyArrowPatch for edge drawing by setting `arrows=True`,
    # but suppress arrowheads with `arrowstyle="-"`
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="-",
        min_source_margin=15,
        min_target_margin=15,
        edge_color=edge_colors
    )

    draw_edge_labels(G, pos)

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
    icon_center = icon_size / 2.0

    # Add the respective image to each node
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["obs"])
        a.text(pos[n][0], pos[n][1], G.nodes[n]["features"])
        a.axis("off")