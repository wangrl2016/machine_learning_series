import itertools
from matplotlib import pyplot
import networkx

subset_sizes = [8, 6, 4, 2]
subset_color = [
    "gold",
    "violet",
    "limegreen",
    "darkorange",
]

def multilayered_graph(*subset_sizes):
    extents = networkx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = networkx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in networkx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G

if __name__ == '__main__':
    G = multilayered_graph(*subset_sizes)
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    pos = networkx.multipartite_layout(G, subset_key="layer")
    pyplot.figure()
    # node_size 节点面积的近似值
    networkx.draw(G, pos, node_color=color, node_size=200, with_labels=False)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
