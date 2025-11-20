import matplotlib.pyplot as plt
import networkx as nx

def visualize_coloring(adj, state, title="Graph Coloring"):
    N = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j] == 1:
                G.add_edge(i, j)

    node_colors = ["lightgray" if c == -1 else f"C{c}" for c in state]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(5,5))
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=700, font_size=14)
    plt.title(title)
    plt.show()
