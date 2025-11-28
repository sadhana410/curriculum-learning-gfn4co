import numpy as np

def load_col_file(path):
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()

    n_nodes = None
    edges = []

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        if line.startswith("c"):
            continue

        #p edge <nodes> <edges>
        if line.startswith("p"):
            parts = line.split()
            #p edge N E
            n_nodes = int(parts[2])
            continue

        #e u v
        if line.startswith("e"):
            _, u, v = line.split()
            edges.append((int(u), int(v)))
            continue

        parts = line.split()
        if len(parts) == 2:
            u, v = map(int, parts)
            edges.append((u, v))
            continue

    if n_nodes is None:
        n_nodes = max(max(u, v) for u, v in edges)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for u, v in edges:
        u -= 1
        v -= 1
        adj[u, v] = 1
        adj[v, u] = 1

    return adj
