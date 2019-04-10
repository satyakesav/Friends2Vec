"""
Creates an adjacency list representation of an undirected graph.
It may be used for verifying the generation of embeddings for graph inputs.

Creates an undirected graph with 'num_nodes' number of nodes and having an average of 'adj_nodes' number of
adjacent nodes and saves the adjacency list representation to a specified output file.
"""

import random

num_nodes = 10
adj_nodes = 3

outfile = "sample_data/sample.adjlist"

adjlist = {}
for u in range(num_nodes):
    if u not in adjlist:
        adjlist[u] = []

    for _ in range(adj_nodes // 2):
        v = random.randint(0, num_nodes-1)

        if u == v:
            continue

        if v not in adjlist:
            adjlist[v] = []

        adjlist[u].append(v)
        adjlist[v].append(u)

with open(outfile, 'w') as f:
    for u in range(num_nodes):
        s = str(u+1)
        adjlist[u] = list(set(adjlist[u]))
        adjlist[u].sort()

        for v in adjlist[u]:
            s += " " + str(v+1)

        f.write(s + "\n")

