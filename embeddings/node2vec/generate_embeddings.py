from argparse import ArgumentParser
import sys

import networkx as nx

sys.path.append('embeddings/node2vec')
from node2vec import Node2Vec


def create_graph(input_file):
    g = nx.Graph()

    adjlist = {}
    with open(input_file, 'r') as f:
        for line in f:
            u = int(line.split()[0])
            adjlist[u] = list(map(int, line.split()[1:]))

    for u in adjlist.keys():
        g.add_node(u)

    for u in adjlist.keys():
        for v in adjlist[u]:
            g.add_edge(u, v)

    return g


def process(args):
    graph = create_graph(args.input)

    node2vec = Node2Vec(graph, dimensions=args.dimensions, walk_length=args.walk_length,
                        num_walks=args.number_walks, workers=args.workers)

    model = node2vec.fit(window=args.window_size)

    model.wv.save_word2vec_format(args.output)


def main():
    parser = ArgumentParser(description='Arguments for DeepWalk algorithm.')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--dimensions', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
