"""
Copyright (c) 2014, Bryan Perozzi, Rami Al-Rfou
All rights reserved.

DeepWalk is distributed under the terms of GNU LIBRARY GENERAL PUBLIC LICENSE version 3
"""

import os
import random
import sys
from argparse import ArgumentParser

from six.moves import range
import psutil
from multiprocessing import cpu_count

from gensim.models import Word2Vec

sys.path.append('embeddings/deepwalk')
import graph


p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass


def process(args):
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks, path_length=args.walk_length,
                                        alpha=0, rand=random.Random(args.seed))
    print("Training...")
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, hs=1, workers=args.workers)

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
