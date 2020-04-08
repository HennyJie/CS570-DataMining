'''
@Author: Hejie Cui
@Date: 2020-04-05 14:02:21
@LastEditTime: 2020-04-08 17:17:24
@Description: hw4 for CS570 DataMining
@FilePath: /CS570-DataMining/hw4/pagerank.py
'''
import sys
import numpy as np


def page_rank(M, d=0.85, max_error=1e-5, max_iter=100) -> np.ndarray:
    """[the main logic of pagerank algorithm, reference: wikipedia]

    Arguments:
        M {[type]} -- [the iteration metrix]

    Keyword Arguments:
        d {float} -- [probability with which teleports will occur] (default: {0.85})
        max_error {[type]} -- [total error in ranks should be less than epsilon] (default: {1e-5})
        max_iter {int} -- [maximum number of times to apply power iteration] (default: {100})

    Returns:
        np.ndarray -- [description]
    """
    n = M.shape[1]
    v = np.random.rand(n, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / n)
    for i in range(max_iter):
        current_v = v
        v = np.dot(M_hat, v)
        if np.sum(np.abs(current_v - v)) < max_error:
            break
    return v


def generate_metrix(edges: list) -> np.ndarray:
    """[generate the metrix for pagerank iteration]

    Arguments:
        edges {list} -- [the edges contained in the graph]

    Returns:
        np.ndarray -- [the iteration metrix]
    """
    all_nodes = []
    for edge in edges:
        all_nodes.append(edge[0])
        all_nodes.append(edge[1])
    node_num = len(np.unique(all_nodes))

    M = np.zeros((node_num, node_num))
    for edge in edges:
        M[edge[1], edge[0]] = 1
    divide = np.sum(M, axis=0)
    divide[divide == 0] = 1
    M = M / divide
    return M


def run_pagerank(argv: list):
    """[main function]

    Arguments:
        argv {list} -- [the input argv list]
    """
    input_graph = argv[0]
    output_file = argv[1]
    edges = []
    with open(input_graph, 'r') as file:
        lines = file.readlines()[1:-1]
    for line in lines:
        start, end = line.split(' -> ')
        start = start.strip()
        end = end.strip()
        # here I use num as the name of vertex
        edges.append([int(start), int(end)])

    M = generate_metrix(edges)
    rank_score = page_rank(M, 0.85, 1e-5, 100)
    rank_score_list = []

    # sort the page by descending (highest to lowest) value of pagerank
    for score in rank_score:
        rank_score_list.append(score[0])
    rank = np.argsort(rank_score_list)[::-1]

    with open(output_file, 'w') as output_file:
        output_file.write('vertex,pagerank\n')
        for i in rank:
            output_file.write('{},{}\n'.format(i, rank_score_list[i]))


if __name__ == '__main__':
    assert len(
        sys.argv) == 3, "Parameters wrong, please provide two parameters: 1.input graph file, 2.output file."
    run_pagerank(sys.argv[1:])
