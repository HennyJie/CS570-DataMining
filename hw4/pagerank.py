import sys
import numpy as np
from collections import defaultdict


# the main logic of pagerank algorithm, reference: wikipedia
def page_rank(M, d=0.85, max_error=1e-5, max_iter=100):
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


# generate the metrix for pagerank iteration
def generate_metrix(edges):
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


# main function
def run_pagerank(argv):
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
