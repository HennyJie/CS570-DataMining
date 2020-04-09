'''
@Author: Hejie Cui
@Date: 2020-04-05 14:02:21
@LastEditTime: 2020-04-09 19:15:34
@Description: PageRank Algorithm, hw4 for CS570 DataMining
@FilePath: /CS570-DataMining/hw4/pagerank.py
'''
import sys
import numpy as np
import re
from collections import defaultdict


def page_rank(M: np.ndarray, d: float = 0.85, max_error: float = 1e-5) -> np.ndarray:
    """the main logic of pagerank algorithm, reference: wikipedia

    Arguments:
        M {np.ndarray} -- the iteration metrix

    Keyword Arguments:
        d {float} -- probability with which teleports will occur (default: {0.85})
        max_error {float} -- the maximum difference in ranks score after each iteration (default: {1e-5})

    Returns:
        np.ndarray -- [description]
    """
    n = M.shape[1]
    v = np.ones((n, 1))/n
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / n)
    flag = False
    while not flag:
        current_v = v
        v = np.dot(M_hat, v)
        if np.sum(np.abs(current_v - v)) < max_error:
            flag = True
    return v


def generate_metrix(all_nodes: np.ndarray, edges: list) -> np.ndarray:
    """generate the metrix for pagerank iteration

    Arguments:
        edges {list} -- the edges contained in the graph

    Returns:
        np.ndarray -- the iteration metrix
    """

    node_num = len(all_nodes)

    M = np.zeros((node_num, node_num))
    for edge in edges:
        end = all_nodes.tolist().index(edge[1])
        start = all_nodes.tolist().index(edge[0])
        M[end, start] = 1
    divide = np.sum(M, axis=0)
    divide[divide == 0] = 1
    M = M / divide
    return M


def run_pagerank(argv: list):
    """main function

    Arguments:
        argv {list} -- the input argv list
    """
    input_graph = argv[0]
    output_file = argv[1]
    edges = []
    with open(input_graph, 'r') as file:
        lines = file.readlines()[1:-1]
    for line in lines:
        edge = re.findall(r'\s*(\S+)*\s*->\s*(\S+)*', line)
        if edge:
            edge = [i for i in edge[0]]
            edges.append(edge)

    all_nodes = []
    for edge in edges:
        all_nodes.append(edge[0])
        all_nodes.append(edge[1])
    all_nodes = np.unique(all_nodes)

    M = generate_metrix(all_nodes, edges)
    rank_score = page_rank(M, 0.85, 1e-5)
    rank_score_list = []

    # sort the page by descending (highest to lowest) value of pagerank, In case of ties in PageRank value,
    # lines are then sorted by ascending (lowest to highest) vertex name
    for score in rank_score:
        rank_score_list.append(score[0])
    rank_score_dict = defaultdict(float)
    for i, score in enumerate(rank_score_list):
        rank_score_dict[i] = rank_score_list[i]
    rank = sorted(rank_score_dict, key=lambda k: (-rank_score_dict[k], k))

    with open(output_file, 'w') as output_file:
        output_file.write('vertex,pagerank\n')
        for i in rank:
            output_file.write('{},{}\n'.format(
                all_nodes[i], rank_score_list[i]))


if __name__ == '__main__':
    assert len(
        sys.argv) == 3, "Parameters wrong, please provide two parameters: 1.input graph file, 2.output file."
    run_pagerank(sys.argv[1:])
