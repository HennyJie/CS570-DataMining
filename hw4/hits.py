'''
@Author: Hejie Cui
@Date: 2020-04-08 14:44:56
@LastEditTime: 2020-04-09 19:20:01
@Description: HITS Algorithm, hw4 for CS570 DataMining
@FilePath: /CS570-DataMining/hw4/hits.py
'''
import sys
import math
import numpy as np
import re


class Page:
    def __init__(self, id, edges: list):
        """page init

        Arguments:
            id {str or int} -- id/name of each page
            edges {list} -- all the edges contained in the graph
        """
        self.id = id
        self.authority = 1
        self.hub = 1
        self.incoming_neighbors = [edge[0] for edge in edges if edge[1] == id]
        self.outcoming_neighbors = [edge[1] for edge in edges if edge[0] == id]

    def __gt__(self, other) -> bool:
        """custom compare function

        Arguments:
            other {Page} -- another page object

        Returns:
            bool -- comparasion result
        """
        if self.authority == other.authority:
            if self.hub == other.hub:
                return self.id < self.id
            return self.hub > other.hub
        return self.authority > other.authority


def hits(pages: list, max_error: float = 1e-5, max_iter: int = 100):
    """the main logic of hits algorithm

    Arguments:
        pages {list} -- a list of page objects

    Keyword Arguments:
        max_error {float} -- the minimum difference of both authority and hub score sum between two iteration(default: {1e-5})
        max_iter {int} -- maximum number of times to apply iteration (default: {100})
    """
    page_dict = {page.id: page for page in pages}
    for i in range(max_iter):
        change = 0
        norm = 0
        for page in pages:
            tmp = page.authority
            page.authority = 0
            for q in page.incoming_neighbors:
                page.authority += page_dict[q].hub
            norm += page.authority ** 2
        norm = math.sqrt(norm)
        for page in pages:
            page.authority /= norm
            change += abs(tmp - page.authority)

        norm = 0
        for page in pages:
            tmp = page.hub
            page.hub = 0
            for r in page.outcoming_neighbors:
                page.hub += page_dict[r].authority
            norm += page.hub ** 2
        norm = math.sqrt(norm)
        for page in pages:
            page.hub /= norm
            change += abs(tmp - page.hub)

        if change < max_error:
            break


def run_hits(argv: list):
    """tha main function

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
        try:
            edge[0] = int(edge[0])
        except:
            pass
        try:
            edge[1] = int(edge[1])
        except:
            pass
        all_nodes.append(edge[0])
        all_nodes.append(edge[1])
    all_nodes = np.unique(all_nodes)

    pages = []
    for node in all_nodes:
        pages.append(Page(node, edges))
    hits(pages)

    # sort the documents by descending (highest to lowest) value of authority score; in case of authority ties,
    # sort by descending value of hub score; in case of both authority and hub ties, sort by ascending (lowest to highest) vertex name.
    # the sort function is realized by reloading the __lt__ function in class Page.
    pages = sorted(pages, reverse=True)

    with open(output_file, 'w') as output_file:
        output_file.write('vertex,authority,hub\n')
        for page in pages:
            output_file.write('{},{},{}\n'.format(
                page.id, page.authority, page.hub))


if __name__ == '__main__':
    assert len(
        sys.argv) == 3, "Parameters wrong, please provide two parameters: 1.input graph file, 2.output file."
    run_hits(sys.argv[1:])
