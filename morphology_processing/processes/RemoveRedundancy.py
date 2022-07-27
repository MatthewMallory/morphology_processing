from neuron_morphology.morphology import Morphology
from sklearn.neighbors import KDTree
import numpy as np
from morph_utils.graph_traversal import bfs_tree


class RemoveRedundancy:
    def __init__(self, morphology, redundancy_dist_threshold, redundancy_percentage_threshold, **kwargs):
        self.morphology = morphology
        self.redundancy_dist_threshold = redundancy_dist_threshold
        self.redundancy_percentage_threshold = redundancy_percentage_threshold
        self.process_name = "remove_redundancy"

    def process(self):
        return remove_redundancy(self.morphology, self.redundancy_dist_threshold, self.redundancy_percentage_threshold)


def remove_redundancy(morph, dist_thresh, percent_thresh, **kwargs):
    """
    At each branch point compare the shorter branch to the longer branch and see how many nodes in the shorter branch
    are within dist_thresh of the longer branch. If the shorter branch has 10 nodes and 8 of them are within 25 (pixels)
    of a node in longer branch, they will be removed.

    :param morph:
    :param dist_thresh:
    :param percent_thresh:
    :param kwargs:
    :return:
    """
    node_ids = [n['id'] for n in morph.nodes()] + [-1]
    orphans = [n for n in morph.nodes() if n['parent'] not in node_ids]
    for no in orphans:
        morph.node_by_id(no['id'])['parent'] = -1

    nodes_to_remove = []
    for root_node in morph.get_roots():
        segment, _ = bfs_tree(root_node, morph)
        furcation_nodes = [n for n in segment if (len(morph.get_children(n)) > 1) and (n['type'] != 1)]
        for f_node in furcation_nodes:
            child_segment_dict = {}
            for child in morph.get_children(f_node):
                child_seg, _ = bfs_tree(child, morph)
                child_segment_dict[child['id']] = child_seg
            sorted_keys = sorted(child_segment_dict, key=lambda k: len(child_segment_dict[k]))
            short_key = sorted_keys[0]
            long_key = sorted_keys[-1]
            short_seg = child_segment_dict[short_key]
            long_seg = child_segment_dict[long_key]
            seg_1 = [(n['x'], n['y'], n['z']) for n in long_seg]
            long_seg_lookup_tree = KDTree(np.array(seg_1))
            overlap_node_ct = 0
            for no in short_seg:
                coord = np.array([no['x'], no['y'], no['z']]).reshape(1, 3)
                dist, ind = long_seg_lookup_tree.query(coord, k=1)
                dist = dist[0][0]
                if dist < dist_thresh:
                    overlap_node_ct += 1
            percent_of_overlap = overlap_node_ct / len(short_seg)
            if percent_of_overlap > percent_thresh:
                [nodes_to_remove.append(i['id']) for i in short_seg]

    keeping_nodes = [n for n in morph.nodes() if n['id'] not in nodes_to_remove]

    redundancy_removed_morph = Morphology(
        keeping_nodes,
        node_id_cb=lambda node: node['id'],
        parent_id_cb=lambda node: node['parent'])

    results_dict = {}
    results_dict['morph'] = redundancy_removed_morph
    results_dict['redundant_nodes_removed'] = nodes_to_remove
    return results_dict
