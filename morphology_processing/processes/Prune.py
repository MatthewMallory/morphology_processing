from neuron_morphology.morphology import Morphology
from morph_utils.graph_traversal import bfs_tree

class PruneTree:
    def __init__(self, morphology, num_node_thresh, **kwargs):
        self.process_name = "Prune_Segments"
        self.required_parametes = ["morphology", "distance_threshold"]
        self.morphology = morphology
        self.num_node_thresh = num_node_thresh

    def process(self):
        morphology = self.morphology
        num_node_thresh = self.num_node_thresh

        nodes_to_remove = set()
        pruin_count = 0
        bifur_nodes = [n for n in morphology.nodes() if len(morphology.get_children(n)) > 1]
        for bif_node in bifur_nodes:
            children = morphology.get_children(bif_node)
            for child in children:
                child_remove_nodes, child_seg_length = bfs_tree(child, morphology)
                if child_seg_length < num_node_thresh:
                    pruin_count += 1
                    [nodes_to_remove.add(n['id']) for n in child_remove_nodes]

        keeping_nodes = [n for n in morphology.nodes() if n['id'] not in nodes_to_remove]

        pruned_morph = Morphology(
            keeping_nodes,
            node_id_cb=lambda node: node['id'],
            parent_id_cb=lambda node: node['parent'])
        # print('{} Pruining sites, {} Nodes removes'.format(pruin_count ,len(nodes_to_remove)))
        results_dict = {}
        results_dict['morph'] = pruned_morph
        return results_dict