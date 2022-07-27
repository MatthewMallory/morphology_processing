from neuron_morphology.morphology import Morphology


class EdgeArtifact:

    def __init__(self, morphology, image_shape_xyz, **kwargs ):
        self.process_name = "Edge_Artifact_Removal"
        self.required_parametes = ["morphology", "image_shape_xyz"]
        self.morphology = morphology
        self.image_shape_xyz = image_shape_xyz

    def process(self):

        morphology = self.morphology
        image_shape_xyz = self.image_shape_xyz

        print("This is what I think self.image_shape_xyz = ",image_shape_xyz)

        nodes_to_exclude = set()
        edge_nodes = 0
        for no in morphology.nodes():
            if no['x'] < image_shape_xyz[0] / 2:
                x_dist_from_edge = no['x']
            else:
                x_dist_from_edge = (image_shape_xyz[0] - no['x'])

            if no['y'] < image_shape_xyz[1] / 2:
                y_dist_from_edge = no['y']
            else:
                y_dist_from_edge = (image_shape_xyz[1] - no['y'])

            min_dist_to_edge = min(x_dist_from_edge, y_dist_from_edge)
            if min_dist_to_edge < 10:
                edge_nodes += 1
                nodes_to_exclude.add(no['id'])
        print('Found {} edge artifact nodes to remove'.format(edge_nodes))

        keeping_nodes = [n for n in morphology.nodes() if n['id'] not in nodes_to_exclude]
        fixed_edge_artifact_morph = Morphology(
            keeping_nodes,
            node_id_cb=lambda node: node['id'],
            parent_id_cb=lambda node: node['parent'])
        results_dict = {}
        results_dict['morph'] = fixed_edge_artifact_morph

        return results_dict


