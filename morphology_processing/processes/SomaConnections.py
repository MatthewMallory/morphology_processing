from scipy.spatial.distance import euclidean
from morph_utils.modifications import check_morph_for_segment_restructuring

class SomaConnections:
    def __init__(self,morphology,soma_connection_threshold):
        self.morphology = morphology
        self.soma_connection_threshold = soma_connection_threshold

    def process(self):
        """
        This function will connect any root nodes that are within distance_threshold of the soma coordinate

        distance_threshold: Make sure this unit is the same as swc (i.e. micron or pixel)

        """
        mod_morph = self.morphology.clone()
        # First do this to make sure roots are closest to the soma
        mod_morph = check_morph_for_segment_restructuring(mod_morph)[0]
        soma = mod_morph.get_soma()
        soma_id = soma['id']
        soma_coord = (soma['x'], soma['y'], soma['z'])

        root_nodes_list = [n for n in mod_morph.nodes() if n['parent'] == -1]
        root_nodes_list = [n for n in root_nodes_list if n['type'] != 1]
        ids_to_connect = []
        for root_node in root_nodes_list:
            coord = (root_node['x'], root_node['y'], root_node['z'])
            dist = euclidean(coord, soma_coord)

            if dist < self.soma_connection_threshold:
                ids_to_connect.append(root_node['id'])

        for no_id in ids_to_connect:
            mod_morph.node_by_id(no_id)['parent'] = soma_id

        return_dict = {'morph': mod_morph}

        return return_dict