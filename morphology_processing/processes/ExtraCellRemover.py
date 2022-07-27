import numpy as np
from collections import defaultdict
from morph_utils.graph_traversal import bfs_tree

from sklearn.neighbors import KDTree
from neuron_morphology.morphology import Morphology
from scipy.ndimage.measurements import label
from scipy.spatial import distance


def get_other_soma_coords(auto_trace_morph, soma_segmentation, img_image_shape_xyz):
    """
    returns dictionary with other soma labeled and the soma from the swc
    """
    soma_coord = (auto_trace_morph.get_soma()['x'], auto_trace_morph.get_soma()['y'], auto_trace_morph.get_soma()['z'])
    xs, ys = soma_segmentation['# x'].values, soma_segmentation['y'].values
    somas_2d = np.zeros((img_image_shape_xyz[1], img_image_shape_xyz[0]), dtype=np.bool)

    for x, y in zip(xs, ys):
        if (x < img_image_shape_xyz[0]) and (y < img_image_shape_xyz[1]):
            somas_2d[y, x] = True
    structure = np.ones((3, 3), dtype=np.int)
    labeled, num_comps = label(somas_2d, structure)

    other_soma_ct = 0
    somas_dict = {}
    somas_dict['from_swc'] = np.array(soma_coord)
    for lab in np.unique(labeled):
        if lab != 0:
            ycoords, xcoords = np.where(labeled == lab)
            xc_mean = np.mean(xcoords)
            yc_mean = np.mean(ycoords)
            dist_to_swc_soma = (((soma_coord[0] - xc_mean) ** 2) + ((soma_coord[1] - yc_mean) ** 2)) ** 0.5
            if (len(xcoords) > 500) and (dist_to_swc_soma > 1000):
                this_soma = soma_segmentation.loc[soma_segmentation['# x'].between(min(xcoords), max(xcoords)) |
                                                  soma_segmentation['y'].between(min(ycoords), max(ycoords))]
                somas_dict[lab] = this_soma.mean()[0:3].values
                other_soma_ct += 1
    return somas_dict, other_soma_ct


def get_pct_nodes_valid_to_invalid_within_denoise_radius(auto_trace_morph,
                                                         root_coord,
                                                         denoise_radius,
                                                         known_valid_root_nodes,
                                                         known_invalid_root_nodes):
    """
    This funciton returns the percent of visited nodes that are labeled 1 within denoise_radius r of current root node
    """

    known_valid_segments = [l for l in auto_trace_morph.get_tree_list() if l[0]['id'] in known_valid_root_nodes]
    known_invalid_segments = [l for l in auto_trace_morph.get_tree_list() if l[0]['id'] in known_invalid_root_nodes]
    #     unknown_segments = [l for l in auto_trace_morph.get_tree_list() if l[0]['id'] not in known_valid_root_nodes]
    # Consider factoring unknown segment as assumed invalid

    if known_invalid_segments != []:
        invalid_coords = []
        for seg in known_invalid_segments:
            for no in seg:
                invalid_coords.append((no['x'], no['y'], no['z']))
        all_invalid_nodes = np.array(invalid_coords)
        invalid_nodes_lookup_tree = KDTree(all_invalid_nodes, leaf_size=2)

        valid_coords = []
        for seg in known_valid_segments:
            for no in seg:
                valid_coords.append((no['x'], no['y'], no['z']))
        all_valid_nodes = np.array(valid_coords)
        valid_nodes_lookup_tree = KDTree(all_valid_nodes, leaf_size=2)

        number_valid_within_denoise_radius = \
            valid_nodes_lookup_tree.query_radius(root_coord.reshape(1, 3), r=denoise_radius, count_only=True)[0]
        number_invalid_within_denoise_radius = \
            invalid_nodes_lookup_tree.query_radius(root_coord.reshape(1, 3), r=denoise_radius, count_only=True)[0]
        if number_valid_within_denoise_radius == 0:
            ratio = 0

        elif number_invalid_within_denoise_radius == 0:
            ratio = 1

        else:
            ratio = number_valid_within_denoise_radius / (
                    number_invalid_within_denoise_radius + number_valid_within_denoise_radius)
        return ratio
    else:
        #         There are no known invalid root coords yet
        return 1


# Features for root node classification
def get_dist_to_nearest_valid_node(auto_trace_morph, root_coord, analyzed_root_nodes):
    """
    Where a valid node is one that has previously been visited and labeled one, or explicity connects back to soma

    This is a features for root node classification

    """
    known_valid_root_nodes = analyzed_root_nodes
    known_valid_segments = [l for l in auto_trace_morph.get_tree_list() if l[0]['id'] in known_valid_root_nodes]
    unknown_segments = [l for l in auto_trace_morph.get_tree_list() if l[0]['id'] not in known_valid_root_nodes]
    all_coords = []
    for seg in known_valid_segments:
        for no in seg:
            all_coords.append((no['x'], no['y'], no['z']))
    all_valid_nodes = np.array(all_coords)

    valid_nodes_lookup_tree = KDTree(all_valid_nodes, leaf_size=2)
    dis, ind = valid_nodes_lookup_tree.query(root_coord.reshape(1, 3), k=1)
    return dis[0][0]


def get_root_node_by_id(start_node_id, morphology):
    curr_node = morphology.node_by_id(start_node_id)
    counter = 0
    while counter != 1:
        curr_parent = curr_node['parent']
        if curr_parent == -1:
            counter = 1
        else:
            curr_node = morphology.node_by_id(curr_node['parent'])
    return curr_node


class ExtraCellRemover:

    def __init__(self, morphology, image_shape_xyz, soma_segmentation, extraneous_cell_classifier, denoise_radius, **kwargs):
        self.morphology = morphology
        self.image_shape_xyz = image_shape_xyz
        self.soma_segmentation = soma_segmentation
        self.extraneous_cell_classifier = extraneous_cell_classifier
        self.denoise_radius = denoise_radius
        self.process_name = "Extra_Cell_Remover"

    def process(self):
        """
        Classifies each root node as signal or noise
        return: edited morphology, somas_dict, labels_df
        """

        somas_dict, other_soma_ct = get_other_soma_coords(self.morphology, self.soma_segmentation, self.image_shape_xyz)
        print("Found {} other somas in this ch1 mip".format(other_soma_ct))
        auto_trace_nodes = self.morphology.nodes()
        autotrace_root_nodes = [n for n in auto_trace_nodes if (n['parent'] == -1) and (n['type'] != 1)]
        soma_node = self.morphology.get_soma()
        if soma_node != None:
            soma_coord = [soma_node['x'], soma_node['y'], soma_node['z']]
        else:
            print("    Warning: Not traversing nodes from soma out.")
            xs = np.array([n['x'] for n in self.morphology.nodes()])
            ys = np.array([n['y'] for n in self.morphology.nodes()])
            zs = np.array([n['z'] for n in self.morphology.nodes()])
            soma_coord = xs.mean(), ys.mean(), zs.mean()

        sorted_root_nodes = sorted(autotrace_root_nodes,
                                   key=lambda n: distance.euclidean([n['x'], n['y'], n['z']], soma_coord))

        labels_dict = {}
        feat_dict = defaultdict(dict)
        known_valid_root_nodes = set()
        known_invalid_root_nodes = set()
        known_valid_root_nodes.add(self.morphology.get_soma()['id'])

        for root_node in sorted_root_nodes:
            root_coord = np.array((root_node['x'], root_node['y'], root_node['z']))
            root_coord_id = \
            [n for n in self.morphology.nodes() if np.array_equiv(np.array((n['x'], n['y'], n['z'])), root_coord) == True][0]['id']
            neurite_segment = [t for t in self.morphology.get_tree_list() if root_node in t][0]
            neurite_coords = np.array([(n['x'], n['y'], n['z']) for n in neurite_segment])
            neurite_lookup_tree = KDTree(neurite_coords, leaf_size=2)
            d1, i1 = neurite_lookup_tree.query(np.array(soma_coord).reshape(1, 3), k=len(neurite_coords))

            # Ft 1
            shortest_distance_to_soma = d1[0][0]

            # Get dist to soma in x-y in delta terms relative to image size
            closest_coord_to_soma = neurite_coords[i1[0][0]]
            delta_x = abs(
                (closest_coord_to_soma[0] - soma_coord[0]) / self.image_shape_xyz[0]) * 100  # to scale as % between 0-100
            delta_y = abs((closest_coord_to_soma[1] - soma_coord[1]) / self.image_shape_xyz[1]) * 100
            shortest_dist_to_soma_xy = 0.01 * ((delta_x ** 2 + delta_y ** 2) ** 0.5)  # to scale it back to 0-1

            # Get coord that is furthest from Soma
            furthest_coord_from_soma = neurite_coords[i1[0][-1]]

            # Check if x and y lie before or after the x and y midpoint
            if furthest_coord_from_soma[0] < self.image_shape_xyz[0] / 2:
                x_dist_from_edge = furthest_coord_from_soma[0] / self.image_shape_xyz[0]
            else:
                x_dist_from_edge = (self.image_shape_xyz[0] - furthest_coord_from_soma[0]) / self.image_shape_xyz[0]

            if furthest_coord_from_soma[1] < self.image_shape_xyz[1] / 2:
                y_dist_from_edge = furthest_coord_from_soma[1] / self.image_shape_xyz[1]
            else:
                y_dist_from_edge = (self.image_shape_xyz[1] - furthest_coord_from_soma[1]) / self.image_shape_xyz[1]

            if furthest_coord_from_soma[2] < self.image_shape_xyz[2] / 2:
                z_dist_from_edge = furthest_coord_from_soma[2] / self.image_shape_xyz[2]
            else:
                z_dist_from_edge = (self.image_shape_xyz[2] - furthest_coord_from_soma[2]) / self.image_shape_xyz[2]

            # Ft 2 and maybe Ft3
            furthest_coord_rel_dist_to_edge_xy = min((x_dist_from_edge, y_dist_from_edge))
            furthest_coord_rel_dist_to_edge_z = z_dist_from_edge

            # Ft 4 Ratio of distance to soma over distance to edge
            distance_to_soma_over_distance_to_edge = furthest_coord_rel_dist_to_edge_xy / shortest_dist_to_soma_xy

            # Ft 5 distance to nearest known valid node
            dist_to_nearest_valid_node = get_dist_to_nearest_valid_node(self.morphology, root_coord, known_valid_root_nodes)

            # Ft 6
            good_bad_ratio = get_pct_nodes_valid_to_invalid_within_denoise_radius(self.morphology, root_coord, self.denoise_radius,
                                                                                  known_valid_root_nodes,
                                                                                  known_invalid_root_nodes)

            ## ft 7 and 8 (7: Closer to other soma or swc soma, 8: pix dist to other soma)
            if len(somas_dict) > 1:
                soma_c = somas_dict['from_swc']
                distance_to_beat = (((root_node['x'] - soma_c[0]) ** 2) +
                                    ((root_node['y'] - soma_c[1]) ** 2) +
                                    ((root_node['z'] - soma_c[2]) ** 2)) ** 0.5
                closer_to_good_or_bad_soma = 0
                distances = []
                for soma_label, s_cord in somas_dict.items():
                    if soma_label != 'from_swc':
                        d, i = neurite_lookup_tree.query(np.array(s_cord).reshape(1, 3), k=1)
                        distances.append(d[0][0])
                        if d[0][0] < distance_to_beat:
                            closer_to_good_or_bad_soma = 1
                min_dist_to_other_soma = min(distances)


            else:
                # No bad soma to be closer to
                closer_to_good_or_bad_soma = 0
                min_dist_to_other_soma = max(self.image_shape_xyz)

            # print(closer_to_good_or_bad_soma, min_dist_to_other_soma)

            # ft 9 dist to other soma over distance to swc soma
            dist_to_good_soma_over_dist_to_other = shortest_distance_to_soma / min_dist_to_other_soma

            # Compile Results in feature dictionary
            feat_dict[root_coord_id]['min_dist_to_soma_pix'] = shortest_distance_to_soma
            feat_dict[root_coord_id]['shortest_dist_to_edge_xy_norm'] = furthest_coord_rel_dist_to_edge_xy
            feat_dict[root_coord_id]['shortest_dist_to_edge_z_norm'] = furthest_coord_rel_dist_to_edge_z
            feat_dict[root_coord_id]['dist_to_soma_over_dist_to_edge'] = distance_to_soma_over_distance_to_edge
            feat_dict[root_coord_id]['dist_to_nearest_valid_node'] = dist_to_nearest_valid_node
            feat_dict[root_coord_id]['percent_valid_nodes_within_denoise_radius'] = good_bad_ratio
            feat_dict[root_coord_id]['closer_to_other_soma'] = closer_to_good_or_bad_soma
            feat_dict[root_coord_id]['min_distance_to_other_soma'] = min_dist_to_other_soma
            feat_dict[root_coord_id]['dist_to_swc_soma_over_other_soma'] = dist_to_good_soma_over_dist_to_other

            feat_input = np.array((shortest_distance_to_soma,
                                   furthest_coord_rel_dist_to_edge_xy,
                                   furthest_coord_rel_dist_to_edge_z,
                                   distance_to_soma_over_distance_to_edge,
                                   dist_to_nearest_valid_node,
                                   good_bad_ratio,
                                   closer_to_good_or_bad_soma,
                                   min_dist_to_other_soma,
                                   dist_to_good_soma_over_dist_to_other)).reshape(9, 1).T

            prediction = self.extraneous_cell_classifier.predict(feat_input)[0]
            feat_dict[root_coord_id]['prediction'] = prediction

            # Update traversed node list depending on label
            if prediction == 1:
                known_valid_root_nodes.add(root_node['id'])
                labels_dict[root_node['id']] = 1
            else:
                known_invalid_root_nodes.add(root_node['id'])
                labels_dict[root_node['id']] = 0

        print('     {} Good Root Nodes, {} Noise Root Nodes \n'.format(len(known_valid_root_nodes),
                                                                       len(known_invalid_root_nodes)))
        all_good_nodes = []
        for gn in known_valid_root_nodes:
            seg_list, _ = bfs_tree(self.morphology.node_by_id(gn), self.morphology)
            all_good_nodes = all_good_nodes + seg_list

        cleaned_morph = Morphology(
            all_good_nodes,
            node_id_cb=lambda node: node['id'],
            parent_id_cb=lambda node: node['parent'])

        results_dict = {}
        results_dict['morph'] = cleaned_morph
        results_dict['feature_dict'] = feat_dict
        results_dict['soma_dict'] = somas_dict

        return results_dict
