from morph_utils.graph_traversal import bfs_tree
from sklearn.neighbors import KDTree
import numpy as np
from scipy import stats
from skeleton_keys.cmds import upright_corrected_swc
import os
from neuron_morphology.swc_io import morphology_from_swc

class ApicalClassification:
    def __init__(self,morphology,specimen_id,apical_classifier,**kwargs):
        self.morphology = morphology
        self.specimen_id = specimen_id
        self.apical_classifier = apical_classifier
        self.process_name = "apical_classification"

    def process(self):

        original_morph = self.morphology.clone()

        #TODO remove this I-O step
        tmp_ur_infile = "./temp_{}_raw.swc".format(self.specimen_id)
        tmp_ur_ofile = "./temp_{}_ur.swc".format(self.specimen_id)
        schem = {"swc_path": tmp_ur_infile,
                 "specimen_id": self.specimen_id,
                 "output_file": tmp_ur_ofile,
                 "surface_and_layers_file": None,
                 "correct_for_shrinkage": False,
                 "correct_for_slice_angle": False
                 }
        upright_corrected_swc.main(schem)
        uprighted_morpho = morphology_from_swc(tmp_ur_ofile)


        results_dict = {}

        feat_dict = get_segments_feat_dict(uprighted_morpho)
        if feat_dict is not None:
            for base_node_id in feat_dict.keys():
                vals = feat_dict[base_node_id].values()
                apical_classifier_input = np.asarray(list(vals)).reshape(1, len(vals))
                apical_classifier_input = np.nan_to_num(apical_classifier_input, 0)
                prediction = self.apical_classifier.predict(apical_classifier_input)[0]

                if prediction == 1:
                    base_node = original_morph.node_by_id(base_node_id)

                    # Not modifying axon segments because new spiny model is more accurate/relabel with axon segmentation
                    if base_node['type'] != 2:
                        this_segment, _ = bfs_tree(base_node, original_morph)
                        for no in this_segment:
                            no_id = no['id']
                            original_morph.node_by_id(no_id)['type'] = 4
                    else:
                        print("not modifying this segment because i trust the axon segmentation too much to overwrite it")

        results_dict['morph'] = original_morph

        return results_dict


def get_segments_feat_dict(morph):
    soma = morph.get_soma()
    root_seg_list = [n for n in morph.nodes() if n['parent'] == -1]
    root_seg_list = [n for n in root_seg_list if n['type'] != 1]

    segments_to_analyze = morph.get_children(soma) + root_seg_list
    segments_to_analyze = [s for s in segments_to_analyze if
                           all([np.isnan(s[ind]) == False for ind in ['x', 'y', 'z']])]

    if segments_to_analyze != []:
        indi_seg_start_coords = np.asarray([[n['x'], n['y'], n['z']] for n in segments_to_analyze])
        start_nodes_lookup_tree = KDTree(indi_seg_start_coords)

        whole_cell_xs = [n['x'] for n in morph.nodes()]
        whole_cell_ys = [n['y'] for n in morph.nodes()]

        whole_cell_x_range = max(whole_cell_xs) - min(whole_cell_xs)
        whole_cell_y_range = max(whole_cell_ys) - min(whole_cell_ys)

        results_dict = {}
        for start_node in segments_to_analyze:

            start_id = start_node['id']
            this_segment, segment_length = bfs_tree(start_node, morph)
            if segment_length > 5:
                results_dict[start_id] = {}
                xs = [no['x'] for no in this_segment]
                ys = [no['y'] for no in this_segment]
                stdev_x = np.std(xs)
                stdev_y = np.std(ys)

                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)

                delta_y = np.mean([no['y'] - morph.get_soma()['y'] for no in this_segment])

                slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
                slope = abs(slope)
                slope = angle_between((0, slope), (slope, slope))

                soma_to_coord_vector = [soma['x'] - start_node['x'], soma['y'] - start_node['y']]
                soma_to_upright_vector = [0, soma['y'] - start_node['y']]
                soma_coord_xy_angle = angle_between(soma_to_coord_vector, soma_to_upright_vector)

                # Get nearest start node and its features
                dists, inds = start_nodes_lookup_tree.query(
                    np.array([start_node['x'], start_node['y'], start_node['z']]).reshape(1, 3), k=2)
                nearest_node_coords = indi_seg_start_coords[inds[0][-1]]
                nearest_start_node = [n for n in segments_to_analyze if
                                      np.array_equal(nearest_node_coords, np.array([n['x'], n['y'], n['z']]))][0]
                nearest_neighbor_segment, _ = bfs_tree(nearest_start_node, morph)

                nearest_neighbor_xs = [no['x'] for no in nearest_neighbor_segment]
                nearest_neighbor_ys = [no['y'] for no in nearest_neighbor_segment]
                nearest_neighbor_stdev_x = np.std(nearest_neighbor_xs)
                nearest_neighbor_stdev_y = np.std(nearest_neighbor_ys)
                nearest_neighbor_x_range = max(nearest_neighbor_xs) - min(nearest_neighbor_xs)
                nearest_neighbor_y_range = max(nearest_neighbor_ys) - min(nearest_neighbor_ys)

                nearest_neighbor_delta_y = np.mean([no['y'] - morph.get_soma()['y'] for no in nearest_neighbor_segment])

                nearest_neighbor_slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
                nearest_neighbor_slope = abs(nearest_neighbor_slope)
                nearest_neighbor_slope = angle_between((0, nearest_neighbor_slope),
                                                       (nearest_neighbor_slope, nearest_neighbor_slope))

                soma_to_coord_vector_nn = [soma['x'] - nearest_start_node['x'], soma['y'] - nearest_start_node['y']]
                soma_to_upright_vector_nn = [0, soma['y'] - nearest_start_node['y']]
                soma_coord_xy_angle_nn = angle_between(soma_to_coord_vector_nn, soma_to_upright_vector_nn)

                results_dict[start_id]['stdev_x'] = stdev_x / x_range
                results_dict[start_id]['stdev_y'] = stdev_y / y_range
                results_dict[start_id]['x_range'] = x_range / whole_cell_x_range
                results_dict[start_id]['y_range'] = y_range / whole_cell_y_range
                results_dict[start_id]['delta_y'] = delta_y / whole_cell_y_range
                results_dict[start_id]['soma_xy_angle'] = soma_coord_xy_angle
                results_dict[start_id]['Slope_times_delta_y'] = slope * (delta_y / whole_cell_y_range)

                # results_dict[start_id]['stdev_x_nearest_neighbor'] = nearest_neighbor_stdev_x/nearest_neighbor_x_range
                # results_dict[start_id]['stdev_y_nearest_neighbor'] = nearest_neighbor_stdev_y/nearest_neighbor_y_range
                results_dict[start_id]['x_range_nearest_neighbor'] = nearest_neighbor_x_range / whole_cell_x_range
                results_dict[start_id]['y_range_nearest_neighbor'] = nearest_neighbor_y_range / whole_cell_y_range
                results_dict[start_id]['delta_y_nearest_neighbor'] = nearest_neighbor_delta_y / whole_cell_y_range
                results_dict[start_id]['soma_xy_angle_nearest_neighbor'] = soma_coord_xy_angle_nn
                results_dict[start_id]['Slope_times_delta_y_nearest_neighbor'] = nearest_neighbor_slope * (
                            nearest_neighbor_delta_y / whole_cell_y_range)
        return results_dict

    else:
        print("No Segments To Analyze, this list:")
        print(morph.get_children(soma) + [n for n in morph.nodes() if (n['parent'] == -1) and (n['type'] != 1)])
        print("Got reduced to:")
        print([s for s in segments_to_analyze if all([np.isnan(s[ind]) == False for ind in ['x', 'y', 'z']])])
        print("after:")
        print(" [s for s in segments_to_analyze if all([np.isnan(s[ind])==False for ind in ['x','y','z']])]")
        return None


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
