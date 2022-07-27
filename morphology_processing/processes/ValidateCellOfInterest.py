import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from morphology_processing.database_queries import query_for_63x_soma_coords
from neuron_morphology.swc_io import Morphology
from morph_utils.modifications import check_morph_for_segment_restructuring
from morphology_processing.processes import SomaConnections

class ValidateCellOfInterest:
    def __init__(self,morphology,specimen_id,soma_segmentation,cell_of_interest_threshold,soma_connection_threshold,**kwargs):
        self.morphology = morphology
        self.specimen_id = specimen_id
        self.soma_segmentation = soma_segmentation
        self.cell_of_interest_threshold = cell_of_interest_threshold
        self.soma_connection_threshold = soma_connection_threshold
        self.process_name = "validate_cell_of_interest"

    def process(self):


        """
        Load 63x soma drawing and check distance from autotrace soma. if its above threshold,
        Use x-y values from soma drawing and getting average z value within this x-y range
        """
        soma_63_xs, soma_63_ys = query_for_63x_soma_coords(self.specimen_id)
        mean_x = np.nanmean(soma_63_xs)  # mean()
        mean_y = np.nanmean(soma_63_ys)  # .mean()
        swc_soma = self.morphology.get_soma()

        if swc_soma != None:
            dist = (((mean_x - swc_soma['x']) ** 2) + ((mean_y - swc_soma['y']) ** 2)) ** 0.5
            if dist > self.cell_of_interest_threshold:
                print("Distance {} was above Threshold {}".format(dist, self.cell_of_interest_threshold))
                fig = plt.gcf()
                ax = plt.gca()
                ax.scatter([n['x'] for n in self.morphology.nodes()], [n['y'] for n in self.morphology.nodes()], c='grey', s=0.5)
                ax.scatter(soma_63_xs, soma_63_ys, c='blue', label='63x draw')
                ax.scatter(self.morphology.get_soma()['x'], self.morphology.get_soma()['y'],
                           c='orange', marker='X', s=50, label='BeforeAutotrace')

                max_drawn_soma_xs, min_drawn_soma_xs = max(soma_63_xs), min(soma_63_xs)
                max_drawn_soma_ys, min_drawn_soma_ys = max(soma_63_ys), min(soma_63_ys)

                soma_segmentation = self.soma_segmentation[['# x', 'y', 'z']]
                segmentation_coords_within_drawing = soma_segmentation.copy(deep=True)

                segmentation_coords_within_drawing = segmentation_coords_within_drawing[
                    segmentation_coords_within_drawing['# x'] < max_drawn_soma_xs]
                segmentation_coords_within_drawing = segmentation_coords_within_drawing[
                    segmentation_coords_within_drawing['# x'] > min_drawn_soma_xs]
                segmentation_coords_within_drawing = segmentation_coords_within_drawing[
                    segmentation_coords_within_drawing['y'] < max_drawn_soma_ys]
                segmentation_coords_within_drawing = segmentation_coords_within_drawing[
                    segmentation_coords_within_drawing['y'] > min_drawn_soma_ys]

                # No segmentation xy coordinates match with soma drawing. Either bad segmentation or updated soma drawing.
                if segmentation_coords_within_drawing.empty:
                    print("No seg coords found within soma drawing xy - range")
                    print("Using Average Z from segmentation")
                    new_x = mean_x
                    new_y = mean_y
                    new_z = soma_segmentation['z'].mean()
                else:
                    mean_segmentation_within_drawing = segmentation_coords_within_drawing.mean(axis=0)
                    new_x = mean_x
                    new_y = mean_y
                    new_z = mean_segmentation_within_drawing['z']

                self.morphology.get_soma()['x'] = new_x
                self.morphology.get_soma()['y'] = new_y
                self.morphology.get_soma()['z'] = new_z
                ax.scatter(self.morphology.get_soma()['x'], self.morphology.get_soma()['y'],
                           c='green', marker='X', s=5, label='FixedAutotrace')
                plt.legend()
                self.morphology, change_status = check_morph_for_segment_restructuring(self.morphology)

                # we moved the soma, restructured segments so leaf node nearest soma is root of each independent component
                # now we should check for soma connections with new soma
                # This can either be done in depth/will take more time with segmentation MIPs, or just use simple distance threshold
                proc = SomaConnections.SomaConnections(self.morphology, self.soma_connection_threshold)
                self.morphology = proc.process()['morph']

                print(
                    "We moved the soma. Now checking to see if we need to re-structuring the tree to make sure closest leaf node in each disconnected tree is the root")
                print("Changed made: {}".format(change_status))


            else:
                fig = plt.gcf()
                ax = plt.gca()
                ax.scatter([n['x'] for n in self.morphology.nodes()], [n['y'] for n in self.morphology.nodes()], c='grey', s=0.5)
                ax.scatter(soma_63_xs, soma_63_ys, c='blue', label='63x draw')
                ax.scatter(self.morphology.get_soma()['x'], self.morphology.get_soma()['y'],
                           c='orange', marker='X', s=50, label='Autotrace soma')
                plt.legend()

            results_dict = {}
            results_dict['morph'] = self.morphology
            results_dict['fig_ax'] = fig, ax
            results_dict['soma-drawing-to-swc-distance'] = dist
            return results_dict

        else:
            root_nodes = [n for n in self.morphology.nodes() if n['parent'] == -1]
            center = [mean_x, mean_y]
            sorted_root_nodes = sorted(root_nodes, key=lambda x: distance.euclidean([x['x'], x['y']], center))
            new_soma = sorted_root_nodes[0]
            self.morphology.node_by_id(new_soma['id'])['type'] = 1

            # Create Soma Connections\
            new_soma_children = []
            for no in sorted_root_nodes:
                if no != new_soma:
                    dist = distance.euclidean([no['x'], no['y'], no['z']], [new_soma['x'], new_soma['y'], new_soma['z']])
                    if dist < 100:
                        new_soma_children.append(no['id'])
            for no_id in new_soma_children:
                self.morphology.node_by_id(no_id)['parent'] = new_soma['id']

            morphology_with_soma = Morphology(self.morphology.nodes(),
                                              parent_id_cb=lambda x:x['parent'],
                                              node_id_cb=lambda x:x['id'])


            fig = plt.gcf()
            ax = plt.gca()
            ax.scatter([n['x'] for n in morphology_with_soma.nodes()], [n['y'] for n in morphology_with_soma.nodes()],
                       c='grey', s=0.5)
            ax.scatter(soma_63_xs, soma_63_ys, c='blue', label='63x draw')
            ax.scatter(morphology_with_soma.get_soma()['x'], morphology_with_soma.get_soma()['y'],
                       c='orange', marker='X', s=50, label='BeforeAutotrace')
            results_dict = {}
            results_dict['morph'] = morphology_with_soma
            results_dict['fig_ax'] = fig, ax
            results_dict['soma-drawing-to-swc-distance'] = None
            return results_dict

