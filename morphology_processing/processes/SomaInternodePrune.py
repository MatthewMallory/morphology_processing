import numpy as np
from numpy.linalg import eig, inv
from scipy.ndimage.measurements import label
from neuron_morphology.morphology import Morphology


class SomaInternodePrune:
    def __init__(self, morphology, image_shape_xyz, soma_segmentation, z_scale_factor, x_res, y_res, **kwargs):
        self.morphology = morphology
        self.image_shape_xyz = image_shape_xyz
        self.soma_segmentation = soma_segmentation
        self.z_scale_factor = z_scale_factor
        self.x_res = x_res
        self.y_res = y_res
        self.process_name = "soma_internode_prune"

    def process(self):
        """
        Worker function for this step of processing.This function will:
        0. Find Soma Radius
        1. Remove soma connections that are inside the soma radius.
        2. Add soma radius to the morphology
        3. remove nodes that fall within soma radius
    
        Note because we are trying to include soma radius z-dimension, all distances should be converted to um
        """

        # calc a soma radius
        radius, soma_ellipse_axes = get_soma_radius(self.morphology,
                                                    self.image_shape_xyz,
                                                    self.soma_segmentation,
                                                    self.z_scale_factor)
        print('     Soma radius = {}'.format(radius))
        # See if any soma children are 2x the radius of the soma away. #In scenarios where we have a very small radius
        # For some of the cells, the biocytin leaked out and we want to make sure we dont further remove any nodes by using a default thresh of 30
        if radius > 20:
            thresh = 2 * radius
        else:
            thresh = 50

        soma = self.morphology.get_soma()
        soma_children = self.morphology.children_of(soma)
        soma_children_to_disconnect = []
        for child in soma_children:

            dist_from_soma = ((((soma['x'] * self.x_res) - (child['x'] * self.x_res)) ** 2) +
                              (((soma['y'] * self.y_res) - (child['y'] * self.y_res)) ** 2) +
                              (((soma['z'] * self.z_scale_factor) - (child['z'] * self.z_scale_factor)) ** 2)) ** 0.5
            if dist_from_soma > thresh:
                soma_children_to_disconnect.append(child['id'])

        print(
            '     Disconnecting {} soma children that are more than {}um from soma'.format(
                len(soma_children_to_disconnect),
                thresh))
        for dis_no in soma_children_to_disconnect:
            self.morphology.node_by_id(dis_no)['parent'] = -1

        # Wierd edge nodes, Slice artifact removal.
        # remove any nodes that are within this radius
        nodes_inside_soma = []
        for child in soma_children:
            dist_from_soma = ((((soma['x'] * self.x_res) - (child['x'] * self.x_res)) ** 2) +
                              (((soma['y'] * self.y_res) - (child['y'] * self.y_res)) ** 2) +
                              (((soma['z'] * self.z_scale_factor) - (child['z'] * self.z_scale_factor)) ** 2)) ** 0.5
            # dist_from_soma = (((soma['x']-child['x'])**2) + ((soma['y']-child['y'])**2) + ((soma['z']-child['z'])**2))**0.5
            if dist_from_soma < radius:
                nodes_inside_soma.append(child)

        # Visit each soma-child node that is within calculated radius of soma
        # Adding each
        nodes_to_exclude = set()
        terminal_nodes = set()
        print('     Found {} soma children inside soma radius \n'.format(len(nodes_inside_soma)))
        for no in nodes_inside_soma:
            dfs_removing_nodes(self.morphology,
                               no,
                               nodes_to_exclude,
                               radius,
                               terminal_nodes,
                               self.z_scale_factor,
                               self.x_res,
                               self.y_res)

        # When deleting nodes inside the soma we still want the node to soma conneciton. These terminal nodes represent
        # The nodes where a soma_segmentationment transitions from inside to outside the soma.
        for noid in terminal_nodes:
            self.morphology.node_by_id(noid)['parent'] = self.morphology.get_soma()['id']

        keeping_nodes = [n for n in self.morphology.nodes() if n['id'] not in nodes_to_exclude]

        fixed_soma_and_edge_morph = Morphology(
            keeping_nodes,
            node_id_cb=lambda node: node['id'],
            parent_id_cb=lambda node: node['parent'])

        fixed_soma_and_edge_morph.get_soma()['radius'] = radius

        results_dict = {}
        results_dict['morph'] = fixed_soma_and_edge_morph
        results_dict['soma_ellipse_axes'] = soma_ellipse_axes
        results_dict['radius'] = radius
        return results_dict


def dfs_removing_nodes(morph, node, visited, radius, terminal_nodes, z_scale_factor,x_res,y_res):
    """
    Remove nodes in a dfs manor until the distance is outside
    the soma radius. The terminal nodes are nodes that now will need to connect to the
    soma because of their ancestoral removal
    """
    soma = morph.get_soma()
    if node['id'] not in visited:
        visited.add(node['id'])
        for ch_no in morph.get_children(node):
            dist_to_soma = ((((soma['x'] * x_res) - (ch_no['x'] * x_res)) ** 2) +
                            (((soma['y'] * y_res) - (ch_no['y'] * y_res)) ** 2) +
                            (((soma['z'] * z_scale_factor) - (ch_no['z'] * z_scale_factor)) ** 2)) ** 0.5
            if dist_to_soma < radius:
                dfs_removing_nodes(morph, ch_no, visited, radius, terminal_nodes, z_scale_factor,x_res,y_res)
            else:
                terminal_nodes.add(ch_no['id'])
    return visited, terminal_nodes


def get_soma_radius(morphology, image_shape_xyz, soma_soma_segmentationmentation, z_scale_factor):
    """
    This function will fit an ellipse to the xy and yz plane of soma cloud. It will take the axes and reduce to
    three (removing redundant y).

    Returns: radius, ellipse x y and z axes
    """
    soma_ax1 = get_soma_axes_xy(morphology, image_shape_xyz, soma_soma_segmentationmentation)
    soma_ax2 = get_soma_axes_yz(morphology, image_shape_xyz, soma_soma_segmentationmentation, z_scale_factor)
    three_axes = reduce_to_unique_axes(soma_ax1, soma_ax2)
    # return np.mean(three_axes),three_axes
    return np.mean(soma_ax1), three_axes


def reduce_to_unique_axes(ax1, ax2):
    """
    This funtion will take in the four axes values found in xy and yz plane
    ellipse fitting and reduce the two most similar (y axis) their average.
    returns: average of the three values and the three axes values
    """
    all_ax_vals = np.concatenate([ax1, ax2])
    for i in range(0, len(all_ax_vals)):
        curr_val = all_ax_vals[i]
        other_vals = [a for a in all_ax_vals if a != curr_val]
        if len(other_vals) != len(all_ax_vals) - 1:
            print('there are identical values')
            three_vals = np.unique(all_ax_vals)
        else:
            all_diffs = {}
            for i, val1 in enumerate(all_ax_vals):
                for j, val2 in enumerate(all_ax_vals):
                    if (i != j) and ((j, i) not in list(all_diffs.keys())):
                        all_diffs[i, j] = abs(val1 - val2)
            nearest_pair_ind = [k for k, v in all_diffs.items() if v == min(all_diffs.values())][0]
            unique_indices = [n for n in range(0, len(all_ax_vals)) if n not in nearest_pair_ind]
            three_vals = [(all_ax_vals[nearest_pair_ind[0]] + all_ax_vals[nearest_pair_ind[1]]) / 2] + [all_ax_vals[i]
                                                                                                        for i in
                                                                                                        unique_indices]

    return three_vals


def get_soma_axes_yz(morphology, image_shape_xyz, soma_soma_segmentationmentation, z_scale_factor):
    """
    Get minor and major axes values in yz plane
    """
    try:
        soma_coord = (morphology.get_soma()['x'], morphology.get_soma()['y'], morphology.get_soma()['z'])
        xs, ys, zs = soma_soma_segmentationmentation['# x'].values, soma_soma_segmentationmentation['y'].values, \
                     soma_soma_segmentationmentation['z'].values
        somas_2d = np.zeros((image_shape_xyz[1], image_shape_xyz[2]), dtype=np.bool)
        for z, y in zip(zs, ys):
            somas_2d[y, z] = True
        structure = np.ones((3, 3), dtype=np.int)
        labeled, num_comps = label(somas_2d, structure)
        soma_cc_label = labeled[int(soma_coord[1]), int(soma_coord[2])]
        soma_coords_arrays = np.where(labeled == soma_cc_label)
        soma_zs = soma_coords_arrays[1]
        soma_ys = soma_coords_arrays[0]
        soma_coords = list(zip(soma_zs, soma_ys))
        # Get 2D Area zy
        perimeter_coords = []
        if (max(soma_ys) - min(soma_ys)) < 300:
            for y in range(min(soma_ys), max(soma_ys) + 1):
                coords_at_y = [c for c in soma_coords if c[1] == y]
                min_z = min([c[0] for c in coords_at_y])
                max_z = max([c[0] for c in coords_at_y])
                perimeter_coords.append((y, min_z))
                perimeter_coords.append((y, max_z))
            perimeter_coords = np.array(perimeter_coords).T

            arc = 5
            R = np.arange(0, arc * np.pi, 0.01)
            a = fitEllipse(perimeter_coords[0] * 0.1144, perimeter_coords[1] * z_scale_factor)
            center = ellipse_center(a)
            phi = ellipse_angle_of_rotation(a)
            phi = ellipse_angle_of_rotation(a)
            axes = ellipse_axis_length(a)
            return axes  # np.mean(axes)
        else:
            print('issue aligning soma from swc and image soma. perhaps mislabeled soma in swc return defaul')
            return np.array([1, 1])
    except:
        print('Issue Finding Soma For Specimen. Returning [2,3]')
        return np.array([1, 1])


def get_soma_axes_xy(morphology, image_shape_xyz, soma_soma_segmentationmentation):
    """
    Get minor and major axes values in xy plane
    """
    try:
        soma_coord = (morphology.get_soma()['x'], morphology.get_soma()['y'], morphology.get_soma()['z'])
        xs, ys = soma_soma_segmentationmentation['# x'].values, soma_soma_segmentationmentation['y'].values
        somas_2d = np.zeros((image_shape_xyz[1], image_shape_xyz[0]), dtype=np.bool)
        for x, y in zip(xs, ys):
            somas_2d[y, x] = True
        structure = np.ones((3, 3), dtype=np.int)
        labeled, num_comps = label(somas_2d, structure)
        soma_cc_label = labeled[int(soma_coord[1]), int(soma_coord[0])]
        soma_coords_arrays = np.where(labeled == soma_cc_label)
        soma_xs = soma_coords_arrays[0]
        soma_ys = soma_coords_arrays[1]
        soma_coords = list(zip(soma_xs, soma_ys))
        # Get 2D Area XY
        perimeter_coords = []
        if (max(soma_xs) - min(soma_xs)) < 300:
            for x in range(min(soma_xs), max(soma_xs) + 1):
                coords_at_x = [c for c in soma_coords if c[0] == x]
                min_y = min([c[-1] for c in coords_at_x])
                max_y = max([c[-1] for c in coords_at_x])
                perimeter_coords.append((x, min_y))
                perimeter_coords.append((x, max_y))
            perimeter_coords = np.array(perimeter_coords).T
            arc = 5
            R = np.arange(0, arc * np.pi, 0.01)
            a = fitEllipse(perimeter_coords[0] * 0.1144, perimeter_coords[1] * 0.1144)
            center = ellipse_center(a)
            phi = ellipse_angle_of_rotation(a)
            phi = ellipse_angle_of_rotation(a)
            axes = ellipse_axis_length(a)
            return axes  # np.mean(axes)
        else:
            print('issue aligning soma from swc and image soma. perhaps mislabeled soma in swc return defaul')
            return np.array([1, 1])
    except:
        print('Issue Finding Soma For Specimen. Returning [1,2]')
        return np.array([1, 1])


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(E)
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])
