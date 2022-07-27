
import shutil
from scipy.spatial.distance import cdist


import numpy as np
import pandas as pd
from collections import deque
import tifffile as tif
import itertools
from operator import add
import os
from collections import defaultdict
from tifffile import imsave
from scipy import ndimage as ndi
from neuron_morphology.swc_io import morphology_from_swc, morphology_to_swc
from scipy.spatial import distance

class ConnectSegments:
    def __init__(self, morphology, specimen_id, specimen_dir, z_scale_factor,**kwargs):
        self.morphology = morphology
        self.specimen_id = specimen_id
        self.specimen_dir = specimen_dir
        self.z_scale_factor = z_scale_factor
        self.process_name = "connect_broken_segments"

    def process(self):
        return connection_algorithm_by_compartment(self.morphology,
                                                   self.specimen_id,
                                                   self.specimen_dir,
                                                   self.z_scale_factor)

def connection_algorithm_by_compartment(morph, specimen_id, specimen_dir, z_scale_factor, **kwargs):
    print(
        "This algorithm aims to use Olga's original method but restrict connections to occur within types. i.e. only connect dendrite to dendrite or axon to axon")
    temp_input_morph = os.path.join(specimen_dir, f"{specimen_id}_temp_conenction_swc.swc")
    morphology_to_swc(morph, temp_input_morph)
    res = process_specimen(specimen_dir, temp_input_morph, z_scale_factor)

    print("The Status of process specimens = ", res)

    results_dict = {}
    if res == True:
        input_skeleton_dir = os.path.join(specimen_dir, 'Skeleton1')
        output_swc = os.path.join(specimen_dir, f"{specimen_id}_after_connections.swc")
        connected_morph = run_skeleton_to_swc(specimen_dir, input_skeleton_dir, temp_input_morph, output_swc)
        connected_morph.get_soma()['x'] = morph.get_soma()['x']
        connected_morph.get_soma()['y'] = morph.get_soma()['y']
        connected_morph.get_soma()['z'] = morph.get_soma()['z']
        connected_morph_soma = re_attach_somas(morph, connected_morph, specimen_id)

        if os.path.exists(input_skeleton_dir):
            shutil.rmtree(input_skeleton_dir)

        results_dict['morph'] = connected_morph_soma
    else:
        results_dict['morph'] = morph

    return results_dict


def re_attach_somas(m1, m2, sp):
    """
    m1: morph with soma connections
    m2: morph whose soma is unconneted
    Used to fix issue of soma detachment after connection algorithms
    """
    coords_attached_to_soma = [(n['x'], n['y'], n['z']) for n in m1.get_children(m1.get_soma())]
    for coord in coords_attached_to_soma:
        coord_to_fix = [n for n in m2.nodes() if (n['x'] == coord[0]) and (n['y'] == coord[1]) and (n['z'] == coord[2])]
        if coord_to_fix == []:
            continue
        else:
            no_id = coord_to_fix[0]['id']
            m2.node_by_id(no_id)['parent'] = m2.get_soma()['id']
    temp_file = f'{sp}.swc'
    morphology_to_swc(m2, temp_file)
    new_morph = morphology_from_swc(temp_file)
    os.remove(temp_file)
    return new_morph


def load_swc_type_restricted(filepath, node_type_restriction):
    node_type_restriction = [int(c) for c in node_type_restriction]
    # load swc file as a N X 7 numpy array
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:

            if not l.startswith('#'):
                cells = l.split(' ')

                if len(cells) == 7:
                    node_type = cells[1]
                    if int(node_type) in node_type_restriction:
                        cells = [float(c) for c in cells]
                        swc.append(cells)

                elif len(cells) == 8:
                    node_type = cells[1]
                    if int(node_type) in node_type_restriction:
                        cells = [float(c) for c in cells[0:7]]
                        swc.append(cells)

    return np.array(swc)


def calculate_vector(data):
    datamean = data.mean(axis=0)  # calculate mean
    _, _, vv = np.linalg.svd(data - datamean)  # svd of mean-centered data
    return vv[0]  # return 1st principal component


def connect_segments(trace, z_val, cthres=0.26, dthres=30):
    # trace - input (i.e. auto) trace
    # use only head-tail connections
    # select connections based on angle before calculating min distance (cthres=0.26)
    # select connections based on distance (dthres = 30 (um))
    print('cthres', cthres)  # angle threshold
    print('dthres', dthres)  # distance threshold
    # Step 1
    # find all connected components (heads and tails) # id,type,x,y,z,r,pid
    # for all connected component find all ends (head is excluded)
    select1 = np.where(trace[:, 6] == -1)
    heads = select1[0][1:]
    tails = select1[0][2:] - 1
    tails = np.append(tails, trace.shape[0] - 1)
    num_cc = heads.shape[0]

    # for every cc add head and ends
    ends_list = []  # ends of each cc
    ccp_list = []  # head and ends of each cc
    ends_all = []  # all ends
    ccp = []  # all ccp
    for i in range(num_cc):
        ends_j = []
        ccp_j = [heads[i]]
        ccp.append(heads[i])
        for j in range(heads[i], tails[i] + 1):
            idx = np.where(trace[j, 0] == trace[heads[i]:tails[i] + 1, 6])[0]
            if not idx.size:
                ends_all.append(j)
                ends_j.append(j)
                ccp.append(j)
                ccp_j.append(j)
        ends_list.append(ends_j)
        ccp_list.append(ccp_j)
    ends = np.array(ends_all)
    print('# of cc', num_cc)
    print('# of heads', heads.shape[0])
    print('# of ends', ends.shape[0])
    print('# of ccp', len(ccp))

    # Step 2
    # calculate vector for every ccp and for every connection
    # for every ccp find adjucent N ponts (N=9) using pid and calculate vector (normalized)
    xyz_pxl = np.array([0.1144, 0.1144, z_val])
    vectors_list = []  # vectors for each cc
    vectors = []  # all vectors
    num_points = 30
    for i in range(len(ccp_list)):  # loop through all cc
        cc_ind = np.arange(heads[i], tails[i] + 1)
        vectors_i = []
        for j in range(len(ccp_list[i])):  # loop through all ccp for given cc
            ccp_ind = [ccp_list[i][j]]
            sid = int(trace[ccp_list[i][j], 0])
            pid = int(trace[ccp_list[i][j], 6])
            if pid == -1:
                for n in range(num_points):
                    idx = np.where(trace[cc_ind, 6] == sid)[0]
                    if idx.shape[0] > 0:
                        idx = idx[0]
                        ccp_ind.append(cc_ind[idx])
                        sid = int(trace[cc_ind[idx], 0])
            else:
                for n in range(num_points):
                    idx = np.where(trace[cc_ind, 0] == pid)[0]
                    if idx.shape[0] > 0:
                        idx = idx[0]
                        ccp_ind.append(cc_ind[idx])
                        pid = int(trace[cc_ind[idx], 6])
            vector = calculate_vector(trace[ccp_ind, 2:5] * xyz_pxl)  # take into account voxel anisotropy
            # fix wrong orientation (sign) of vector
            vect_diff = (trace[ccp_ind[-1], 2:5] * xyz_pxl
                         - trace[ccp_ind[0], 2:5] * xyz_pxl) / np.linalg.norm(trace[ccp_ind[-1], 2:5] * xyz_pxl
                                                                              - trace[ccp_ind[0], 2:5] * xyz_pxl)
            if np.dot(-vector, vect_diff) > np.dot(vector, vect_diff):
                vector *= -1
            vectors.append(vector)
            vectors_i.append(vector)
        vectors_list.append(vectors_i)
    print('vectors', len(vectors))
    print('vectors_list', len(vectors_list))

    # create vectors array for ends only
    vectors_ends = []
    for v in vectors_list:
        for v1 in v[1:]:
            vectors_ends.append(v1)
    print(len(vectors_ends))
    if len(vectors_ends) > 0:

        for i, v in enumerate(vectors_ends):
            if i == 0:
                vectors_ends_xyz = v
            else:
                vectors_ends_xyz = np.vstack((vectors_ends_xyz, v))

    else:
        return np.array([]), np.array([])

    #  Step 3
    # for every cc find connections
    # modified by sorting all connections based on angle (cthres = 0.26) before calculating min dist
    ccp_idx_list = []  # list of connection start points for every cc
    min_dist_idx_list = []  # list of connection end points for every cc
    min_dist_list = []  # list of distances for every cc
    for i in range(num_cc):
        ends_exclude = [p for p in ends_all if p not in ends_list[i]]  # exclude ends of given cc
        idx_exclude = [ends_all.index(p) for p in ends_all if p not in ends_list[i]]
        # take into account voxel anisotropy
        pdist = cdist(trace[heads[i:i + 1], 2:5] * xyz_pxl, trace[ends_exclude, 2:5] * xyz_pxl, 'euclidean')
        # for every connection calculate collinearity factors c1, c2
        # sort all connections: remove all where c1 or c2 < cthresh
        # find connection with min_dist from selected ones
        min_dist = []
        min_dist_idx = []
        cc_idx = []
        ccp_list_new = []
        # for every connection calculate normalized vector
        cvect = trace[ends_exclude, 2:5] * xyz_pxl - trace[heads[i], 2:5] * xyz_pxl  # take into account voxel anisotr
        cvect_norm = np.linalg.norm(cvect, axis=1)
        cvect = cvect / np.stack((cvect_norm, cvect_norm, cvect_norm), axis=1)
        c_all = np.zeros((len(idx_exclude), 2))
        for n, ind in enumerate(idx_exclude):
            # calculate 'collinearity factor'
            c1 = np.dot(-vectors_list[i][0], cvect[n])
            c2 = np.dot(vectors_ends_xyz[ind], cvect[n])
            c_all[n, 0] = c1
            c_all[n, 1] = c2
        select = np.where((c_all[:, 0] > cthres) & (c_all[:, 1] > cthres))
        if select[0].shape[0] > 0:
            ccp_list_new.append(heads[i])
            min_dist_i = np.min(pdist[0, select[0]])
            min_dist.append(min_dist_i)
            idx = np.where(pdist[0, :] == min_dist_i)[0][0]
            min_dist_idx_i = ends_exclude[idx]
            min_dist_idx.append(min_dist_idx_i)
            # check if there are multiple connections to the same cc
            for k, l in enumerate(ccp_list):
                if min_dist_idx_i in l:
                    cc_idx.append(k)  # find all cc to which this connection points/ends
        min_dist = np.array(min_dist)
        min_dist_idx = np.array(min_dist_idx, dtype=np.int64)  # ensure dtype when []
        ccp_idx = np.array(ccp_list_new, dtype=np.int64)  # exclude ccp without connection
        min_dist_list.append(min_dist)
        min_dist_idx_list.append(min_dist_idx)
        ccp_idx_list.append(ccp_idx)

    print('min_dist_list', len(min_dist_list), 'min_dist_idx_list', len(min_dist_idx_list),
          'ccp_idx_list', len(ccp_idx_list))
    ccp_idx_all = np.concatenate(ccp_idx_list)
    min_dist_idx_all = np.concatenate(min_dist_idx_list)
    min_dist_all = np.concatenate(min_dist_list)
    # print('ccp_idx_all', ccp_idx_all.shape, 'min_dist_idx', min_dist_idx_all.shape, 'min_dist_all',
    #       min_dist_all.shape, np.min(min_dist_all), np.max(min_dist_all), np.mean(min_dist_all))

    # Step 4
    # find min_dist_idx with multiple connections and remove longer
    min_dist_idx_unique = np.unique(min_dist_idx_all)
    idx_remove = []
    for i in range(min_dist_idx_unique.shape[0]):
        idx = np.where(min_dist_idx_unique[i] == min_dist_idx_all)[0]
        if idx.shape[0] > 1:
            for k, j in enumerate(idx):
                if k != np.argmin(min_dist_all[idx]):
                    idx_remove.append(j)
    idx_remove.sort()
    idx_keep = [l for l in range(ccp_idx_all.shape[0]) if l not in idx_remove]
    print('total connections', ccp_idx_all.shape[0])
    print('removed connections', len(idx_remove))
    print('remained connections', len(idx_keep))

    # connections endponts e1,e2
    e1 = ccp_idx_all[idx_keep]
    e2 = min_dist_idx_all[idx_keep]
    print('e1', e1.shape, e1.dtype, 'e2', e2.shape, e2.dtype)

    # min dist keep
    min_dist_keep_all = min_dist_all[idx_keep]
    print('min_dist_keep_all', min_dist_keep_all.shape)

    # select distance below threshold
    select2 = np.where(min_dist_keep_all < dthres)

    # connections endponts p1,p2
    p1 = e1[select2[0]]
    p2 = e2[select2[0]]
    print('p1', p1.shape, 'p2', p2.shape)

    return p1, p2


def connect_voxels(trace, idx1, idx2):
    t1, x1, y1, z1 = (trace[idx1, 1:5]).astype(np.int32)
    t2, x2, y2, z2 = (trace[idx2, 1:5]).astype(np.int32)
    t = np.max(np.array([t1, t2]))  # if at least one end is 3, type is 3 (dendrite)
    nodes = []
    dim_max = np.argmax(np.array(([abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)])))
    print("Dim Max = {}".format(dim_max))
    print("x1,y1,z1 = {} {} {}".format(x1, y1, z1))
    print("x2,y2,z2 = {} {} {}".format(x2, y2, z2))
    if dim_max == 0:

        # y = a1*x + b1
        # z = a2*x + b2
        a1 = (y1 - y2) / (x1 - x2)
        b1 = y1 - a1 * x1
        a2 = (z1 - z2) / (x1 - x2)
        b2 = z1 - a2 * x1
        if x2 > x1:
            for x in range(x1 + 1, x2):  # exclude ends
                y = np.round(a1 * x + b1)
                z = np.round(a2 * x + b2)
                nodes.append(np.array([t, x, y, z]))
        elif x1 > x2:
            for x in range(x2 + 1, x1):  # exclude ends
                y = np.round(a1 * x + b1)
                z = np.round(a2 * x + b2)
                nodes.append(np.array([t, x, y, z]))
    elif dim_max == 1:
        # x = a1*y + b1, z = a2*y + b2
        a1 = (x1 - x2) / (y1 - y2)
        b1 = x1 - a1 * y1
        a2 = (z1 - z2) / (y1 - y2)
        b2 = z1 - a2 * y1
        if y2 > y1:
            for y in range(y1 + 1, y2):
                x = np.round(a1 * y + b1)
                z = np.round(a2 * y + b2)
                nodes.append(np.array([t, x, y, z]))
        elif y1 > y2:
            for y in range(y2 + 1, y1):
                x = np.round(a1 * y + b1)
                z = np.round(a2 * y + b2)
                nodes.append(np.array([t, x, y, z]))
    else:
        # x = a1*z + b1, y = a2*z + b2
        a1 = (x1 - x2) / (z1 - z2)
        b1 = x1 - a1 * z1
        a2 = (y1 - y2) / (z1 - z2)
        b2 = y1 - a2 * z1
        if z2 > z1:
            for z in range(z1 + 1, z2):
                x = np.round(a1 * z + b1)
                y = np.round(a2 * z + b2)
                nodes.append(np.array([t, x, y, z]))
        elif z1 > z2:
            for z in range(z2 + 1, z1):
                x = np.round(a1 * z + b1)
                y = np.round(a2 * z + b2)
                nodes.append(np.array([t, x, y, z]))
    print("Len nodes", len(nodes))
    if len(nodes) == 0:
        nodes.append(np.array([t1, x1, y1, z1]))
        nodes.append(np.array([t2, x2, y2, z2]))
    nodes = np.stack(nodes)
    return nodes


def create_skeleton(specimen_dir, trace, new_nodes):
    print(specimen_dir, trace.shape, new_nodes.shape)
    print('specimen_dir:', specimen_dir)

    # create labeled skeleton
    arbor = np.concatenate((trace[1:, 2:5], new_nodes[:, 1:]))  # exclude soma node
    node_type = np.concatenate((trace[1:, 1], new_nodes[:, 0]))

    # save labeled skeleton
    np.savetxt(os.path.join(specimen_dir, 'Segmentation_skeleton_labeled2.csv'),
               np.column_stack((arbor, node_type)), fmt='%u', delimiter=',', header='x,y,z,type')

    # find stack size
    z_max = -1
    for ch in [2, 3]:
        filename = os.path.join(specimen_dir, 'Segmentation_ch{}.csv'.format(ch))
        if not os.path.exists(filename):
            filename = os.path.join(specimen_dir, 'Left_Segmentation_ch{}.csv'.format(ch))
        df = pd.read_csv(filename)
        this_zmax = max(df['z'])
        if this_zmax > z_max:
            z_max = this_zmax
    z_max += 1

    # filename = os.path.join(specimen_dir, 'Segmentation_ch2.csv')
    # if not os.path.exists(filename):
    #     filename = os.path.join(specimen_dir, 'Left_Segmentation_ch2.csv')
    # df = pd.read_csv(filename)
    # # z = df.values[-1,2]
    # z = df['z'].max()    ###### New Line
    # z_max=z+1###### New Line
    # z_max = max(z+1, int(np.max(arbor[:,2])) + 1)
    filename = os.path.join(specimen_dir, 'MAX_Single_Tif_Images.tif')
    backup_filename = os.path.join(specimen_dir, "MAX_Segmentation_ch1.tif")
    if os.path.exists(filename):
        img = tif.imread(filename)
        cell_stack_size = z_max, img.shape[0], img.shape[1]  # z is estimation <= true size
        print('stack size', cell_stack_size)
    elif os.path.exists(backup_filename):
        img = tif.imread(backup_filename)
        cell_stack_size = z_max, img.shape[0], img.shape[1]  # z is estimation <= true size
        print('stack size', cell_stack_size)
    else:
        img_left = tif.imread(os.path.join(specimen_dir, 'MAX_Left_Segmentation_ch1.tif'))
        img_right = tif.imread(os.path.join(specimen_dir, 'MAX_Right_Segmentation_ch1.tif'))
        cell_stack_size = z_max, img_left.shape[0], img_right.shape[1] + img_left.shape[
            1]  # z is estimation <= true size

    print('stack size', cell_stack_size)

    # create stack
    stack = np.zeros(cell_stack_size, dtype=np.uint8)
    print('stack', stack.shape, stack.dtype)
    for a in arbor:
        stack[int(a[2]), int(a[1]), int(a[0])] = 1
    print('stack', stack.shape, stack.dtype)

    # save stack as multiple tif files
    print('saving stack')
    savedir = os.path.join(specimen_dir, 'Skeleton1')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    for i in range(stack.shape[0]):
        tif.imsave(os.path.join(savedir, '%03d.tif' % (i)), stack[i, :, :])


def process_specimen(specimen_dir, swc_file, z_val):
    print('specimen_dir:', specimen_dir)
    print(swc_file)

    full_trace = load_swc_type_restricted(swc_file, [1, 2, 3, 4])

    nodes_list = []
    no_connections = True
    for compartment in [2, 3]:
        restrictions = [1, compartment]

        trace = load_swc_type_restricted(swc_file, restrictions)

        print("Compartment {}".format(compartment), trace.shape, trace.dtype)

        # sort trace based on id (0th column) (for cases when nodes of swc file are disordered)
        trace = trace[trace[:, 0].argsort()]

        print('connecting segments')
        p1, p2 = connect_segments(trace, z_val)
        print("p1 and p2 lengths", len(p1), len(p2))
        # for every connection create voxels
        if len(p1) > 0:
            no_connections = False
            print("{} Connections found for type {}".format(len(p1), compartment))
            for i in range(p1.shape[0]):
                nodes = connect_voxels(trace, p1[i], p2[i])
                nodes_list.append(nodes)
            print(len(nodes_list), nodes_list[0].shape)
            nodes_all = np.concatenate(nodes_list)
            print(nodes_all.shape)

    if no_connections:
        print("P1 len == 0 for all compartments. aka no connections to be made, returning original morphology")
        return False

    else:
        print('creating_skeleton')
        create_skeleton(specimen_dir, full_trace, nodes_all)
        return True





### Skeleton To SWC


def run_skeleton_to_swc(specimen_dir, input_skeleton_dir, input_swc, output_swc):
    skeleton_to_swc_parallel(specimen_dir, input_skeleton_dir, input_swc, output_swc)
    connected_morph = morphology_from_swc(output_swc)
    return connected_morph


def stitch_skeleton_dir(specimen_dir, skeleton_dir):
    left_skeleton_dir = os.path.join(specimen_dir, 'Left_Skeleton')
    right_skeleton_dir = os.path.join(specimen_dir, 'Right_Skeleton')

    left_files = [f for f in os.listdir(left_skeleton_dir) if f.endswith('.tif')]
    left_files.sort()
    right_files = [f for f in os.listdir(right_skeleton_dir) if f.endswith('.tif')]
    right_files.sort()

    for img in left_files:
        left_img_path = os.path.join(left_skeleton_dir, img)
        right_img_path = os.path.join(right_skeleton_dir, img)

        left_img = tif.imread(left_img_path)  # ,cv2.IMREAD_UNCHANGED)
        right_img = tif.imread(right_img_path)  # ,cv2.IMREAD_UNCHANGED)
        combined_img = np.append(left_img, right_img, axis=1)
        imsave(os.path.join(skeleton_dir, img), combined_img)


def euclid_dist(centroid, coord):
    return (((centroid[0] - coord[0]) ** 2) + ((centroid[1] - coord[1]) ** 2) + ((centroid[2] - coord[2]) ** 2)) ** .5


# this function uses BFS to assign parent child relationships in the neighbors dict
# BFS was used rather than DFS to handle instances of cycles without issue
def assign_parent_child_relation(start_node, start_nodes_parent, parent_dict, neighbors_dict):
    parent_dict[start_node] = start_nodes_parent
    queue = deque([start_node])
    while len(queue) > 0:
        current_node = queue.popleft()
        #         print('')
        #         print('current node {} {}'.format(node_dict[current_node], current_node))
        my_connections = neighbors_dict[current_node]
        #         [print('my connections {} {}'.format(node_dict[ss], ss)) for ss in my_connections]
        for node in my_connections:
            #             print('checking node {}'.format(node_dict[node]))
            if node not in parent_dict:
                #                 print('Assigning node {} to be the child of {}'.format(node_dict[node],node_dict[current_node]))
                parent_dict[node] = current_node
                queue.append(node)
            else:
                p = 'Initial start node' if parent_dict[node] == start_nodes_parent else str([parent_dict[node]])


#                 print('{} already has a parent {}'.format(node_dict[node], p))

# This will take a dictionary which represents the directed graph of components that need to be merged across chunk boundaries as input
# It will return the reduced version of this graph eliminating any redundancies
def consolidate_conn_components(test):
    pre_qc = set()
    all_nodes = {}
    for k, v in test.items():
        pre_qc.add(k)
        all_nodes[k] = set()
        for vv in v:
            pre_qc.add(vv)
            all_nodes[vv] = set()

    nodes_to_remove = set()
    for start_node in test.keys():
        rename_count = 0
        # print('')
        # print('Start Node {} assignment = {}'.format(start_node, all_nodes[start_node]))
        if all_nodes[start_node] == set():
            # print('Starting at {}'.format(start_node))
            queue = deque([start_node])
            start_value = start_node
            back_track_log = set()
            while len(queue) > 0:

                current_node = queue.popleft()

                if all_nodes[current_node] == set():
                    # print('assigning current node {} to {}'.format(current_node,start_value))
                    back_track_log.add(current_node)
                    all_nodes[current_node].add(start_value)
                    if current_node in test.keys():
                        # print('appending children to the queue')
                        [queue.append(x) for x in test[current_node]]
                else:
                    rename_count += 1
                    if rename_count < 2:
                        nodes_to_remove.add(start_node)
                        # print('Backtracking when i got to node {}'.format(current_node))
                        start_value = next(iter(all_nodes[current_node]))
                        # print('Updating the value to be {}'.format(start_value))
                        for node in back_track_log:
                            # print('Going back to update node {} to {}'.format(node,start_value))
                            all_nodes[node].clear()
                            all_nodes[node].add(start_value)

                    else:
                        # need to remove all nodes that have this already assigned value
                        # and updated them to the start_value assigned on line 36
                        value_to_remove = next(iter(all_nodes[current_node]))
                        # print('Found {} already labeled node at {}. Its label = {}'.format(rename_count,current_node,value_to_remove))
                        nodes_to_push_rename = [k for k, v in all_nodes.items() if value_to_remove in v]

                        for node in nodes_to_push_rename:
                            all_nodes[node].clear()
                            all_nodes[node].add(start_value)
                            if node in test.keys():  # cant remove it from the keys if its a leaf
                                nodes_to_remove.add(start_node)

        # else:
        #     print('was going to analyze {} but its already assigned to {}'.format(start_node,all_nodes[start_node]))

    my_dict = defaultdict(set)
    for k, v in all_nodes.items():
        if k != next(iter(v)):
            my_dict[next(iter(v))].add(k)

    post_qc = set()
    for k, v in my_dict.items():
        post_qc.add(k)
        for vv in v:
            post_qc.add(vv)

    for i in pre_qc:
        if i not in post_qc:
            # print('Node {} is in the input dict but not output'.format(i))
            it_worked = 0
        else:
            it_worked = 1

    return my_dict  # ,it_worked


# Processing Functions
def skeleton_to_swc_parallel(specimen_dir, skeleton_dir, input_swc, output_swc_path):
    max_stack_size = 7000000000
    post_processed_morph = morphology_from_swc(input_swc)
    try:
        original_morph_soma = post_processed_morph.get_soma()
        original_soma_coord = np.array([original_morph_soma['x'], original_morph_soma['y'], original_morph_soma['z']])
        centroid = (int(original_morph_soma['x']), int(original_morph_soma['y']), int(original_morph_soma['z']))
        no_soma = False
    except:
        centroid = tuple([0, 0, 0])
        original_soma_coord = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        no_soma = True

    # Stitch together skeleton images when necessary
    skeleton_dir_L = os.path.join(specimen_dir, 'Left_Skeleton')
    if os.path.exists(skeleton_dir) == False:
        os.mkdir(skeleton_dir)

    if os.path.exists(skeleton_dir_L) == True:
        # check if there is already a skeleton dir
        if os.path.exists(skeleton_dir):
            # make sure all the files are in skeleton dir
            if len([f for f in os.listdir(skeleton_dir) if '.tif' in f]) != len(
                    [f for f in os.listdir(skeleton_dir_L) if '.tif' in f]):
                stitch_skeleton_dir(specimen_dir, skeleton_dir)
        else:
            stitch_skeleton_dir(specimen_dir, skeleton_dir)

    # Load centroid file, check if left right division is present. if it is check for scenario where left centroid file is empty and right is not. need to add to the x value
    # of the centroid because the image was split. X Coordinates in the right image will need to be increased by half of the whole image x-size
    centroid_file = os.path.join(specimen_dir, 'Segmentation_soma_centroid.csv')

    left_segmentation_file = os.path.join(specimen_dir, 'Left_Segmentation_soma_centroid.csv')
    if os.path.exists(left_segmentation_file):
        temp_img = tif.imread(os.path.join(skeleton_dir, '001.tif'))
        left_df = pd.read_csv(os.path.join(specimen_dir, 'Left_Segmentation_soma_centroid.csv'))
        right_df = pd.read_csv(os.path.join(specimen_dir, 'Right_Segmentation_soma_centroid.csv'))
        if left_df.empty and not right_df.empty:
            right_df['# x'] = right_df['# x'] + int(temp_img.shape[1] / 2)
            right_df.to_csv(os.path.join(specimen_dir, 'Right_Segmentation_soma_centroid.csv'))
            centroid_file = os.path.join(specimen_dir, 'Right_Segmentation_soma_centroid.csv')
        elif not left_df.empty and right_df.empty:
            centroid_file = os.path.join(specimen_dir, 'Left_Segmentation_soma_centroid.csv')
        else:
            # theyre both empty or theyre both full, assume the latter
            right_df['# x'] = right_df['# x'] + int(temp_img.shape[1] / 2)
            left_df = left_df.append(right_df)
            pd.DataFrame(data=left_df.mean(axis=0)).T.to_csv(centroid_file)
    print('     Centroid File = {}'.format(centroid_file))

    # Make sure the soma coords file exists, stitch together if needed
    soma_coords_file = os.path.join(specimen_dir, 'Segmentation_ch1.csv')
    split_side_file = os.path.join(specimen_dir, 'Left_Segmentation_ch1.csv')

    if os.path.exists(split_side_file):
        print('     stitching together the ch1 L+R segmentation csvs')
        temp_img = tif.imread(os.path.join(skeleton_dir, '001.tif'))
        ch1_Left_df = pd.read_csv(os.path.join(specimen_dir, 'Left_Segmentation_ch1.csv'))
        ch1_Right_df = pd.read_csv(os.path.join(specimen_dir, 'Right_Segmentation_ch1.csv'))
        ch1_Right_df['# x'] = ch1_Right_df['# x'] + int(temp_img.shape[1] / 2)
        ch1_Left_df.append(ch1_Right_df).to_csv(soma_coords_file, index=False)

    # Should alway exists, should not be left and right
    skeleton_labels_file = os.path.join(specimen_dir, 'Segmentation_skeleton_labeled2.csv')  #### Olga, Change Here ####
    print('     Skeleton Dir = {}'.format(skeleton_dir))

    # Calculate how many files to load as not to exceed memory limit per iteration
    filelist = [f for f in os.listdir(skeleton_dir) if f.endswith('.tif')]
    filelist.sort()
    filename = os.path.join(skeleton_dir, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = len(filelist), img.shape[0], img.shape[1]
    cell_stack_memory = cell_stack_size[0] * cell_stack_size[1] * cell_stack_size[2]
    print('     cell_stack_size (z,y,x):', cell_stack_size, cell_stack_memory)
    # if cell stack memory>max_stack_size need to split
    num_parts = int(np.ceil(cell_stack_memory / max_stack_size))
    print('     num_parts:', num_parts)

    idx = np.append(np.arange(0, cell_stack_size[0], int(np.ceil(cell_stack_size[0] / num_parts))),
                    cell_stack_size[0] + 1)
    shared_slices = idx[1:-1]
    both_sides_of_slices = np.append(shared_slices, shared_slices - 1)

    # Initialize variables before begining chunk look
    connected_components_on_border = {}
    for j in both_sides_of_slices:
        connected_components_on_border[j] = []
    previous_cc_count = 0
    slice_count = 0
    full_neighbors_dict = {}
    node_component_label_dict = {}
    cc_dict = {}

    for i in range(num_parts):
        print('     At Part {}'.format(i))
        idx1 = idx[i]
        idx2 = idx[i + 1]
        filesublist = filelist[idx1:idx2]
        # print('part ', i, idx1, idx2, len(filesublist))

        # load stack and run connected components
        cv_stack = []
        for i, f in enumerate(filesublist):
            filename = os.path.join(skeleton_dir, f)
            img = tif.imread(filename)  # ,cv2.IMREAD_UNCHANGED)
            cv_stack.append(img)
        three_d_array = np.stack(cv_stack)

        s = ndi.generate_binary_structure(3, 3)
        labels_out, current_number_of_components = ndi.label(three_d_array, structure=s)
        # labels_out = cc3d.connected_components(three_d_array, connectivity=26) #a 3d matrix where pixel values = connected component labels
        # current_number_of_components = np.max(labels_out)

        # print('There are {} CCs in this stack of images'.format(current_number_of_components))

        # Create range for connected components across all chunks of image stack
        if previous_cc_count == 0:
            cc_range = range(1, current_number_of_components + 1)
        else:
            cc_range = range(previous_cc_count + 1, previous_cc_count + 1 + current_number_of_components)

        for cc in cc_range:
            cc_dict[cc] = {'X': [], 'Y': [], 'Z': []}

        # Load each image slice by slice so that we can get coordinates to the connected components
        for single_image in labels_out:
            single_image_unique_labels = np.unique(single_image)  # return indices and ignore 0
            for unique_label in single_image_unique_labels:
                if unique_label != 0:
                    indices = np.where(single_image == unique_label)
                    conn_comp_apriori_num = unique_label + previous_cc_count
                    [cc_dict[conn_comp_apriori_num]['Y'].append(coord) for coord in indices[0]]
                    [cc_dict[conn_comp_apriori_num]['X'].append(coord) for coord in indices[1]]
                    [cc_dict[conn_comp_apriori_num]['Z'].append(x) for x in [slice_count] * len(indices[1])]

                    if slice_count in both_sides_of_slices:
                        connected_components_on_border[slice_count].append(conn_comp_apriori_num)

            slice_count += 1

        ################################################################################################################
        # Iterate through this chunks connected components and update entire image stack neighbors dictionary
        ################################################################################################################

        for conn_comp in cc_range:
            # print('Analyzing Conn Component {}'.format(conn_comp))
            coord_values = cc_dict[conn_comp]
            component_coordinates = np.array([coord_values['X'], coord_values['Y'], coord_values['Z']]).T

            # Making a node dictionary for this con comp so we can lookup in the 26 node check step
            node_dict = {}
            count = 0
            for c in component_coordinates:
                count += 1
                node_dict[tuple(c)] = count
                node_component_label_dict[tuple(c)] = conn_comp

            # 26 nodes to check in defining neighbors dict
            movement_vectors = ([p for p in itertools.product([0, 1, -1], repeat=3)])
            neighbors_dict = {}
            for node in component_coordinates:

                node_neighbors = []
                for vect in movement_vectors:
                    node_to_check = tuple(list(map(add, tuple(node), vect)))
                    if node_to_check in node_dict.keys():
                        node_neighbors.append(node_to_check)

                # remove myself from my node neightbors list
                node_neighbors = set([x for x in node_neighbors if x != tuple(node)])
                neighbors_dict[tuple(node)] = node_neighbors
                full_neighbors_dict[conn_comp] = neighbors_dict

        previous_cc_count += current_number_of_components

    ################################################################################################################
    # All image chunks have been loaded and full neighbors dict is constructed. Merge CCs across the chunk indices
    # Find nodes on left and right of the slice whos Z index == slice edge
    ################################################################################################################
    print('     Merging Conn Components across chunk indexes')
    # Initializing Nodes on either side of slice boundary
    nodes_to_left_of_boundary = {}
    for x in shared_slices - 1:
        nodes_to_left_of_boundary[x] = defaultdict(list)

    nodes_to_right_of_boundary = {}
    for x in shared_slices:
        nodes_to_right_of_boundary[x] = defaultdict(list)

    # assigning nodes only with z value on edge to left or right side
    for key, val in connected_components_on_border.items():
        for con_comp_label in val:
            coord_values = full_neighbors_dict[con_comp_label].keys()
            for coord in coord_values:
                z = coord[-1]
                if z == key:
                    if z in shared_slices - 1:
                        nodes_to_left_of_boundary[key][con_comp_label].append(tuple(coord))
                    else:
                        nodes_to_right_of_boundary[key][(tuple(coord))] = con_comp_label
                        # Redundancy
            # coord_values = cc_dict[con_comp_label]
            # component_coordinates = np.array([coord_values['X'],coord_values['Y'],coord_values['Z']]).T
            # for coord in component_coordinates:
            #     z = coord[-1]
            #     if z == key:
            #         if z in shared_slices-1:
            #             nodes_to_left_of_boundary[key][con_comp_label].append(tuple(coord))
            #         else:
            #             nodes_to_right_of_boundary[key][(tuple(coord))] = con_comp_label

    ################################################################################################################
    # Check the 26 boxes surrounding each node that lives on the left side
    # Update full neighbors dictionary
    # Create dictionary of conn components that need to merge across slize index
    ################################################################################################################

    movement_vectors = ([p for p in itertools.product([0, 1, -1], repeat=3)])
    full_merge_dict = defaultdict(set)
    merging_ccs = defaultdict(set)

    for slice_locations in shared_slices:
        # print(slice_locations)
        left_side = slice_locations - 1
        right_side = slice_locations

        # Iterate through Left Conn Components that have nodes on the boundary
        # Find nodes on the other side and their corresponding CC label indicating a need to merge

        for cc_label in nodes_to_left_of_boundary[left_side].keys():
            # print(cc_label)

            cc_coords_to_check = nodes_to_left_of_boundary[left_side][cc_label]
            for left_node in cc_coords_to_check:
                for vect in movement_vectors:
                    node_to_check_on_other_side = tuple(list(map(add, tuple(left_node), vect)))
                    if node_to_check_on_other_side in nodes_to_right_of_boundary[right_side]:
                        right_cc = nodes_to_right_of_boundary[right_side][node_to_check_on_other_side]

                        # Update Neighbors Dictionary
                        # print('IM ADDING {} to {} Neighbor Dict'.format(node_to_check_on_other_side,left_node))
                        full_neighbors_dict[cc_label][left_node].add(node_to_check_on_other_side)
                        full_neighbors_dict[right_cc][node_to_check_on_other_side].add(left_node)

                        merging_ccs[cc_label].add(right_cc)
                        # print(merging_ccs)

    full_merge_dict = consolidate_conn_components(merging_ccs)

    # print('DID IT WORK = {}'.format(did_it_work))
    ################################################################################################################
    # Merge Connected Components Across Chunk Slices
    ################################################################################################################

    # merging these values in full neighbors dict
    for keeping_cc, merging_cc in full_merge_dict.items():
        for merge_cc in merging_cc:
            # pdate full neighbors dict
            full_neighbors_dict[keeping_cc].update(full_neighbors_dict[merge_cc])

            del full_neighbors_dict[merge_cc]

    ################################################################################################################
    # Loads soma segmentation coordinates. Compress to x-y plane, run connected components to determine number of somas. Picks soma closest to center of image
    ################################################################################################################
    ##MM 10-9-20
    # Just using soma cooridnate from the previous version of swc file rather than re-do this calc
    # Modified line 547 & 568 for distance calc, and centroid is defined at top of this function as well

    # df = pd.read_csv(soma_coords_file)
    # if df.empty:
    #     no_soma = True
    #     print('     No Soma = {}'.format(no_soma))
    #     centroid = tuple([0,0,0])
    #     soma_coords = np.asarray([[0,0,0],[1,1,1],[2,2,2]])
    #     my_tree = KDTree(soma_coords,leaf_size=2)
    # else:
    #     no_soma = False
    #     print('     No Soma = {}'.format(no_soma))
    #     set_of_unique_xy = set()
    #     list_of_soma_coords = []
    #     # print('df is {}'.format(len(df)))
    #     for i in df.index:
    #         row = df.iloc[i]
    #         x = row[0]
    #         y = row[1]
    #         z = row[2]
    #         y_x = (y,x)
    #         set_of_unique_xy.add(y_x)
    #         coord = tuple([x,y,z])
    #         list_of_soma_coords.append(coord)
    #     soma_coords = np.asarray(list_of_soma_coords)

    #     #create soma dict with x and y as keys and z range values
    #     soma_dict = {}
    #     # print('len of soma coords = {}'.format(len(soma_coords)))
    #     ccc = 0
    #     for x_loc in np.unique(soma_coords.T[0]):
    #         ccc+=1
    #         soma_dict[x_loc] = {}
    #         coords_at_this_x =  np.asarray([x for x in soma_coords if x[0] == x_loc])
    #         for y_loc in np.unique(coords_at_this_x.T[1]):
    #             coords_at_this_y = np.asarray([x for x in coords_at_this_x if x[1] == y_loc])
    #             max_z = coords_at_this_y[:,2].max()
    #             min_z = coords_at_this_y[:,2].min()
    #             if min_z != max_z:
    #                 soma_dict[x_loc][y_loc] = range(min_z,max_z)
    #             else:
    #                 soma_dict[x_loc][y_loc] = range(min_z,max_z+1)

    #     #get unique x,ys build numpy array then run conn components #this should be better than max projections
    #     soma_img = np.zeros(img.shape)  #from skeleton loading
    #     for i in set_of_unique_xy:
    #         soma_img[i] = 255
    #     ccs,num_ccs = ndi.label(soma_img)
    #     cc_count = 0
    #     list_of_cc_labels = []
    #     for i in np.unique(ccs):
    #         if i != 0:
    #             index = np.where(ccs == i)
    #             if len(index[0]) > 600:
    #                 cc_count+=1
    #                 list_of_cc_labels.append(i)
    #     print('     There are {} somas in {}'.format(cc_count,sp_id))

    #     if cc_count > 1:
    #         cc_coord_dict = defaultdict(set)
    #         for cc_ind in list_of_cc_labels:
    #             index = np.where(ccs == cc_ind)
    #             y_range = range(index[0].min(),index[0].max()+1)
    #             x_range = range(index[1].min(),index[1].max()+1)

    #             for x in x_range:
    #                 for y in soma_dict[x]:
    #                     my_depth_range = soma_dict[x][y]

    #                     if len(my_depth_range) > 1:
    #                         my_corrected_depth_range = range(my_depth_range[0],my_depth_range[-1]+2)
    #                         [cc_coord_dict[cc_ind].add(tuple([x,y,zz])) for zz in [z for z in my_corrected_depth_range]]
    #                     else:
    #                         my_corrected_depth_range = my_depth_range
    #                         [cc_coord_dict[cc_ind].add(tuple([x,y,zz])) for zz in [z for z in my_corrected_depth_range]]

    #         con_comp_centers_dict = {}
    #         cc_label_dict = {}
    #         for cc_label,set_of_coords in cc_coord_dict.items():
    #             center = np.array(list(set_of_coords)).mean(axis=0)
    #             con_comp_centers_dict[tuple(center)] = (((center[0] - img.shape[1]/2)**2) + ((center[1] - img.shape[0]/2)**2))**0.5
    #             cc_label_dict[tuple(center)] = cc_label

    #         min_dist = sorted(con_comp_centers_dict.values())[0]
    #         centroid = [k for k in con_comp_centers_dict.keys() if con_comp_centers_dict[k] == min_dist][0]
    #         soma_coords =  np.asarray(list(cc_coord_dict[cc_label_dict[centroid]]))

    #     else:
    #         # print('Loading the centroid file')
    #         centroid_df = pd.read_csv(centroid_file)
    #         centroid = (centroid_df['# x'].values[0],centroid_df['y'].values[0],centroid_df['z'].values[0])
    #         centroid = tuple(list(map(int,centroid)))

    #     my_tree = KDTree(soma_coords,leaf_size=2)

    ################################################################################################################
    # Find Starting Node for each conn component. Check if its within 50 pixels of soma. Assign parent accordingly
    ################################################################################################################
    # run connected components on entire dataset now
    parent_dict = {}
    parent_dict[centroid] = -1

    for conn_comp in full_neighbors_dict.keys():
        # print('at conn_comp {}'.format(conn_comp))
        neighbors_dict = full_neighbors_dict[conn_comp]
        if len(full_neighbors_dict[conn_comp]) > 2:
            leaf_nodes = [x for x in neighbors_dict.keys() if len(neighbors_dict[x]) == 1]

            # There is no leaf node to start, so we will make it by removing a connection from closest to soma.
            if leaf_nodes == []:
                # find node closest to soma
                dist_dict = {}
                for coord in full_neighbors_dict[conn_comp].keys():
                    dist_to_soma = euclid_dist(centroid, coord)
                    dist_dict[coord] = dist_to_soma
                start_node = min(dist_dict, key=dist_dict.get)
                while len(full_neighbors_dict[conn_comp][start_node]) > 1:  ###REMOVE CYCLE
                    # print('removing cycle')
                    removed = full_neighbors_dict[conn_comp][start_node].pop()
                    full_neighbors_dict[conn_comp][removed].discard(start_node)

                # Check how far it is from soma cloud
                dist = distance.euclidean(original_soma_coord, start_node)
                # dist,ind = my_tree.query(np.asarray(start_node).reshape(1,3), k=1)

                # print('Distance = {}'.format(dist))
                if dist < 50:
                    # print('assigning soma centroid as the start node')
                    start_parent = centroid
                else:
                    start_parent = 0

                assign_parent_child_relation(start_node, start_parent, parent_dict, neighbors_dict)

            # At least one leaf node exists
            else:
                dist_dict = {}
                for coord in leaf_nodes:
                    dist_to_soma = euclid_dist(centroid, coord)
                    dist_dict[coord] = dist_to_soma
                start_node = min(dist_dict, key=dist_dict.get)

                # Check how far it is from soma cloud
                dist = distance.euclidean(original_soma_coord, start_node)
                # dist,ind = my_tree.query(np.asarray(start_node).reshape(1,3), k=1)

                # print('Distance = {}'.format(dist))
                if dist < 50:
                    # print('assigning soma centroid as the start node')
                    start_parent = centroid
                else:
                    start_parent = 0

                assign_parent_child_relation(start_node, start_parent, parent_dict, neighbors_dict)

    # In case with fake centroid remove centroid from parent dict and centroid
    if no_soma == True:
        parent_dict.pop(centroid)
        for k, v in parent_dict.items():
            if v == centroid:
                parent_dict[k] = -1

                # number each node for sake of swc
    ct = 0
    big_node_dict = {}
    for j in parent_dict.keys():
        ct += 1
        big_node_dict[tuple(j)] = ct

    # Load node type labels
    skeleton_labeled = pd.read_csv(skeleton_labels_file)
    skeleton_coord_labels_dict = {}
    for n in skeleton_labeled.index:
        skeleton_coord_labels_dict[
            (skeleton_labeled.loc[n].values[0], skeleton_labeled.loc[n].values[1], skeleton_labeled.loc[n].values[2])] = \
        skeleton_labeled.loc[n].values[3]

        # Make swc list for swc file writing
    swc_list = []
    for k, v in parent_dict.items():
        # id,type,x,y,z,r,pid
        if v == 0:
            parent = -1
            node_type = skeleton_coord_labels_dict[k]
        elif v == -1:
            parent = -1
            node_type = 1
        else:
            parent = big_node_dict[v]
            node_type = skeleton_coord_labels_dict[k]
        swc_line = [big_node_dict[k]] + [node_type] + list(k) + [1.0] + [parent]

        swc_list.append(swc_line)

    with open(output_swc_path, 'w') as f:
        f.write('# id,type,x,y,z,r,pid')
        f.write('\n')
        for sublist in swc_list:
            for val in sublist:
                f.write(str(val))
                f.write(' ')
            f.write('\n')
