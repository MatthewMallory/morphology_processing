import os
import platform
from neuron_morphology.swc_io import morphology_from_swc, morphology_to_swc, Morphology
from morphology_processing.processes.SomaInternodePrune import SomaInternodePrune


class Resample:
    def __init__(self, morphology, specimen_id, path_to_vaa3d, node_spacing_for_resampling,
                 image_shape_xyz, soma_segmentation, z_scale_factor, x_res, y_res, **kwargs):
        self.morphology = morphology
        self.path_to_vaa3d = path_to_vaa3d
        self.node_spacing_for_resampling = node_spacing_for_resampling
        self.specimen_id = specimen_id
        self.image_shape_xyz = image_shape_xyz
        self.soma_segmentation = soma_segmentation
        self.z_scale_factor = z_scale_factor
        self.x_res = x_res
        self.y_res = y_res
        self.process_name = "resample"

    def process(self):
        return downsample_morph(self.morphology, self.specimen_id, self.path_to_vaa3d, self.node_spacing_for_resampling,
                                self.image_shape_xyz, self.soma_segmentation, self.z_scale_factor, self.x_res,
                                self.y_res)


def downsample_morph(morph, specimen_id, path_to_vaa3d, node_spacing, image_shape_xyz, soma_segmentation,
                     z_scale_factor, x_res, y_res, **kwargs):
    """
    corrections call will fix the internode creation that downsampling creates
    """
    root_dir = os.getcwd()
    os.chdir(path_to_vaa3d)
    if platform.system() == 'Windows':
        print("Windows: to do")
    else:
        original_morph = morph.clone()
        input_path = os.path.abspath('input_{}.swc'.format(specimen_id))
        output_path = os.path.abspath('Output_{}.swc'.format(specimen_id))
        morphology_to_swc(morph, input_path)
        exe = 'start_vaa3d.sh'
        cmd = 'sh {} -x resample_swc -f resample_swc -i {} -o {} -p {}'.format(exe, input_path, output_path,
                                                                               node_spacing)
        cmd2 = 'export DISPLAY=:30;Xvfb :30 -auth /dev/null & sh {} -x resample_swc -f resample_swc -i {} -o {} -p {}'.format(
            exe, input_path, output_path, node_spacing)
        print("Going to execute command:\n")
        print(cmd)
        print('from  directory:\n')
        print(os.path.abspath(os.getcwd()))
        try:
            os.system(cmd2)
        except:
            try:
                os.system(cmd)
            except:
                os.chdir(root_dir)
                return None

        downsampled_morph = morphology_from_swc(output_path)
        os.remove(output_path)
        os.chdir(root_dir)

        results_dict = {}
        # Run internode pruning
        somaprune = SomaInternodePrune(downsampled_morph, image_shape_xyz, soma_segmentation, z_scale_factor, x_res,
                                       y_res)
        internode_pruned = somaprune.process()['morph']

        # fix_soma_children_types(original_morph, internode_pruned)

        results_dict['morph'] = internode_pruned
        return results_dict


def fix_soma_children_types(m1, m2):
    """
    m1 = morphology object with correct labels
    m2 = morphology object with incorrect labels

    TODO this may need more consideration. e.g. we have no idea what nodes in m2 look like. Just because we found a match
    node and type is not consistent, doesn't mean the fix is to switch the single node... perhaps this should be more
    detailed.
    """
    coords_attached_to_soma = {(n['x'], n['y'], n['z']): n['type'] for n in m1.get_children(m1.get_soma())}
    for coord, no_type in coords_attached_to_soma.items():
        coord_to_fix = [n for n in m2.nodes() if (n['x'] == coord[0]) and (n['y'] == coord[1]) and (n['z'] == coord[2])]
        if coord_to_fix == []:
            continue
        else:
            no_id = coord_to_fix[0]['id']
            m2.node_by_id(no_id)['type'] = no_type

    fix_morph = Morphology(m2.nodes(),
                           parent_id_cb = lambda x:x['parent'],
                           node_id_cb=lambda x:x['id'])
    return fix_morph
