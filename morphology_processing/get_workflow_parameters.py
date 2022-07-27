from neuron_morphology.swc_io import morphology_from_swc
import pandas as pd
import tifffile as tif
import os
import pkg_resources
import _pickle as  cPickle
import json
from .database_queries import query_for_z_resolution


def load_statics():
    """
    Load static template files

    logistic_regression_model_extra_cell_remover.pkl: logistic regression classifier trained to classifier extraneous cells vs. cell of interes
    Apical_Best_Classifier.pkl: Random Forest Classifier trained to classify apical dendrite
    avg_layer_depths.json: Average depth of cortical layers from mouse VISp patch seq dataset
    default_parameters.json: default hyperparameters for workflow processing

    :return:
    """
    depths_file = pkg_resources.resource_filename(__name__, 'template_files/avg_layer_depths.json')
    extra_cell_clf_pth = pkg_resources.resource_filename(__name__,'template_files/logistic_regression_model_extra_cell_remover.pkl')
    apical_clf_pth = pkg_resources.resource_filename(__name__,'template_files/Apical_Best_Classifier.pkl')
    default_user_parameters = pkg_resources.resource_filename(__name__, 'template_files/default_parameters.json')

    with open(apical_clf_pth, 'rb') as ap_clf_p:
        apical_clf = cPickle.load(ap_clf_p)

    with open(extra_cell_clf_pth, 'rb') as fn:
        extra_cell_clf = cPickle.load(fn)

    with open(depths_file, 'r') as jf:
        average_layer_depths = json.load(jf)

    with open(default_user_parameters, 'r') as jp:
        default_params = json.load(jp)

    return {"apical_classifier": apical_clf,
            "extra_cell_classifier": extra_cell_clf,
            "average_mouse_layer_depths": average_layer_depths,
            'default_parameters': default_params}


def get_input_dict(specimen_id, spdir, raw_morph_path):
    """
    This function assumes that a cell was run through the autotrace pipeline and therfore has all the auxillary files produced
    from that such as the maximum intensity projection of the soma channel (CH1) segmentation. See below link for example
    https://github.com/ogliko/patchseq-autorecon/blob/master/pipeline/example_pipeline.sh

    Some parameters are only required for certain steps of workflow processing, but others may be used in multiple workflow processes. For example
    the path to vaa3d (vaa3d_path) is only needed to run node resampling. Eventually this will be replaced with a
    resampling function in morph_utils.

    :param specimen_id: int, specimen identifier
    :param spdir: the root autotrace directory that contains all auxillary files.
    :param raw_morph_path:
    :return:
    """
    print("Loading Input Dict: {}".format(specimen_id))
    ch1_seg_path = os.path.join(spdir, 'Segmentation_ch1.csv')
    ch2_seg_path = os.path.join(spdir, 'Segmentation_ch2.csv')
    ch3_seg_path = os.path.join(spdir, 'Segmentation_ch3.csv')
    raw_morph = morphology_from_swc(raw_morph_path)
    seg = pd.read_csv(ch1_seg_path)

    if os.path.exists(os.path.join(spdir, 'MAX_Left_Segmentation_ch1.tif')):
        left_img = tif.imread(os.path.join(spdir, 'MAX_Left_Segmentation_ch1.tif'))
        right_img = tif.imread(os.path.join(spdir, 'MAX_Right_Segmentation_ch1.tif'))
        shape = (left_img.shape[1] + right_img.shape[1],
                 left_img.shape[0],
                 max([n['z'] for n in raw_morph.nodes()]))
        if not os.path.exists(ch2_seg_path):
            combine_seg_L_R(specimen_dir=spdir, channel=2, output_file=ch2_seg_path)
        if not os.path.exists(ch3_seg_path):
            combine_seg_L_R(specimen_dir=spdir, channel=3, output_file=ch3_seg_path)
    else:
        img = tif.imread(os.path.join(spdir, 'MAX_Segmentation_ch1.tif'))
        shape = (img.shape[1],
                 img.shape[0],
                 int(max([n['z'] for n in raw_morph.nodes()])))

    ch2_seg = pd.read_csv(ch2_seg_path)
    ch3_seg = pd.read_csv(ch3_seg_path)
    non_specific_segmentation = ch2_seg[['# x', 'y', 'z']].append(ch3_seg[['# x', 'y', 'z']]).values

    vaa3d_path = '//allen/programs/celltypes/workgroups/mousecelltypes/Matt_Mallory/Va3d_Linux_Older/Vaa3D_CentOS_64bit_v3.100/'
    static_inputs = load_statics()

    zscale = query_for_z_resolution(specimen_id)

    ap_clf = static_inputs['apical_classifier']

    input_dict = {'specimen_id': specimen_id,
                  'initial_morph': raw_morph,
                  'soma_segmentation': seg,
                  'apical_classifier': ap_clf,
                  'non_specific_segmentation': non_specific_segmentation,
                  'extraneous_cell_classifier': static_inputs['extra_cell_classifier'],
                  'image_shape_xyz': shape,
                  'path_to_vaa3d': vaa3d_path,
                  'z_scale_factor': zscale,
                  'avg_layer_depths': static_inputs['average_mouse_layer_depths'],
                  'specimen_dir': spdir,
                  'axon_segmentation': ch2_seg[['# x', 'y', 'z']]
                  }

    print("Loading Complete: {}".format(specimen_id))

    return input_dict




def combine_seg_L_R(specimen_dir, channel, output_file):
    temp_img = tif.imread(os.path.join(specimen_dir, 'MAX_Left_Segmentation_ch{}.tif'.format(channel)))
    left_csv = os.path.join(specimen_dir, 'Left_Segmentation_ch{}.csv'.format(channel))
    right_csv = os.path.join(specimen_dir, 'Right_Segmentation_ch{}.csv'.format(channel))
    if (os.path.exists(left_csv)) and (os.path.exists(right_csv)):
        left_df = pd.read_csv(left_csv)
        right_df = pd.read_csv(right_csv)
        right_df['# x'] = right_df['# x'] + int(temp_img.shape[1])
        left_df = left_df.append(right_df)
        left_df.to_csv(output_file)

