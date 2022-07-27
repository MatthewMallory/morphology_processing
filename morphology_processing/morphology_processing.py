import os
from neuron_morphology.swc_io import morphology_from_swc, morphology_to_swc
from collections import deque
from morphology_processing.processes import (AddRadius, ConnectSegments, EdgeArtifact,
                                             ExtraCellRemover, Prune, ValidateCellOfInterest,
                                             ApicalClassification, RemoveRedundancy, Resample,
                                             SegmentRelabelling, SomaConnections,
                                             SomaInternodePrune, SortTreeIDs, )
from morph_utils.visuals import basic_morph_plot
import matplotlib.pyplot as plt
from morphology_processing.Workflow import workflow_from_file

FUNC_NAMES = {

    "AddRadius":AddRadius.AddRadius,
    "ConnectSegments":ConnectSegments.ConnectSegments,
    "ApicalClassification": ApicalClassification.ApicalClassification,
    "EdgeArtifact": EdgeArtifact.EdgeArtifact,
    "ExtraCellRemover": ExtraCellRemover.ExtraCellRemover,
    "Prune": Prune.PruneTree,
    "RemoveRedundancy": RemoveRedundancy.RemoveRedundancy,
    "Resample": Resample.Resample,
    "SegmentRelabelling": SegmentRelabelling.SegmentRelabelling,
    "SomaConnections": SomaConnections.SomaConnections,
    "SomaInternodePrune": SomaInternodePrune.SomaInternodePrune,
    "SortTreeIDs": SortTreeIDs.SortTreeIDs,
    "ValidateCellOfInterest": ValidateCellOfInterest.ValidateCellOfInterest,
}


def processes_swc_file(swc_path, input_dict, workflow, outdir, save_intermediate=True, visualize_cells=True):
    """
    Worker function that will take swc file and run it through the workflow. Where workflow is
    a Workflow.workflow class instance.
    :param swc_path: path to input swc for processing
    :param input_dict: input dictionary with all parameters needed for processing. see morphology_processing.get_workflow_parameters for clarificiations
    :param workflow: a directed acyclical graph workflow.
    :param outdir: output directory
    :param save_intermediate: if you want to save swc files that are at non-terminal nodes of workflow
    :param visualize_cells: will create visuals of each swc file along workflow
    :return: None
    """
    if type(workflow) == str:
        workflow = workflow_from_file(workflow)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    initial_morph = morphology_from_swc(swc_path)
    processing_steps = workflow.dfs_traversal()

    morph_queue = deque([initial_morph])
    processed_morphologys = []
    for process_node in processing_steps:

        curr_morph = morph_queue.popleft()

        if visualize_cells:
            # clone for before and after viz
            preproc_morph = curr_morph.clone()

        process_name = process_node['process_name']
        process = FUNC_NAMES[process_name]
        process_id = process_node['id']

        input_dict['morphology'] = curr_morph
        this_process = process(**input_dict)
        print("Running Process:")
        print(this_process.process_name)
        result_dict = this_process.process()
        curr_morph = result_dict['morph']
        processed_morphologys.append(curr_morph)

        filename = f"Step{process_id}_{process_name}.swc"
        outfile = os.path.join(outdir, filename)

        if visualize_cells:
            fig_ofile = outfile.replace(".swc", ".png")
            fig, axe = plt.subplots(1, 2)
            basic_morph_plot(preproc_morph, axe[0], "before", scatter=False)
            basic_morph_plot(curr_morph, axe[1], "after", scatter=False)
            plt.suptitle(process_name)

            for ax in axe:
                ax.set_aspect('equal')

            fig.savefig(fig_ofile, dpi=300, bbox_inches='tight')
            plt.clf()

        if save_intermediate:
            morphology_to_swc(curr_morph, outfile)

        # add morphs to morph_queue in dfs manor as well
        children_processes = workflow.get_children(process_node)
        for _ in children_processes:
            morph_queue.appendleft(curr_morph)

        if not children_processes:
            outfile = outfile.replace(".swc","_terminal_process.swc")
            morphology_to_swc(curr_morph, outfile)
