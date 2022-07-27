
import argschema as ags
from morphology_processing.morphology_processing import processes_swc_file
from morphology_processing.Workflow import workflow_from_file
import os
from morphology_processing.get_workflow_parameters import get_input_dict, load_statics
import json


class IO_Schema(ags.ArgSchema):
    workflow_file = ags.fields.InputFile(description='input workflow txt file')
    root_outdir = ags.fields.OutputDir(description='root output directory')
    specimen_paths_json = ags.fields.InputFile(description="Json file with format: {specimen_id_1: {"
                                                           "'input_swc':'..path/spid_1.swc', "
                                                           "'specimen_autotrace_directory':'../path/'}}")
    user_defined_parameters_json = ags.fields.InputFile(
        description='inputs that are not intrinsic to autotrace pipeline e.g. pruning threshold',
        allow_none=True, default=None)



def main(workflow_file, specimen_paths_json, root_outdir, user_defined_parameters_json, **kwargs):
    with open(specimen_paths_json, "r") as f:
        specimen_path_info = json.load(f)

    if user_defined_parameters_json:
        with open(user_defined_parameters_json, "r") as f2:
            user_parameters = json.load(f2)

    else:
        user_parameters = load_statics()['default_parameters']

    work_flow = workflow_from_file(workflow_file)
    print("There are {} steps in the workflow".format(len(work_flow)))
    for specimen_id, specimen_dict in specimen_path_info.items():

        swc_src = specimen_dict['input_swc']
        specimen_autotrace_dir = specimen_dict['specimen_autotrace_directory']

        specimen_outdir = os.path.join(root_outdir, specimen_id)

        parameters_dict = get_input_dict(specimen_id=specimen_id, spdir=specimen_autotrace_dir, raw_morph_path=swc_src)
        parameters_dict["specimen_dir"] = specimen_autotrace_dir

        for k, v in user_parameters.items():
            parameters_dict[k] = v

        processes_swc_file(swc_src, parameters_dict, work_flow, specimen_outdir)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(**module.args)
