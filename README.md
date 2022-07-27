# morphology_processing
A repository for creating and executing morphology processing workflows

This repository was designed to post-process single cell morphologies that have been generated with the code found here:
https://github.com/ogliko/patchseq-autorecon 
Although it can work with cells generated outside the patchseq-autorecon workflow, that is not currently supported in this version.

In this repository you can create directed acylical graph structured processing workflows for swc files. The default post processing
workflow, which was used in Gliko et al 2022, can be found here. The generalized format for a workflow text file is as follows:
# process_id, process_name, process_parent_id
1, MyFirstProcess, -1
2, MySecondProcess, 1
3, MyThirdProcess, 2
4, MyFourthProcess, 2

Where each process_name should be one of the keys found in FUNC_NAMES (link here). 

New processes can be added using the following steps:
1. creating a file in morphology_processing/processes/MyNewProcess.py file 
2. Adding this to imports in morphology_processing.py
3. Adding this to FUNC_NAMES dictionary in morphology_processing.py
