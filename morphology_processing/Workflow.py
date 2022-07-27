from collections import deque
import pandas as pd


class InvalidWorkflow(ValueError):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'The workflow format seems to be invalid '


def validate_workflow(workflow):
    root_ct = len([item for item in workflow.nodes if item['parent_id'] == -1])
    if root_ct != 1:
        raise InvalidWorkflow()

    node_ids = [n['id'] for n in workflow.nodes]
    orphaned_nodes = [n for n in workflow.nodes if (n['parent_id'] not in node_ids) and (n['parent_id'] != -1)]
    if orphaned_nodes:
        raise InvalidWorkflow()


def workflow_from_file(input_file):
    df = pd.read_csv(input_file, names=('id', 'process_name', 'parent_id'), comment='#', sep=" ", index_col=False)
    node_list = df.to_dict('record')

    workflow = Workflow(node_list)
    return workflow


class Workflow:

    def __init__(self, list_of_nodes):
        self.nodes = list_of_nodes
        validate_workflow(self)

    def __len__(self):
        return len(self.nodes)

    def get_root(self):
        return [n for n in self.nodes if n['parent_id'] == -1][0]

    def get_children(self, node):
        return [n for n in self.nodes if n['parent_id'] == node['id']]

    def dfs_traversal(self):

        queue = deque([self.get_root()])
        dfs_nodes = []
        while len(queue) > 0:
            curr_node = queue.popleft()
            dfs_nodes.append(curr_node)
            children = self.get_children(curr_node)
            for ch in children:
                queue.appendleft(ch)

        return dfs_nodes
