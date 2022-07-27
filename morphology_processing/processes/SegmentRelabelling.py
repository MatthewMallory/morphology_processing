from collections import deque


class SegmentRelabelling:

    def __init__(self, morphology, dendrite_min_fraction_for_relabel, **kwargs):
        self.morphology = morphology
        self.dendrite_min_fraction_for_relabel = dendrite_min_fraction_for_relabel
        self.process_name = "segment_relabel"

    def process(self):
        return segment_relabeling(self.morphology, self.dendrite_min_fraction_for_relabel)


def segment_relabeling(morph, dendrite_min_fraction_for_relabel, **kwargs):
    """
    """
    soma_nodes = [n for n in morph.nodes() if n['type'] == 1]
    results_dict = {}
    if soma_nodes != []:
        soma = soma_nodes[0]
        for child in morph.get_children(morph.node_by_id(soma['id'])):
            start_node = morph.node_by_id(child['id'])
            label = get_majority_of_segment(start_node, morph)
            relabel_segment(start_node, label, morph)

    for root_node in morph.get_roots():
        if root_node['type'] != 1:
            label = get_majority_of_segment(root_node, morph, dendrite_min_fraction_for_relabel)
            relabel_segment(root_node, label, morph)

    results_dict['morph'] = morph
    return results_dict


def get_majority_of_segment(start_node, morph, dendrite_min_fraction_for_relabel=0.5):
    count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    queue = deque([start_node['id']])
    while len(queue) > 0:
        current_node = queue.popleft()
        try:
            my_type = morph.node_by_id(current_node)['type']
        except:
            my_type = 5
        count_dict[my_type] += 1
        for n in morph.get_children(morph.node_by_id(current_node)):
            queue.append(n['id'])
    # dendrite favored threshold
    if count_dict[3] / (count_dict[3] + count_dict[2]) > dendrite_min_fraction_for_relabel:
        return 3
    else:
        component_label = [k for k, v in count_dict.items() if v == max(count_dict.values())][0]
        return component_label


#
# def segment_relabeling(morph,**kwargs):
#     """
#     soma_and_disconn_components: set true to relabel disconnected segments as well as
#                                  soma stems
#     """
#     soma_nodes = [n for n in morph.nodes() if n['type'] ==1]
#     results_dict = {}
#     if soma_nodes != []:
#         soma = soma_nodes[0]
#         for child in morph.get_children(morph.node_by_id(soma['id'])):
#             start_node = morph.node_by_id(child['id'])
#             label = get_majority_of_segment(start_node,morph)
#             relabel_segment(start_node,label,morph)
#         for root_node in morph.get_roots():
#             if root_node['type'] != 1:
#                 label = get_majority_of_segment(root_node,morph)
#                 relabel_segment(root_node,label,morph)
#         results_dict['morph'] = morph
#         return results_dict
#
#
# def get_majority_of_segment(start_node,morph):
#     count_dict = {1:0,2:0,3:0,4:0,5:0}
#     queue = deque([start_node['id']])
#     while len(queue) > 0:
#         current_node = queue.popleft()
#         my_type = morph.node_by_id(current_node)['type']
#         count_dict[my_type]+=1
#         for n in morph.get_children(morph.node_by_id(current_node)):
#             queue.append(n['id'])
#
#     component_label = [k for k,v in count_dict.items() if v == max(count_dict.values())][0]
#     return component_label

def relabel_segment(start_node, label, morph):
    queue = deque([start_node['id']])
    while len(queue) > 0:
        current_node = queue.popleft()
        morph.node_by_id(current_node)['type'] = label
        for n in morph.get_children(morph.node_by_id(current_node)):
            queue.append(n['id'])
