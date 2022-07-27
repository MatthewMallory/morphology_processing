from morph_utils.modifications import sort_morph_ids


class SortTreeIDs:
    def __init__(self, morphology,**kwargs):
        self.morphology = morphology
        self.process_name = "sort_node_ids"

    def process(self):
        return {'morph': sort_morph_ids(self.morphology)}
