from collections import defaultdict
import torch


class QueryBuffer():
    def __init__(self):
        self.memory = dict()

    def __call__(self, proposals):
        for i, id_ in enumerate(proposals.gt_ids):
            id = int(id_)
            if id < 0:
                continue
            if id in self.memory:
                self.memory[id]['embeddings'] = torch.cat([self.memory[id]['embeddings'], proposals.output_embedding[i].unsqueeze(0)], dim=0)
                self.memory[id]['pos'] = torch.cat([self.memory[id]['pos'], proposals.ref_pts[i].unsqueeze(0)], dim=0)
            else:
                self.memory[id] = dict()
                self.memory[id]['embeddings'] = proposals.output_embedding[i].unsqueeze(0)
                self.memory[id]['pos'] = proposals.ref_pts[i].unsqueeze(0)

    def clear(self):
        self.memory.clear()