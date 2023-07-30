from torch.utils.data import Dataset
import torch
from metrics.metrics import kappa


class PMAESDataSet(Dataset):
    def __init__(self, prompt_id, essay, linguistic, readability, score):
        super(PMAESDataSet, self).__init__()
        self.prompt_id = prompt_id
        self.essay = essay
        self.linguistic = linguistic
        self.readability = readability
        self.score = score

    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):
        return {
            'prompt': self.prompt_id[item],
            'pos_ids': torch.tensor(self.essay[item], dtype=torch.long),
            'ling': torch.tensor(self.linguistic[item], dtype=torch.float),
            'read': torch.tensor(self.readability[item], dtype=torch.float),
            'score': torch.tensor(self.score[item], dtype=torch.float),
        }
