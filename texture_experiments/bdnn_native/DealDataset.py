import torch
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def datasetpr(dataset_train):
    datasetpro = []
    labelspro = []
    for data,label in dataset_train:
        data1 = data
        label1 = label
        datasetpro.append(data1)
        labelspro.append(label1)
    return torch.Tensor(datasetpro),torch.Tensor(labelspro)

class DealDataset(Dataset):

    def __init__(self, X, Y):
        self.images = X
        self.labels = Y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        return img, label