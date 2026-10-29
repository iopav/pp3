import torch

def map_labels_to_zero_based(labels):
    label_map = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    # Assuming labels is a tensor of integers
    labels_mapped = torch.tensor([label_map[label.item()] for label in labels])
    return labels_mapped

 # Y_train is now a PyTorch tensor
