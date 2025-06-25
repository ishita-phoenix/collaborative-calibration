import torch
import torch.nn as nn

'''Specifically for MCQ responses'''
class MaxSoftRandIndex(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, matrix: torch.Tensor) -> float:
        N, K = matrix.shape
        den = (N * (N - 1)) / 2
        total_same_prob = 0
        for i in range(N):
            for j in range(i, N):
                for c in range(K):
                    total_same_prob += matrix[i][c] * matrix[j][c]
        return total_same_prob / den
    
class SoftRandIndexLossMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.msri = MaxSoftRandIndex()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, alpha=0.1) -> float:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        msri_loss = self.msri(probs)
        loss = ce_loss - alpha * msri_loss
        return loss
    
'''General use'''
class GenSoftRandIndex(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, matrix: torch.Tensor, one_hot: torch.Tensor) -> float:
        N, K = matrix.shape
        den = (N * (N - 1)) / 2
        total_same_prob = 0
        for i in range(N):
            for j in range(i, N):
                for c in range(K):
                    S_U = matrix[i][c] * matrix[j][c]
                    S_V = one_hot[i][c] * one_hot[j][c]
                    total_same_prob += (S_U * S_V) + (1-S_U) * (1-S_V)
        return total_same_prob / den
    
class SoftRandIndexLossGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.gsri = GenSoftRandIndex()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, alpha=0.1) -> float:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        one_hot_targets = torch.nn.functional.one_hot(targets, logits.shape[1]).float()
        sri_loss = self.gsri(probs, one_hot_targets)
        loss = ce_loss - alpha * sri_loss
        return loss
        

'''testing MaxSoftRandIndex'''
# probs = torch.softmax(torch.rand(4,3), dim=1)
# # print(probs)
# criterion = MaxSoftRandIndex()
# print(criterion(probs))

'''testing SoftRandIndexLoss'''
# logits = torch.rand(4,3)
# targets = torch.tensor([0,0,0,0])
# print(f'logits {logits}')
# print(f'targets {targets}')

# criterion = SoftRandIndexLoss()
# print(criterion(logits, targets))

# logits = torch.rand(5,2)
# targets = torch.tensor([0,0,0,0,0])
# print(f'logits {logits}')
# print(f'targets {targets}')

# criterion = SoftRandIndexLoss()
# print(criterion(logits, targets))

logits_spread = torch.tensor([[10.0, 2.0, 0.0],
                       [2.0, 10.0, 0.0],
                       [0.0, 2.0, 10.0],
                       [10.0, 0.0, 2.0],
                       [10.0, 2.0, 2.0]], requires_grad=True)

logits_close = torch.tensor([[10.0, 2.0, 2.0],
                       [2.0, 2.0, 10.0],
                       [2.0, 2.0, 10.0],
                       [10.0, 2.0, 2.0],
                       [10.0, 2.0, 2.0]], requires_grad=True)

targets_spread = torch.tensor([0, 1, 2, 0, 0])
targets_close = torch.tensor([0, 2, 2, 0, 0])

# criterion = SoftRandIndexLossGen()
# print(criterion(logits, targets_idx))

criterion = SoftRandIndexLossMax()
print(f'spread: {criterion(logits_spread, targets_spread, alpha=1)+1}')
print(f'close: {criterion(logits_close, targets_close, alpha=1)+1}')



