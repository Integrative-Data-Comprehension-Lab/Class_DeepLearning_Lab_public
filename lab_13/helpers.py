import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


def evaluate_one_epoch(model, device, dataloader, criterion, epoch = 0):
    """ Evaluate the model for one epoch"""
    model.eval()

    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    
    with torch.no_grad():
        for X, y, lengths in dataloader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)

            logits = model(X, lengths)
            loss = criterion(logits, y)

            y_pred = logits.argmax(dim = 1) # shape: (batch_size,)
            
            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (y_pred == y).type(torch.float).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


class TranslationDataset(Dataset):
    def __init__(self, path_tsv):
        self.pairs = []
        with open(path_tsv, encoding="utf-8") as f:
            for line in f:
                tgt, src = line.strip().split("\t")
                self.pairs.append((src, tgt))
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        return src, tgt


def plot_token_frequency_histogram(counter):
    """ Plots a histogram of token frequencies. """

    token_freqs = list(counter.values())
    plt.figure(figsize=(8, 4))
    plt.hist(token_freqs, bins=50, log=True, edgecolor='black')
    plt.title("Histogram of Token Frequencies (log-scaled)")
    plt.xlabel("Token Frequency (occurrences per token)")
    plt.ylabel("Number of Unique Tokens (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_token_count_histogram(token_counts):
    plt.figure(figsize=(8, 4))
    plt.hist(token_counts, bins=50, color='skyblue', edgecolor='black')
    plt.title("Histogram of Token Counts per Text")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()