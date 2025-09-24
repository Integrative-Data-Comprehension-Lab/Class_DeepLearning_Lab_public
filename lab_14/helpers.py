import torch
from torch.utils.data import Dataset

from tqdm import tqdm


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


def train_one_epoch(model, device, dataloader, criterion, optimizer, epoch, clip = 1.0):
    model.train()

    total_loss = 0.0

    for src, tgt in tqdm(dataloader, desc=f'Training Epoch {epoch + 1}', total = len(dataloader), leave=False):
        src, tgt = src.to(device), tgt.to(device)

        output = model(src, tgt) # (batch_size, tgt_len, vocab_size)

        # exclude first token (<sos>) from loss, flatten
        output_dim = output.shape[-1]
        output_flat = output[:, 1:].reshape(-1, output_dim)
        tgt_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(output_flat, tgt_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_one_epoch(model, device, dataloader, criterion, epoch = 0):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Validation/Test', total = len(dataloader), leave=False):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_prob=0.0)

            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flat, tgt_flat)
            
            total_loss += loss.item()

    return total_loss / len(dataloader)