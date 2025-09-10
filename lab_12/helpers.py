import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

import torch


def train_one_epoch(model, device, dataloader, criterion, optimizer, epoch):
    """ train for one epoch """
    model.train()
    total_loss = 0.0

    dataloader_tqdm = tqdm(dataloader, desc=f'Training Epoch {epoch + 1}', total=len(dataloader))

    for X, y in dataloader_tqdm:
        X, y = X.to(device), y.to(device)
        
        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        dataloader_tqdm.set_postfix({"loss": f"{loss.item():.4e}"})
    
    dataloader_tqdm.close()

    return total_loss / len(dataloader)


def evaluate_one_epoch(model, device, dataloader, criterion, epoch = 0):
    """ Evaluate the model for one epoch"""
    model.eval()

    total_loss, total_correct = 0.0, 0.0

    with torch.no_grad():
        dataloader_tqdm = tqdm(dataloader, desc='Validation/Test', total=len(dataloader))
        for X, y in dataloader_tqdm:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            y_pred = logits.argmax(dim = 1) # shape: (batch_size,)

            total_loss += loss.item()
            total_correct += (y_pred == y).type(torch.float).sum().item()

            dataloader_tqdm.set_postfix({"loss": f"{loss.item():.4e}"})
        dataloader_tqdm.close()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    print(f"Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}\n")

    return accuracy


def compute_confusion_matrix(model, dataloader, n_labels, device):
    confusion = torch.zeros(n_labels, n_labels)
    model.eval()
    with torch.no_grad():
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)
            logits = model(input)
            predicted_index = torch.argmax(logits, dim=1).item()
            true_index = target.item()
            confusion[true_index][predicted_index] += 1
            
    # Normalize rows
    for i in range(n_labels):
        if confusion[i].sum() > 0:
            confusion[i] /= confusion[i].sum()
    return confusion


def draw_confusion_matrix(model, dataloader, class_names, device):
    confusion = compute_confusion_matrix(model, dataloader, len(class_names), device)
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    ax.set_xticklabels([''] + class_names, rotation=90)
    ax.set_yticklabels([''] + class_names)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title("Confusion Matrix")
    plt.show()