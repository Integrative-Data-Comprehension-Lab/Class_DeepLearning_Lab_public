import time, os, shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

def create_dataloaders(train_dataset, test_dataset, device, batch_size, num_worker):
    kwargs = {}
    if device.startswith("cuda"):
        kwargs.update({
            'pin_memory': True,
        })

    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_worker, **kwargs)
    
    return train_dataloader, test_dataloader

def train_one_epoch(model, device, dataloader, criterion, optimizer, epoch):
    """ train for one epoch """
    loss_meter = AverageMeter('Loss', '.4e')
    accuracy_meter = AverageMeter('Accuracy', '6.2f')
    data_time = AverageMeter('Data_Time', '6.3f') # Time for data loading
    batch_time = AverageMeter('Batch_Time', '6.3f') # time for mini-batch train
    metrics_list = [loss_meter, accuracy_meter, data_time, batch_time, ]
    
    model.train() # switch to train mode

    end = time.time()

    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch + 1}', total=len(dataloader))
    for images, target in progress_bar:
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        accuracy = compute_accuracy(output, target)
        loss_meter.update(loss.item(), images.shape[0])
        accuracy_meter.update(accuracy, images.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        progress_bar.set_postfix(avg_metrics = ", ".join([str(x) for x in metrics_list]))
        end = time.time()
    progress_bar.close()

    wandb.log({
        "epoch" : epoch,
        "Train Loss": loss_meter.avg,
        "Train Accuracy": accuracy_meter.avg
    })

def evaluate_one_epoch(model, device, dataloader, criterion, epoch = 0, use_wandb = True):
    loss_meter = AverageMeter('Loss', '.4e')
    accuracy_meter = AverageMeter('Accuracy', '6.2f')
    metrics_list = [loss_meter, accuracy_meter]

    model.eval() # switch to evaluate mode

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation/Test', total=len(dataloader))
        for images, target in progress_bar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            accuracy = compute_accuracy(output, target)
            loss_meter.update(loss.item(), images.shape[0])
            accuracy_meter.update(accuracy, images.shape[0])

            progress_bar.set_postfix(avg_metrics = ", ".join([str(x) for x in metrics_list]))
        progress_bar.close()

    if use_wandb:
        wandb.log({
            "epoch" : epoch,
            "Test Loss": loss_meter.avg, 
            "Test Accuracy": accuracy_meter.avg
        })

    return accuracy_meter.avg

class AverageMeter(object):
    """Tracks and updates the running average of a metric."""
    def __init__(self, metric_name , format_spec = '.4f'):
        self.metric_name = metric_name 
        self.format_spec = format_spec
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.metric_name}: {format(self.avg, self.format_spec)} (n={self.count})"
    

def compute_accuracy(output, target):
    """
    Computes the top-1 classification accuracy .

    Args:
        output (torch.Tensor): Model outputs (logits or probabilities), shape (batch_size, num_classes)
        target (torch.Tensor): Ground truth labels, shape (batch_size,)

    Returns:
        float: Accuracy as a percentage.
    """
    with torch.no_grad():
        pred = output.argmax(dim=1)
        accuracy = (pred == target).float().mean().item() * 100.0
    return accuracy   



def train_model(model, train_dataset, test_dataset, config):
    device = config["device"]

    wandb.init(
        project = config["wandb_project_name"],
        name = config["wandb_experiment_name"],
        config = config
    )

    train_dataloader, test_dataloader = create_dataloaders(train_dataset, test_dataset, device, 
                                                           batch_size = config["batch_size"], 
                                                           num_worker = config["num_workers"])

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = config["learning_rate"])

    start_epoch = 0
    best_accuracy = 0

    if config["resume_training"]:
        load_checkpoint_path = config["best_model_path"] if config["resume_training"] == "best" else config["checkpoint_path"]
        start_epoch, best_accuracy = load_checkpoint(load_checkpoint_path, model, optimizer, device)

    for epoch in range(start_epoch, config["num_epochs"]):
        train_one_epoch(model, device, train_dataloader, criterion, optimizer, epoch)
        test_accuracy = evaluate_one_epoch(model, device, test_dataloader, criterion, epoch)

        ## save checkpoint
        if (epoch + 1) % config["checkpoint_save_interval"] == 0 or (epoch + 1) == config["num_epochs"]: 
            is_best = test_accuracy > best_accuracy
            best_accuracy = max(test_accuracy, best_accuracy)
            save_checkpoint(config["checkpoint_path"], model, optimizer, epoch, best_accuracy, is_best, config["best_model_path"])

    wandb.finish()

    return best_accuracy


def save_checkpoint(filepath, model, optimizer, epoch, best_accuracy, is_best, best_model_path):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'best_accuracy': best_accuracy,
    }
    
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_model_path)


def load_checkpoint(filepath, model, optimizer, device):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        print(f"=> loaded checkpoint '{filepath}' (epoch {start_epoch})")
        return start_epoch, best_accuracy
    else:
        print(f"=> no checkpoint found at '{filepath}'")
        return 0, 0
    

def load_and_evaluate_model(model, test_dataset, config):
    """ Load model checkpoint from config["best_model_path"] and evaulate the model """

    device = config["device"]

    test_dataloader = DataLoader(dataset = test_dataset, batch_size=config["batch_size"], 
                                  shuffle=False, num_workers=config["num_workers"])
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = config["learning_rate"])  # Dummy optimizer to satisfy checkpoint loader
    _, _ = load_checkpoint(config["best_model_path"], model, optimizer, device)

    test_accuracy = evaluate_one_epoch(model, device, test_dataloader, criterion, use_wandb = False)
    print(f"Test-set Accuracy: {test_accuracy:.2f}%")

    return test_accuracy