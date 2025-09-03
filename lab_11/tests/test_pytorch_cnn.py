import pytest
import inspect
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_cnn import SimpleCNN, train_one_epoch, evaluate_one_epoch
from pytorch_cnn import num_params_conv1, num_params_pool1, num_params_conv2, num_params_pool2, num_params_fc1, num_params_fc2


pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 10

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")


def test_SimpleCNN_score_3():
    model = SimpleCNN(out_dim = 10)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 56442  
    assert total_params == expected_params, f"Total number of SimpleCNN model parameters should be {expected_params}, but got {total_params}."

    model = SimpleCNN(out_dim = 20)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 57732 
    assert total_params == expected_params, f"Total number of SimpleCNN model parameters should be {expected_params}, but got {total_params} (when out_dim = 20)."


    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(4, 3, 32, 32)  # Example input (batch_size, channels, height, width)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)

    assert output.shape == (4, 20), "SimpleCNN output shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert torch.sum(output).item() == pytest.approx(135044.375, rel=1e-5), "SimpleCNN forward pass gave different value"

    # for val in output[1,5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[1,5:10].detach(), torch.tensor([15731.5693359375, -3431.8212890625, 22715.591796875, 10027.4677734375, -1623.3018798828125]), rtol=1e-5).all(),"SimpleCNN Forward pass gave different value"


def test_params_count_score_2():
    assert num_params_conv1 == (5*5*3 + 1) * 8, "num_params_conv1 calculation is incorrect"
    assert num_params_pool1 == 0, "num_params_pool1 calculation is incorrect"
    assert num_params_conv2 == (5*5*8 + 1) * 16, "num_params_conv2 calculation is incorrect"
    assert num_params_pool2 == 0, "num_params_pool2 calculation is incorrect"
    assert num_params_fc1 == 400*128 + 128, "num_params_fc1 calculation is incorrect"
    assert num_params_fc2 == 128*10 + 10, "num_params_fc2 calculation is incorrect"


# Sample model for testing
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x
    
class RandomDataset(Dataset):
    def __init__(self, input_size, output_size, num_samples):
        self.input_size = input_size
        self.output_size = output_size
        self.num_samples = num_samples
        self.X = torch.randn(num_samples, input_size)
        self.y = torch.randint(0, output_size, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("input_size", [20])
@pytest.mark.parametrize("output_size", [10])
@pytest.mark.parametrize("learning_rate", [0.01])
@pytest.mark.parametrize("num_batches", [5])
def test_train_loop_score_2(batch_size, input_size, output_size, learning_rate, num_batches):
    model = SampleModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    dataset = RandomDataset(input_size=input_size, output_size=output_size, num_samples=batch_size * num_batches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cpu')
    model.to(device)
    
    initial_params = [param.clone() for param in model.parameters()]
    
    avg_train_loss = train_one_epoch(model, device, dataloader, loss_fn, optimizer, epoch = 0)
    
    accumulated_loss = 0.0
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        accumulated_loss += loss.item()

    expected_avg_train_loss = accumulated_loss / num_batches

    for initial_param, updated_param in zip(initial_params, model.parameters()):
        assert not torch.equal(initial_param, updated_param), "Parameters should be updated after train_one_epoch optimizer step"
   
    assert avg_train_loss == pytest.approx(expected_avg_train_loss, abs=1e-2), "The training loss should be correctly accumulated over all batches in train_one_epoch()"

    avg_train_loss2 = train_one_epoch(model, device, dataloader, loss_fn, optimizer, epoch = 0)
    assert avg_train_loss2 < avg_train_loss, "Loss should be decrease for each epoch in train_one_epoch()"

    source_code = inspect.getsource(train_one_epoch)
    assert "optimizer.zero_grad()" in source_code, ".zero_grad() was not called in train_one_epoch()"

@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("input_size", [20])
@pytest.mark.parametrize("output_size", [10])
@pytest.mark.parametrize("num_batches", [5])
def test_eval_loop_score_3(batch_size, input_size, output_size, num_batches):
    model = SampleModel()
    loss_fn = nn.CrossEntropyLoss()
    
    dataset = RandomDataset(input_size=input_size, output_size=output_size, num_samples=batch_size * num_batches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cpu')
    model.to(device)
    

    accumulated_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            accumulated_loss += loss.item()

            y_pred = logits.argmax(dim=1)
            correct += (y_pred == y).sum().item()

    expected_avg_test_loss = accumulated_loss / len(dataloader)
    expected_accuracy = correct / (batch_size * num_batches)
    
    avg_test_loss, accuracy = evaluate_one_epoch(model, device, dataloader, loss_fn)

    assert not model.training, "@evaluate_one_epoch: Model should be in evaluation mode"

    assert avg_test_loss == pytest.approx(expected_avg_test_loss, abs=1e-2), "The evaluation loss should be correctly accumulated over all batches"

    assert accuracy == pytest.approx(expected_accuracy, abs=1e-2), "The evaluation accuracy should be correctly calculated and accumulated over all batches"


def test_timeout_score_10():
    assert True