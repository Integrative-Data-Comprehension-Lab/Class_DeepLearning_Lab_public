import pytest
import torch
import inspect
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from pytorch_basics import train_logistic_regression, LinearNet, train_loop, evaluation_loop, num_params_W, num_params_b

def test_train_lr_score_1():
    accuracy = train_logistic_regression(num_epochs = 100, learning_rate = 0.1)
    assert accuracy > 0.7, "Accuracy after `train_logistic_regression` should be greater than 0.9 when trained for 100 epochs with learning rate 0.1" 

def test_params_count_score_1():
    assert num_params_W == 1*28*28*10, "num_params_W calculation is incorrect"
    assert num_params_b == 1*10, "num_params_b calculation is incorrect"

def test_linear_score_2():
    model = LinearNet(in_dim = 28*28, out_dim = 10)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 7850  
    assert total_params == expected_params, f"Total number of MultiLayerPerceptron parameters are not correct."

    model = LinearNet(in_dim = 32*32, out_dim = 20)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 20500 
    assert total_params == expected_params, f"Total number of MultiLayerPerceptron parameters are not correct."


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
def test_train_loop_score_3(batch_size, input_size, output_size, learning_rate, num_batches):
    model = SampleModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    dataset = RandomDataset(input_size=input_size, output_size=output_size, num_samples=batch_size * num_batches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cpu')
    model.to(device)
    
    initial_params = [param.clone() for param in model.parameters()]
    
    avg_train_loss = train_loop(model, device, dataloader, loss_fn, optimizer)
    
    accumulated_loss = 0.0
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        accumulated_loss += loss.item()

    expected_avg_train_loss = accumulated_loss / num_batches

    for initial_param, updated_param in zip(initial_params, model.parameters()):
        assert not torch.equal(initial_param, updated_param), "Parameters should be updated after train_loop optimizer step"
   
    assert avg_train_loss == pytest.approx(expected_avg_train_loss, abs=1e-2), "The training loss should be correctly accumulated over all batches in train_loop()"

    avg_train_loss2 = train_loop(model, device, dataloader, loss_fn, optimizer)
    assert avg_train_loss2 < avg_train_loss, "Loss should be decrease for each epoch in train_loop()"

    source_code = inspect.getsource(train_loop)
    assert "optimizer.zero_grad()" in source_code, ".zero_grad() was not called in train_loop()"

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
    
    avg_test_loss, accuracy = evaluation_loop(model, device, dataloader, loss_fn)

    assert not model.training, "@evaluation_loop: Model should be in evaluation mode"

    assert avg_test_loss == pytest.approx(expected_avg_test_loss, abs=1e-2), "The evaluation loss should be correctly accumulated over all batches"

    assert accuracy == pytest.approx(expected_accuracy, abs=1e-2), "The evaluation accuracy should be correctly calculated and accumulated over all batches"

