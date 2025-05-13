import pytest
import inspect
import time

import torch
import torch.nn as nn
from torchvision import transforms

import CNN
from CNN import custom_conv2d, SimpleCNN, BetterCNN, config, main_BetterCNN
from CNN import num_params_conv1, num_params_pool1, num_params_conv2, num_params_pool2, num_params_fc1, num_params_fc2, num_params_fc3
from train_utils import load_checkpoint

pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 30

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")


def test_custom_cnn_score_2():
    conv_layer = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5, bias=True)

    torch.manual_seed(0)
    x = torch.randn(6, 32, 64)
    filter_weights = torch.randn(6, 5, 5)
    filter_bias = torch.randn(1)
    with torch.no_grad():
        conv_layer.weight[0] = filter_weights
        conv_layer.bias[0] = filter_bias


    expected_output = conv_layer(x.unsqueeze(0)).squeeze()
    output = custom_conv2d(x, filter_weights, filter_bias)

    assert output.shape == expected_output.shape, "custom_conv2d output shape is not correct"
    assert torch.allclose(expected_output, output.float(), rtol = 1e-3), "custom_conv2d output value is not correct"

    source_code = inspect.getsource(custom_conv2d)
    code_lines = [line.strip() for line in source_code.split("\n")]
    for_loop_count = sum(1 for line in code_lines if 'for ' in line and ' in ' in line and not line.startswith("#"))
    assert for_loop_count <= 2, "you are allowed to use upto 2 for loops in custom_conv2d implementation"

def test_SimpleCNN_score_2():
    model = SimpleCNN(out_dim = 20)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 64132  
    assert total_params == expected_params, f"Total number of SimpleCNN model parameters should be {expected_params}, but got {total_params}."

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
    assert torch.sum(output).item() == pytest.approx(124011.0234375, rel=1e-5), "SimpleCNN forward pass gave different value"

    # for val in output[1,5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[1,5:10].detach(), torch.tensor([34684.69921875,19193.263671875,72752.765625,-71311.53125, 57295.65625]), rtol=1e-5).all(),"SimpleCNN Forward pass gave different value"


def test_cnn_params_score_1():
    assert num_params_conv1 == (7*7*3 + 1) * 8, "num_params_conv1 calculation is incorrect"
    assert num_params_pool1 == 0, "num_params_pool1 calculation is incorrect"
    assert num_params_conv2 == (4*4*8 + 1) * 16, "num_params_conv2 calculation is incorrect"
    assert num_params_pool2 == 0, "num_params_pool2 calculation is incorrect"
    assert num_params_fc1 == 400*128 + 128, "num_params_fc1 calculation is incorrect"
    assert num_params_fc2 == 128*64 + 64, "num_params_fc2 calculation is incorrect"
    assert num_params_fc3 == 64*10 + 10, "num_params_fc3 calculation is incorrect"
    


def test_BetterCNN_score_5():
    ### Check for the configs
    original_config = {
        "num_workers": 4,
        "batch_size": 128,
        # "learning_rate": 1e-2,
        # "num_epochs": 50,
        "checkpoint_path": "submitted_checkpoints/checkpoint.pth",    # Path to save the most recent checkpoint
        "best_model_path": "submitted_checkpoints/best_model.pth",    # Path to save the best model checkpoint
        "wandb_project_name": "CIFAR10-experiments",
    }

    for key, value in original_config.items():
        assert config[key] == value, "You are NOT allowed to edit `config` or `main_BetterCNN()` except for 'learning_rate' and 'num_epochs'" 

    ### Check for the source code
    source_code = inspect.getsource(BetterCNN)
    assert "torchvision.models" not in source_code, "You are NOT allowed to use torchvision.models"

    source_code = inspect.getsource(CNN)
    code_lines = [line.strip() for line in source_code.split("\n")]
    for forbidden in ["import torchvision.models", "from torchvision.models import", "from torchvision import models"]:
        assert not any([code.startswith(forbidden) for code in code_lines]), "You are NOT allowed to use torchvision.models"

    ### Check for the modules
    ALLOWED_MODULES = {
        nn.Conv2d,
        nn.MaxPool2d,
        nn.ReLU,
        nn.Linear,
        nn.Flatten,
        nn.Sequential,
    }

    model = BetterCNN(out_dim=10)
    for name, module in model.named_modules():
        # print(f"name: '{name}'", type(module))
        if name == "":  # skip the root module
            continue
        assert type(module) in ALLOWED_MODULES, f"Disallowed nn.Module used: {type(module).__name__} in '{name}'"
    
    ### Check for the performance
    config["mode"] = "eval"
    test_accuracy = main_BetterCNN(config)
    assert test_accuracy > 68, f"You should achieve test accuracy > 68% in BetterCNN model"


    ### Check for the epochs
    # device = config["device"]
    # model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr = config["learning_rate"])  # Dummy optimizer to satisfy checkpoint loader
    # epoch, best_accuracy = load_checkpoint(config["best_model_path"], model, optimizer, device)
    # print("epoch", epoch, "best_accuracy", best_accuracy)
    # assert epoch <= 50, "You are NOT allowed to train the model for more than 50 epochs"

def test_timeout_score_10():
    assert True