import pytest
import inspect
import time

import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image

import regularization
from regularization import CustomRandomCrop, CrossEntropyWithL2Loss, SimpleCNN, BetterCNN, config, main_BetterCNN
from regularization import num_params_batchnorm1, num_params_batchnorm2, num_params_batchnorm3
# from train_utils import load_checkpoint

pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 30

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")

def test_randomcrop_score_1():
    image_pil = Image.open("resources/dog-01.jpg")
    transform = transforms.Compose([
        transforms.ToTensor(),
        CustomRandomCrop(size = (224, 224))
    ])

    torch.manual_seed(0)
    image_cropped = transform(image_pil)

    assert torch.allclose(image_cropped.sum(dim = [1, 2]), torch.tensor([19181.27734375, 16130.005859375, 17805.90234375]), rtol = 1e-3), "CustomRandomCrop implementation is not correct"


def test_SimpleCNN_score_1():
    model = SimpleCNN(out_dim = 20)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 57460  
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
    assert torch.sum(output).item() == pytest.approx(253.9154052734375, rel=1e-5), "SimpleCNN forward pass gave different value"

    # for val in output[1,5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[1,5:10].detach(), torch.tensor([-3.1761434078216553,12.25412368774414,-1.3688417673110962,2.6418774127960205, -14.079788208007812]), rtol=1e-5).all(),"SimpleCNN Forward pass gave different value"

def test_L2reg_score_2():
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )

    criterion = CrossEntropyWithL2Loss(weight_decay = 1e-4)

    inputs = torch.randn(16, 10)            # (batch_size=16, input_dim=10)
    targets = torch.randint(0, 5, (16,))    # 5 classes

    logits = model(inputs)                  # (16, 5)
    loss = criterion(logits, targets, model)

    assert torch.allclose(loss, torch.tensor(1.6047970056533813), rtol=1e-6), "CrossEntropyWithL2Loss loss value is not correct."

def test_batchnorm_params_score_1():
    assert num_params_batchnorm1 == 8 * 2, "num_params_batchnorm1 calculation is incorrect"
    assert num_params_batchnorm2 == 16 * 2, "num_params_batchnorm2 calculation is incorrect"
    assert num_params_batchnorm3 == 128 * 2, "num_params_batchnorm3 calculation is incorrect"
    


def test_BetterCNN_score_5():
    ### Check for the configs
    original_config = {
        "num_workers": 4,
        "batch_size": 128,
    }

    for key, value in original_config.items():
        assert config[key] == value, "You are NOT allowed to edit `num_workers` or `batch_size`" 

    ### Check for the source code
    source_code = inspect.getsource(BetterCNN)
    assert "torchvision.models" not in source_code, "You are NOT allowed to use torchvision.models"

    source_code = inspect.getsource(regularization)
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
        nn.Dropout,
        nn.BatchNorm2d,
        nn.BatchNorm1d,
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
    assert test_accuracy > 83, f"You should achieve test accuracy > 70% in BetterCNN model"


def test_timeout_score_10():
    assert True