import pytest
import inspect
import time

import torch
import torch.nn as nn
from torchvision import transforms
import MLP
from MLP import CustomImageDataset, train_one_epoch, evaluate_one_epoch, train_model, MultiLayerPerceptron, num_params_W1, num_params_b1, num_params_W2, num_params_b2, num_params_W3, num_params_b3, config, BetterMLP, main_BetterMLP


pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 30

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")

def test_custom_dataset_score_1():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    custom_datset = CustomImageDataset(root_dir = 'resources/cat_dog_images', 
                                    metadata_filename = "metadata.csv",
                                    transform = transform)
    assert len(custom_datset) == 6, "CustomImageDataset __len__ gave wrong value" 

    test_vals = {
        0 : [torch.Size([3, 800, 1200]), 1156067.0, 1],
        1 : [torch.Size([3, 900, 1200]), 1468958.5, 0],
        2 : [torch.Size([3, 800, 1200]), 1054946.0, 1],
        3 : [torch.Size([3, 900, 1200]), 1658777.625, 0],
        4 : [torch.Size([3, 900, 1200]), 1459332.75, 1],
        5 : [torch.Size([3, 900, 742]), 893281.3125, 0],
    }
    for i in range(len(custom_datset)):
        X, y = custom_datset[i]
        test_shape, test_sum, test_label = test_vals[i]
        assert X.shape == test_shape, f"CustomImageDataset {i}-th image data is not correct"
        assert X.sum().item() == pytest.approx(test_sum, rel=1e-4), f"CustomImageDataset {i}-th image data is not correct"
        assert test_label == y, f"CustomImageDataset {i}-th label is not correct"


    custom_datset = CustomImageDataset(root_dir = 'resources/cat_dog_images', 
                                    metadata_filename = "metadata_imbalanced.csv",
                                    transform = transform)
    assert len(custom_datset) == 5, "CustomImageDataset __len__ gave wrong value" 

    test_vals = {
        0 : [torch.Size([3, 900, 1200]), 1468958.5, 0],
        1 : [torch.Size([3, 800, 1200]), 1054946.0, 1],
        2 : [torch.Size([3, 900, 1200]), 1658777.625, 0],
        3 : [torch.Size([3, 900, 1200]), 1459332.75, 1],
        4 : [torch.Size([3, 900, 742]), 893281.3125, 0],
    }
    for i in range(len(custom_datset)):
        X, y = custom_datset[i]
        test_shape, test_sum, test_label = test_vals[i]
        assert X.shape == test_shape, f"CustomImageDataset {i}-th image data is not correct"
        assert X.sum().item() == pytest.approx(test_sum, rel=1e-4), f"CustomImageDataset {i}-th image data is not correct"
        assert test_label == y, f"CustomImageDataset {i}-th label is not correct"


def test_mlp_score_1():
    model = MultiLayerPerceptron(in_dim = 32*32, hidden_dim = 256, out_dim = 10)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 330762 
    assert total_params == expected_params, f"Total number of MultiLayerPerceptron parameters are not correct."

    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(4, 1, 28, 28)  # Example input (batch_size, channels, height, width)

    model = MultiLayerPerceptron(in_dim = 28*28, hidden_dim = 512, out_dim = 10)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 669706  
    assert total_params == expected_params, f"Total number of MultiLayerPerceptron parameters are not correct."

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)

    assert output.shape == (4, 10), "MultiLayerPerceptron output shape does not match the expected shape."

    assert torch.sum(output).item() == pytest.approx(-22842.507812, rel=1e-3), "MultiLayerPerceptron forward pass gave different value"

    assert torch.isclose(output[1,5:10].detach(), torch.tensor([-517.9559936523438,916.0165405273438,-273.4241027832031,-3733.653076171875,-2873.445556640625]), rtol=1e-3).all(),"MultiLayerPerceptron Forward pass gave different value"

def test_mlp_params_score_1():
    assert num_params_W1 == 1*28*28*512, "num_params_W1 calculation is incorrect"
    assert num_params_b1 == 512, "num_params_b1 calculation is incorrect"
    assert num_params_W2 == 512*512, "num_params_W2 calculation is incorrect"
    assert num_params_b2 == 512, "num_params_b2 calculation is incorrect"
    assert num_params_W3 == 512*10, "num_params_W3 calculation is incorrect"
    assert num_params_b3 == 10, "num_params_b3 calculation is incorrect"

def test_wandb_score_1():
    source_code = inspect.getsource(train_one_epoch)
    code_lines = [line.strip() for line in source_code.split("\n")]
    assert any([code.startswith("wandb.log") for code in code_lines]), "wandb.log was not called in train_one_epoch()"

    source_code = inspect.getsource(evaluate_one_epoch)
    code_lines = [line.strip() for line in source_code.split("\n")]
    assert any([code.startswith("wandb.log") for code in code_lines]), "wandb.log was not called in evaluate_one_epoch()"

    source_code = inspect.getsource(train_model)   
    code_lines = [line.strip() for line in source_code.split("\n")]
    assert any([code.startswith("wandb.init") for code in code_lines]), "wandb.init was not called in train_model()"
    assert any([code.startswith("wandb.finish()") for code in code_lines]), "wandb.finish() was not called in train_model()"
    


def test_gitignore_score_2():
    # Path to the .gitignore file
    gitignore_path = "../.gitignore"
    
    wandb_patterns = ['wandb', '**/wandb', '*/wandb', 'wandb/', '**/wandb/', '*/wandb/']
    checkpoints_patterns = ['checkpoints', '**/checkpoints', '*/checkpoints', 'checkpoints/', '**/checkpoints/', '*/checkpoints/']


    # Initialize flags for checking if both entries exist at the start of a line
    wandb_found = False
    checkpoints_found = False
    
    # Open the .gitignore file and check line-by-line
    with open(gitignore_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()  # Remove any leading/trailing whitespaces
            if any(stripped_line == pattern for pattern in wandb_patterns):
                wandb_found = True
            if any(stripped_line == pattern for pattern in checkpoints_patterns):
                checkpoints_found = True
    
    # Assert that both 'wandb' and 'checkpoints' are found at the start of lines
    assert wandb_found, "'wandb' not found in .gitignore"
    assert checkpoints_found, "'checkpoints' not found in .gitignore"

def test_BetterMLP_score_4():
    original_config = {
        "num_workers": 4,
        "batch_size": 128,
        "learning_rate": 1e-2,
        "num_epochs": 20,
        "checkpoint_path": "submitted_checkpoints/checkpoint.pth",    # Path to save the most recent checkpoint
        "best_model_path": "submitted_checkpoints/best_model.pth",    # Path to save the best model checkpoint
        "wandb_project_name": "MNIST-experiments",
    }

    for key, value in original_config.items():
        assert config[key] == value, "You are NOT allowed to edit `config` or `main_BetterMLP()`" 

    model = BetterMLP(in_dim=1*28*28, out_dim=10)
    for module in model.modules():
        assert not isinstance(module, nn.Conv2d), "You are NOT allowed to use Convolutional Layer"

    source_code = inspect.getsource(BetterMLP)
    assert "torchvision.models" not in source_code, "You are NOT allowed to use torchvision.models"

    source_code = inspect.getsource(MLP)
    code_lines = [line.strip() for line in source_code.split("\n")]
    for forbidden in ["import torchvision.models", "from torchvision.models import", "from torchvision import models"]:
        assert not any([code.startswith(forbidden) for code in code_lines]), "You are NOT allowed to use torchvision.models"
    
    config["mode"] = "eval"
    test_accuracy = main_BetterMLP(config)
    assert test_accuracy > 95, f"You should achieve test accuracy > 95% in BetterMLP model"