import pytest
import inspect
import torch

from resnet import IdentityBlock, ConvBlock, ResNet50, get_model
from resnet import shape_after_stage1, shape_after_stage2, shape_after_stage3, shape_after_stage4, shape_after_stage5, shape_after_avgpool, shape_after_flatten, shape_after_fc

def test_IdentityBlock_score_2():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 256, 56, 56)  # Example input (batch_size, channels, height, width)
    
    block = IdentityBlock(256, 64)
    assert sum(p.numel() for p in block.parameters()) == 70400, "IdentityBlock parameter number does not match"
    
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, -0.01)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, -0.01)

        #for name, param in block.named_parameters():
        #    print(f"{name:<30} Param shape: {str(param.shape):<30} Weight : {torch.sum(param).item()}")

        output = block(input_tensor)
        #print(torch.sum(output).item())
        assert torch.sum(output).item() == pytest.approx(393712.40625, abs=1e-2), "IdentityBlock forward pass gave different value"
        

    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.constant_(m.weight, 0)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 0)
                torch.nn.init.constant_(m.bias, 0)

        output = block(input_tensor)
        assert torch.all(input_tensor == output), "IdentityBlock shortcut path seems to be wrong"

    assert output.shape == input_tensor.shape, "Output shape should be the same as input shape in IdentityBlock"


    block = IdentityBlock(128, 512)
    assert sum(p.numel() for p in block.parameters()) == 3479552, "IdentityBlock parameter number does not match"
    

def test_ConvBlock_score_2():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 256, 56, 56)  # Example input (batch_size, channels, height, width)
    
    block = ConvBlock(256, 128, 2)
    assert sum(p.numel() for p in block.parameters()) == 379392, "ConvBlock parameter number does not match"
    
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, -0.01)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, -0.01)

        #for name, param in block.named_parameters():
        #    print(f"{name:<30} Param shape: {str(param.shape):<30} Weight : {torch.sum(param).item()}")

        output = block(input_tensor)
        # print(torch.sum(output).item())
        assert torch.sum(output).item() == pytest.approx(434.92108154296875, abs=1e-2), "ConvBlock forward pass gave different value"
        

    block = ConvBlock(512, 128, 2)
    assert sum(p.numel() for p in block.parameters()) == 543232, "ConvBlock parameter number does not match"

    block = ConvBlock(256, 256, 2)
    assert sum(p.numel() for p in block.parameters()) == 1184768, "ConvBlock parameter number does not match"

    block = ConvBlock(256, 256, 1)
    assert sum(p.numel() for p in block.parameters()) == 1184768, "ConvBlock parameter number does not match"


def test_ResNet50_score_3():
    ## Source code check
    source_code = inspect.getsource(ResNet50)
    code_lines = [line.strip() for line in source_code.split("\n") if not line.strip().startswith("#")]
    for forbidden in ["resnet50("]:
        assert not any([forbidden in code for code in code_lines]), "You are NOT allowed to use torchvision.models.resnet50() in your implementation of ResNet50"
    for required in ["ConvBlock", "IdentityBlock", "AdaptiveAvgPool2d"]:
        assert any([required in code for code in code_lines]), f"You are required to use {required} in your implementation of ResNet50"

    ## Parameter count check
    model = ResNet50(10)
    assert sum(p.numel() for p in model.parameters()) == 23528522, "ResNet50 parameter number does not match"

    ## Forward pass check
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 3, 64, 64)  # Example input (batch_size, channels, height, width)
    
    model = ResNet50(1000)
    # from torchvision import models
    # model = models.resnet50()

    assert sum(p.numel() for p in model.parameters()) == 25557032, "ResNet50 parameter number does not match"

    with torch.no_grad():
        for name, param in model.named_parameters():
            # if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)

    assert output.shape == (1, 1000), "ResNet50 output shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert torch.sum(output).item() == pytest.approx(2003.714111328125, rel=1e-5), "ResNet50 forward pass gave different value"

    # for val in output[0,5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[0,5:10].detach(), torch.tensor([236.1914825439453,170.70172119140625,215.92333984375, 373.31634521484375, -199.10665893554688, ]), rtol=1e-5).all(),"ResNet50 Forward pass gave different value"


def test_resnet_shapes_score_1():
    assert shape_after_stage1 == (4, 64, 56, 56), "shape_after_stage1 is not correct"
    assert shape_after_stage2 == (4, 256, 56, 56), "shape_after_stage2 is not correct"
    assert shape_after_stage3 == (4, 512, 28, 28), "shape_after_stage3 is not correct"
    assert shape_after_stage4 == (4, 1024, 14, 14), "shape_after_stage4 is not correct"
    assert shape_after_stage5 == (4, 2048, 7, 7), "shape_after_stage5 is not correct"
    assert shape_after_avgpool == (4, 2048, 1, 1), "shape_after_avgpool is not correct"
    assert shape_after_flatten == (4, 2048), "shape_after_flatten is not correct"
    assert shape_after_fc == (4, 1000), "shape_after_fc is not correct"


def test_transfer_learning_score_2():
    # torch.manual_seed(0)  # For reproducibility
    # input_tensor = torch.rand(1, 3, 64, 64)  # Example input (batch_size, channels, height, width)

    config = {
        'model_architecture': 'resnet50',
        'pretrained' : 'IMAGENET1K_V2',
    }
    model = get_model(config, 100)
    for name, param in model.named_parameters():
        if name.startswith("fc.") or name.startswith("layer4."):
            assert param.requires_grad == True, "layer4 and fc layer should have requires_grad = True"
        else:
            assert param.requires_grad == False, "layers except for layer4 and fc should have requires_grad = False"

    assert model.fc.out_features == 100, "fc layer should have out_features same as num_classes"
