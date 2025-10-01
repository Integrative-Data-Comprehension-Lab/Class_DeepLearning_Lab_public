import pytest
import time
import torch

from ViT import MultiHeadSelfAttention, FeedForwardNetwork, TransformerEncoderLayer, ImagePatchifier, PatchEmbedding, ViT

pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 20

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")

def test_MHA_score_3():
    embed_dim = 1024
    num_heads = 8
    dropout_prob = 0.1
    seq_length = 20
    batch_size = 4

    model = MultiHeadSelfAttention(
        embed_dim = embed_dim,
        num_heads = num_heads,
        dropout_prob=dropout_prob,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 4198400, "MultiHeadSelfAttention model parameter number does not match the expected value."

    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_length, embed_dim)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param) * 0.02)

    model.train()
    torch.manual_seed(0)
    output = model(X)

    assert output.shape == (batch_size, seq_length, embed_dim), "MultiHeadSelfAttention output shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert output.sum().item() == pytest.approx(117.18157958984375, rel=1e-5), "MultiHeadSelfAttention forward pass gave different value"

    test_val2 = output[1, 7, 5:10].detach()
    # for val in test_val2:
    #     print(val.item())
    assert torch.isclose(test_val2, torch.tensor([0.36215710639953613, -0.020174294710159302, -0.9210979342460632, -0.43544435501098633, 0.3126012682914734]), rtol=1e-4).all(),"MultiHeadSelfAttention forward pass gave different value"

def test_FFN_score_1():
    embed_dim = 768
    feedforward_dim = 3072
    seq_length = 20
    batch_size = 4

    model = FeedForwardNetwork(
        hidden_dim = embed_dim,
        feedforward_dim = feedforward_dim,
        dropout_prob = 0.2,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 4722432, "FeedForwardNetwork model parameter number does not match the expected value."

    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_length, embed_dim)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(X)

    assert output.shape == (batch_size, seq_length, embed_dim), "FeedForwardNetwork output shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert output.sum().item() == pytest.approx(-105700.421875, rel=1e-5), "FeedForwardNetwork forward pass gave different value"

    test_val2 = output[1, 7, 5:10].detach()
    # for val in test_val2:
    #     print(val.item())
    assert torch.isclose(test_val2, torch.tensor([539.5791015625, -530.8429565429688, -662.0255126953125, -184.68130493164062, 248.78472900390625]), rtol=1e-3).all(),"FeedForwardNetwork forward pass gave different value"


def test_EncoderLayer_score_1():
    embed_dim = 768
    num_heads = 12
    feedforward_dim = 3072
    batch_size = 4
    seq_length = 20

    model = TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout_prob=0.2)
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 7087872, "TransformerEncoderLayer model parameter number does not match the expected value."

    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_length, embed_dim)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(X)

    assert output.shape == (batch_size, seq_length, embed_dim), "TransformerEncoderLayer output shape does not match the expected shape."

    test_val1 = output.sum().item()
    # print(test_val1)
    assert test_val1 == pytest.approx(1963359.75, rel=1e-3), "TransformerEncoderLayer Forward pass gave different value"

    test_val2 = output[1, 7, 5:10].detach()
    # for val in test_val2:
    #     print(val.item())
    assert torch.isclose(test_val2, torch.tensor([2014.8843994140625,-4121.5595703125,0.34315329790115356,-149.4501953125,463.7095947265625]), rtol=1e-3).all(),"TransformerEncoderLayer Forward pass gave different value"


def test_ImagePatchifier_score_2():
    image_size = 224
    patch_size = 16
    batch_size = 4
    num_patches = (image_size // patch_size) ** 2

    model = ImagePatchifier(image_size, patch_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 0, "ImagePatchifier model parameter number does not match the expected value."

    torch.manual_seed(42)
    X = torch.randn(batch_size, 3, image_size, image_size)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(X)

    assert output.shape == (batch_size, num_patches, 3 * patch_size * patch_size), "ImagePatchifier output shape does not match the expected shape."

    test_val1 = output.sum().item()
    # print(test_val1)
    assert test_val1 == pytest.approx(-1461.169921875, rel=1e-3), "ImagePatchifier Forward pass gave different value"

    test_val2 = output[1, 7, 5:10].detach()
    # for val in test_val2:
    #     print(val.item())
    assert torch.isclose(test_val2, torch.tensor([-0.853420078754425,-2.430283546447754,0.754170298576355,-1.944800853729248,-2.127908229827881]), rtol=1e-3).all(),"ImagePatchifier Forward pass gave different value"



def test_PatchEmbedding_score_1():
    image_size = 224
    patch_size = 16
    embed_dim = 1024
    batch_size = 4
    num_patches = (image_size // patch_size) ** 2

    model = PatchEmbedding(num_patches, patch_size, embed_dim)
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 990208, "PatchEmbedding model parameter number does not match the expected value."

    torch.manual_seed(42)
    X = torch.randn(batch_size, num_patches, 3 * patch_size * patch_size)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(X)

    assert output.shape == (batch_size, num_patches + 1, embed_dim), "PatchEmbedding output shape does not match the expected shape."

    test_val1 = output.sum().item()
    # print(test_val1)
    assert test_val1 == pytest.approx(35877.65625, rel=1e-3), "PatchEmbedding Forward pass gave different value"

    test_val2 = output[1, 7, 5:10].detach()
    # for val in test_val2:
    #     print(val.item())
    assert torch.isclose(test_val2, torch.tensor([10.399658203125,-47.199134826660156,-30.492382049560547,6.912761688232422,48.571163177490234]), rtol=1e-3).all(),"PatchEmbedding Forward pass gave different value"




def test_ViT_score_2():
    image_size = 224
    patch_size = 16
    num_channels = 3
    num_classes = 1000
    embed_dim = 768
    num_transformer_layers = 12
    num_heads = 12
    feedforward_dim = 3072
    dropout_prob = 0.1
    batch_size = 4

    model = ViT(
        image_size=image_size,
        num_channels=num_channels,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim = embed_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads = num_heads,
        feedforward_dim=feedforward_dim,
        dropout_prob=dropout_prob,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 86567656, "ViT model parameter number does not match the expected value."

    torch.manual_seed(42)
    X = torch.randn(batch_size, num_channels, image_size, image_size)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(X)

    assert output.shape == (batch_size, num_classes), "ViT output shape does not match the expected shape."

    test_val1 = output.sum().item()
    # print(test_val1)
    assert test_val1 == pytest.approx(-8405.7529296875, rel=1e-4), "ViT Forward pass gave different value"

    test_val2 = output[1,5:10].detach()
    # for val in test_val2:
    #     print(val.item())
    assert torch.isclose(test_val2, torch.tensor([-45.429603576660156,-50.44127655029297,-80.5923080444336,58.09738540649414,-7.729438304901123]), rtol=1e-4).all(),"ViT Forward pass gave different value"

