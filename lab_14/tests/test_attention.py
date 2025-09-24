import pytest
import time

import torch


from attention import Encoder, DotProductAttention, Decoder


pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 10

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")


def test_Encoder_score_1():
    model = Encoder(vocab_size = 100, embed_dim = 32, hidden_dim = 64,
                    num_layers = 3, pad_token_index = 10, dropout = 0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 71936
    assert total_params == expected_params, f"Total number of Encoder model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    X = torch.randint(0, 100, (8, 20))  # (batch_size, src_len)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output, hidden = model(X)

    assert hidden.shape == (3, 8, 64), "Encoder hidden state shape does not match the expected shape."
    assert output.shape == (8, 20, 64), "Encoder output shape does not match the expected shape."

    # print (torch.sum(hidden).item())
    assert torch.sum(hidden).item() == pytest.approx(-18.977497100830078, rel=1e-3), "Encoder forward pass gave different value"

    # print (torch.sum(output).item())
    assert torch.sum(output).item() == pytest.approx(-269.17333984375, rel=1e-3), "Encoder forward pass gave different value"

    # for val in hidden[1, 2, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(hidden[1,2, 5:10].detach(), torch.tensor([-0.9953196048736572, -0.9999762773513794, -0.9256951808929443, 0.9431674480438232, -0.8098224401473999]), rtol=1e-3).all(),"Encoder Forward pass gave different value"

    # for val in output[1, 2, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[1,2, 5:10].detach(), torch.tensor([-0.9986253976821899, -0.9980080127716064, 0.10349994897842407, 2.1457672119140625e-05, 0.9999898076057434]), rtol=1e-3).all(),"Encoder Forward pass gave different value"


def test_Attention_score_4():
    model = DotProductAttention(hidden_dim = 64)

    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 4096  
    assert total_params == expected_params, f"Total number of DotProductAttention model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    decoder_hidden = torch.randn(2, 8, 64)      # (num_layers, batch_size, hidden_dim)
    encoder_outputs = torch.randn(8, 20, 64)    # (batch_size, src_len, hidden_dim)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    context_vector, attention_weights = model(decoder_hidden, encoder_outputs)

    assert context_vector.shape == (8, 64), "DotProductAttention context_vector shape does not match the expected shape."
    assert attention_weights.shape == (8, 20), "DotProductAttention attention_weights shape does not match the expected shape."

    # print (torch.sum(context_vector).item())
    assert torch.sum(context_vector).item() == pytest.approx(-7.69600248336792, rel=1e-3), "DotProductAttention forward pass gave different value"

    # print (torch.sum(attention_weights).item())
    assert torch.sum(attention_weights).item() == pytest.approx(8.0, rel=1e-3), "DotProductAttention forward pass gave different value"

    # for val in context_vector[3, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(context_vector[3, 5:10].detach(), torch.tensor([0.6293327212333679, -0.15233969688415527, 1.1718792915344238, 0.1618402600288391, -0.3478044867515564]), rtol=1e-3).all(), "DotProductAttention Forward pass gave different value"

    # for val in attention_weights[1, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(attention_weights[1, 5:10].detach(), torch.tensor([8.590915712147762e-08, 1.0470998290657008e-08, 1.8273162497028927e-11, 8.878785207055984e-11, 0.003723435802385211]), rtol=1e-3).all(), "DotProductAttention Forward pass gave different value"



def test_Decoder_score_5():
    model = Decoder(vocab_size = 100, embed_dim = 32, hidden_dim = 64,
                    num_layers = 3, pad_token_index = 10, dropout = 0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 88932  
    assert total_params == expected_params, f"Total number of Decoder model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    input_token = torch.randint(0, 100, (8,)) # (batch_size,)
    prev_hidden = torch.randn(3, 8, 64)       # (num_layers, batch_size, hidden_dim)
    encoder_outputs = torch.randn(8, 20, 64)  # (batch_size, src_len, hidden_dim)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    logits, decoder_hidden, attention_weights = model(input_token, prev_hidden, encoder_outputs)

    assert logits.shape == (8, 100), "Decoder logits shape does not match the expected shape."
    assert decoder_hidden.shape == (3, 8, 64), "Decoder hidden state shape does not match the expected shape."
    assert attention_weights.shape == (8, 20), "Decoder attention_weights shape does not match the expected shape."

    # print (torch.sum(logits).item())
    assert torch.sum(logits).item() == pytest.approx(143.30007934570312, rel=1e-3), "Decoder forward pass gave different value"

    # print (torch.sum(decoder_hidden).item())
    assert torch.sum(decoder_hidden).item() == pytest.approx(65.08338928222656, rel=1e-3), "Decoder forward pass gave different value"

    # print (torch.sum(attention_weights).item())
    assert torch.sum(attention_weights).item() == pytest.approx(8.0, rel=1e-3), "Decoder forward pass gave different value"

    # for val in logits[3, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(logits[3, 5:10].detach(), torch.tensor([0.9068845510482788, -10.340373992919922, -8.49719524383545, 3.1468570232391357, -10.984673500061035]), rtol=1e-3).all(), "Decoder Forward pass gave different value"

    # for val in decoder_hidden[1, 2, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(decoder_hidden[1,2, 5:10].detach(), torch.tensor([-0.45227134227752686, 0.27627313137054443, 0.5697533488273621, 1.0, -0.2447052001953125]), rtol=1e-3).all(), "Decoder Forward pass gave different value"

    # for val in attention_weights[1, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(attention_weights[1, 5:10].detach(), torch.tensor([8.313961075145926e-09, 7.490005373256281e-06, 0.22994424402713776, 6.502139798802986e-11, 2.3892519074308893e-09]), rtol=1e-3).all(), "Decoder Forward pass gave different value"


def test_timeout_score_10():
    assert True