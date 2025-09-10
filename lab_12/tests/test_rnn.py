import pytest
import inspect
import time
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rnn import one_hot_encode_string, NamesDataset, CustomRNN, ShakespeareDataset, CharRNN
from rnn import num_params_customRNN


pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 10

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")


def test_onehot_score_1():
    VOCAB_SIZE = 3
    CHAR_TO_INDEX = {'a':0, 'b':1, 'c':2}
    one_hot_encode_string.__globals__["VOCAB_SIZE"] = VOCAB_SIZE
    one_hot_encode_string.__globals__["CHAR_TO_INDEX"] = CHAR_TO_INDEX

    text = "aabacabaaa"
    encoded = one_hot_encode_string(text)
    assert isinstance(encoded, torch.Tensor), "One-hot encoded output should be a torch Tensor"

    assert encoded.shape == (len(text), VOCAB_SIZE), f"Expected shape {(len(text), VOCAB_SIZE)}, got {encoded.shape}"

    assert torch.all(torch.sum(encoded, dim = 1).int() == 1), f"One-hot encoded output is incorrect"
    assert torch.equal(encoded[2], torch.tensor([0,1,0])), f"One-hot encoded output is incorrect"
    assert torch.equal(encoded[3], torch.tensor([1,0,0])), f"One-hot encoded output is incorrect"
    assert torch.equal(encoded[4], torch.tensor([0,0,1])), f"One-hot encoded output is incorrect"

def test_namesdataset_score_1():
    VOCAB = list(string.ascii_letters)
    VOCAB_SIZE = len(VOCAB)
    INDEX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
    CHAR_TO_INDEX = {c: i for i, c in enumerate(VOCAB)}
    one_hot_encode_string.__globals__["VOCAB_SIZE"] = VOCAB_SIZE
    one_hot_encode_string.__globals__["CHAR_TO_INDEX"] = CHAR_TO_INDEX

    names_dataset = NamesDataset(csv_path='/datasets/NLP/names.csv')
    name_tensor, label_tensor = names_dataset[0]

    assert isinstance(name_tensor, torch.Tensor), "name_tensor should be a torch Tensor"
    assert isinstance(label_tensor, torch.Tensor), "name_tensor should be a torch Tensor"

    assert torch.all(names_dataset[1][0].sum(dim = 0) == torch.Tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])), "name_tensor value is incorrect"
    
    assert names_dataset[0][1].item() == 1, "label_tensor value is incorrect"
    assert names_dataset[1000][1].item() == 7, "label_tensor value is incorrect"
    assert names_dataset[2000][1].item() == 4, "label_tensor value is incorrect"



def test_CustomRNN_score_3():
    model = CustomRNN(input_dim = 64, hidden_dim = 128, output_dim = 20)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 27284  
    assert total_params == expected_params, f"Total number of CustomRNN model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.randn(4, 5, 64) # (batch_size, seq_len, input_dim)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)

    assert output.shape == (4, 20), "CustomRNN output shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert torch.sum(output).item() == pytest.approx(-12.138416290283203, rel=1e-3), "CustomRNN forward pass gave different value"

    # for val in output[1,5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[1,5:10].detach(), torch.tensor([12.25534439086914, -9.160823822021484, 12.034659385681152, -6.650180339813232, 10.575711250305176]), rtol=1e-3).all(),"CustomRNN Forward pass gave different value"



def test_params_count_score_1():
    assert num_params_customRNN == 32*64 + 64*64 + 64 + 64*10 + 10, "num_params_customRNN calculation is incorrect"


def test_shakespear_score_1():
    VOCAB = list(string.ascii_letters + " ,.:")
    VOCAB_SIZE = len(VOCAB)
    INDEX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
    CHAR_TO_INDEX = {c: i for i, c in enumerate(VOCAB)}

    ShakespeareDataset.__getitem__.__globals__["VOCAB_SIZE"] = VOCAB_SIZE

    text = "To be, or not to be, that is the question:"
    integer_encoded_text = torch.tensor([CHAR_TO_INDEX[char] for char in text], dtype=torch.long)
    dataset = ShakespeareDataset(integer_encoded_text, seq_length = 5)
    input_seq, target_seq = dataset[1]

    assert isinstance(target_seq, torch.Tensor), "ShakespeareDataset target label should be a torch Tensor"

    assert (input_seq.argmax(dim=1) == torch.tensor([14, 52,  1,  4, 53])).all(), "ShakespeareDataset input sequence is incorrect"
    assert (target_seq == torch.tensor([52,  1,  4, 53, 52])).all(), "ShakespeareDataset target sequence is incorrect"


def test_CharRNN_score_3():
    model = CharRNN(vocab_size = 64, hidden_dim = 128, num_layers = 2)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 66112  
    assert total_params == expected_params, f"Total number of CharRNN model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.randn(4, 5, 64) # (batch_size, seq_len, input_dim)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 128)
    output, hidden = model(input_tensor, hidden)

    assert output.shape == (4, 5, 64), "CharRNN output shape does not match the expected shape."
    assert hidden.shape == (2, 4, 128), "CharRNN hidden state shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert torch.sum(output).item() == pytest.approx(-14.544967651367188, rel=1e-3), "CharRNN forward pass gave different value"

    # print (torch.sum(hidden).item())
    assert torch.sum(hidden).item() == pytest.approx(-30.411645889282227, rel=1e-3), "CharRNN forward pass gave different value"

    # for val in output[1,2, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(output[1,2, 5:10].detach(), torch.tensor([-0.6679195761680603, -10.14504337310791, -4.503549575805664, -5.674564361572266, -4.161682605743408]), rtol=1e-3).all(),"CharRNN Forward pass gave different value"


def test_timeout_score_10():
    assert True