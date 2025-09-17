import pytest
import inspect
import time
import string

import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import spacy

from seq2seq import collate_batch, SentimentClassifier, collate_seq2seq_batch, Encoder, Decoder, Seq2Seq


pytest.global_start_time = time.time()
MAX_DURATION_SECONDS = 10

@pytest.fixture(autouse=True)
def check_global_timeout():
    """Fail the test if total elapsed time exceeds MAX_DURATION_SECONDS."""
    if time.time() - pytest.global_start_time > MAX_DURATION_SECONDS:
        pytest.fail(f"⏰ Test suite exceeded {MAX_DURATION_SECONDS} seconds timeout.")


def test_collate_batch_score_1():
    MAX_LENGTH = 4
    corpus = "I love you very much. I adore you. It was amazing!"
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english") 
    vocab = build_vocab_from_iterator([tokenizer(corpus)], specials=["<pad>", "<unk>"], min_freq = 1)
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    collate_batch.__globals__["MAX_LENGTH"] = MAX_LENGTH
    collate_batch.__globals__["vocab"] = vocab
    collate_batch.__globals__["text_pipeline"] = text_pipeline
    collate_batch.__globals__["label_pipeline"] = label_pipeline

    batch = [
        (1, "It was totally full of love."),
        (2, "The love!"),
    ]

    text_tensor, label_tensor, length_tensor = collate_batch(batch)

    print(vocab.get_itos(), tokenizer(corpus))
    print(text_tensor, label_tensor, length_tensor)

    assert (text_tensor == torch.tensor([[ 8, 12,  1,  1], [ 1,  9, 5,  0]])).all(), "collate_batch output text_tensor is incorrect"
    assert (label_tensor == torch.tensor([0, 1])).all(), "collate_batch output label_tensor is incorrect"
    assert (length_tensor == torch.tensor([4, 3])).all(), "collate_batch output lengths_tensor is incorrect"




def test_Sentiment_score_2():
    model = SentimentClassifier(vocab_size = 100, embed_dim = 32, hidden_dim = 64, num_layers = 3,
                                num_classes = 10, pad_index = 10)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 95498
    assert total_params == expected_params, f"Total number of SentimentClassifier model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    X = torch.randint(0, 100, (8, 50))  # (batch_size, seq_len)
    lengths = torch.randint(1, 50, (8,))    

    with torch.no_grad():
        for name, param in model.named_parameters():
            # print("---", name, param.shape)
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(X, lengths)

    assert output.shape == (8, 10), "SentimentClassifier output shape does not match the expected shape."

    # print (torch.sum(output).item())
    assert torch.sum(output).item() == pytest.approx(11.961236000061035, rel=1e-1), "SentimentClassifier forward pass gave different value"

    for val in output[1,5:10].detach():
        print(val.item())
    assert torch.isclose(output[1,5:10].detach(), torch.tensor([-0.05214540660381317, -2.1009490489959717, 8.540902137756348, -4.330341339111328, 2.180689573287964]), rtol=1e-1).all(),"SentimentClassifier Forward pass gave different value"



def test_collate_seq2seq_score_2():
    SPECIAL_TOKENS = ["<eos>", "<sos>", "<unk>", "<pad>"]

    spacy_fr = spacy.load('fr_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    tokenizer_french = get_tokenizer(lambda text: [tok.text.lower() for tok in spacy_fr.tokenizer(text)])
    tokenizer_english = get_tokenizer(lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)])

    source_corpus = "je suis etudiant. j'aime le deep learning. pytorch est une super bibliotheque."
    target_corpus = "i am a student. i love deep learning. pytorch is a great library."

    source_vocab = build_vocab_from_iterator([tokenizer_french(source_corpus)], specials=SPECIAL_TOKENS, min_freq = 1)
    target_vocab = build_vocab_from_iterator([tokenizer_english(target_corpus)], specials=SPECIAL_TOKENS, min_freq = 1)
    source_vocab.set_default_index(source_vocab["<unk>"])
    target_vocab.set_default_index(target_vocab["<unk>"])

    print("source_vocab:", source_vocab.get_itos(), tokenizer_french(source_corpus))
    print("target_vocab:", target_vocab.get_itos(), tokenizer_english(target_corpus))

    PAD_TOKEN_IDX  = source_vocab["<pad>"]
    SOS_TOKEN_IDX  = source_vocab["<sos>"]
    EOS_TOKEN_IDX  = source_vocab["<eos>"]


    collate_seq2seq_batch.__globals__["SOS_TOKEN_IDX"] = SOS_TOKEN_IDX
    collate_seq2seq_batch.__globals__["EOS_TOKEN_IDX"] = EOS_TOKEN_IDX
    collate_seq2seq_batch.__globals__["PAD_TOKEN_IDX"] = PAD_TOKEN_IDX
    collate_seq2seq_batch.__globals__["source_vocab"] = source_vocab
    collate_seq2seq_batch.__globals__["tokenizer_french"] = tokenizer_french
    collate_seq2seq_batch.__globals__["target_vocab"] = target_vocab
    collate_seq2seq_batch.__globals__["tokenizer_english"] = tokenizer_english

    batch = [
        ("je suis etudiant.", "i am stud."),
        ("j'aime le deep learn.", "i love deep learning."),
    ]

    source_tensor, target_tensor = collate_seq2seq_batch(batch)

    print("SOS_TOKEN_IDX:", SOS_TOKEN_IDX)
    print("EOS_TOKEN_IDX:", EOS_TOKEN_IDX)
    print("PAD_TOKEN_IDX:", PAD_TOKEN_IDX)  
    print("UNK_TOKEN_IDX:", source_vocab["<unk>"])  
    print(source_tensor)
    print(target_tensor)

    assert (source_tensor == torch.tensor([[ 1, 11, 15,  9,  4,  0,  3,  3],[ 1, 10,  5, 12,  7,  2,  4,  0]])).all(), "collate_seq2seq_batch output source_batch is incorrect"
    assert (target_tensor == torch.tensor([[ 1,  6,  7,  2,  4,  0,  3], [ 1,  6, 13,  8, 11,  4,  0]])).all(), "collate_seq2seq_batch output target_batch is incorrect"



def test_Encoder_score_1():
    model = Encoder(vocab_size = 100, embed_dim = 32, hidden_dim = 64,
                    num_layers = 3, pad_token_index = 10, dropout = 0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 94848
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
    hidden, cell = model(X)

    assert hidden.shape == (3, 8, 64), "Encoder hidden state shape does not match the expected shape."
    assert cell.shape == (3, 8, 64), "Encoder cell state shape does not match the expected shape."

    # print (torch.sum(hidden).item())
    assert torch.sum(hidden).item() == pytest.approx(5.9782538414001465, rel=1e-3), "Encoder forward pass gave different value"

    # print (torch.sum(cell).item())
    assert torch.sum(cell).item() == pytest.approx(19.788776397705078, rel=1e-3), "Encoder forward pass gave different value"

    # for val in hidden[1, 2, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(hidden[1,2, 5:10].detach(), torch.tensor([-0.4697258770465851, 0.06566240638494492, 0.8286478519439697, 0.7073583006858826, -0.4758797585964203]), rtol=1e-3).all(),"Encoder Forward pass gave different value"


def test_Decoder_score_3():
    model = Decoder(vocab_size = 100, embed_dim = 32, hidden_dim = 64,
                    num_layers = 3, pad_token_index = 10, dropout = 0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 101348  
    assert total_params == expected_params, f"Total number of Decoder model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    input_token = torch.randint(0, 100, (8,))  # (batch_size,)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    hidden = torch.randn(3, 8, 64)  # (num_layers, batch_size, hidden_dim)
    cell = torch.randn(3, 8, 64)    # (num_layers, batch_size, hidden_dim)
    logits, (hidden, cell) = model(input_token, hidden, cell)

    assert logits.shape == (8, 100), "Decoder logits shape does not match the expected shape."
    assert hidden.shape == (3, 8, 64), "Decoder hidden state shape does not match the expected shape."
    assert cell.shape == (3, 8, 64), "Decoder cell state shape does not match the expected shape."

    # print (torch.sum(hidden).item())
    assert torch.sum(hidden).item() == pytest.approx(-11.723123550415039, rel=1e-3), "Decoder forward pass gave different value"

    # print (torch.sum(cell).item())
    assert torch.sum(cell).item() == pytest.approx(4.298758506774902, rel=1e-3), "Decoder forward pass gave different value"

    # for val in logits[3, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(logits[3, 5:10].detach(), torch.tensor([3.232478141784668, -3.198838949203491, 1.6091868877410889, 1.9938790798187256, 2.052966594696045]), rtol=1e-3).all(), "Decoder Forward pass gave different value"

    # for val in hidden[1, 2, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(hidden[1,2, 5:10].detach(), torch.tensor([-0.13731907308101654, 8.353475517645231e-14, 0.7598028779029846, 1.8182072381023318e-07, 7.51759853301337e-06]), rtol=1e-3).all(), "Decoder Forward pass gave different value"

def test_Seq2Seq_score_1():
    enc = Encoder(vocab_size = 100, embed_dim = 32, hidden_dim = 64,
                    num_layers = 3, pad_token_index = 10, dropout = 0.5)
    dec = Decoder(vocab_size = 200, embed_dim = 32, hidden_dim = 64,
                    num_layers = 3, pad_token_index = 10, dropout = 0.5)
    
    model = Seq2Seq(enc, dec)

    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 205896  
    assert total_params == expected_params, f"Total number of Seq2Seq model parameters should be {expected_params}, but got {total_params}."

    torch.manual_seed(0)  # For reproducibility
    X = torch.randint(0, 100, (8, 20)) # (batch_size, src_len)
    y = torch.randint(0, 200, (8, 30)) # (batch_size, tgt_len)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print("---", name, param.shape)
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    logits = model(X, y, teacher_forcing_prob=0.5)

    assert logits.shape == (8, 30, 200), "Seq2Seq logits shape does not match the expected shape."

    # print (torch.sum(logits).item())
    assert torch.sum(logits).item() == pytest.approx(-2688.73095703125, rel=1e-1), "Seq2Seq forward pass gave different value"

    # for val in logits[3, 5, 5:10].detach():
    #     print(val.item())
    assert torch.isclose(logits[3, 5, 5:10].detach(), torch.tensor([5.115123271942139, -2.8813395500183105, 3.384706497192383, 0.36753740906715393, 3.0655736923217773]), rtol=1e-3).all(), "Seq2Seq Forward pass gave different value"




def test_timeout_score_10():
    assert True