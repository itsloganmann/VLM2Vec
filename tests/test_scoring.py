import torch

from retriever import scoring


def test_maxsim_exact_match():
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    document = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    score = scoring.maxsim_score(query, document)
    assert torch.allclose(score, torch.tensor([2.0]))


def test_dot_score():
    q = torch.tensor([[1.0, 0.0]])
    d = torch.tensor([[0.0, 1.0]])
    score = scoring.dot_score(q, d)
    assert torch.allclose(score, torch.tensor([0.0]))


def test_hybrid_multi_vector():
    q = torch.randn(1, 3, 4)
    d = torch.randn(1, 5, 4)
    score = scoring.hybrid_score(q, d)
    assert score.shape == (1,)


def test_pad_to_static():
    tensors = [torch.ones(2, 3), torch.ones(4, 3)]
    padded = scoring.pad_to_static(tensors, pad_multiple=2)
    assert padded.shape[1] % 2 == 0
    assert padded.shape[-1] == 3
