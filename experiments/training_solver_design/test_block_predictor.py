import torch

from block_predictor import center_blocks, sampled_prediction


def test_arbitrary_center_predictor_preserves_complete_sampling_mean():
    torch.manual_seed(42)
    per_block = torch.randn(4, 11, 7, dtype=torch.double)
    full = per_block.mean(0)
    predictor = torch.randn_like(full)
    assignment = torch.randint(0, 4, (11,))
    labels = torch.tensor([[2, 0], [3, 1]])
    estimates=[]
    for label in range(4):
        selected = labels.flatten() == label
        sampled = 4*per_block[selected].mean(0)/4
        estimates.append(sampled + predictor-sampled_prediction(predictor, assignment, labels, label))
    torch.testing.assert_close(torch.stack(estimates).mean(0),full)


def test_inactive_centers_are_sanitized_and_edges_assigned():
    xy=torch.tensor([[float('nan'),float('inf')],[-3.,20.],[3000.,2000.]])
    radii=torch.tensor([[0,0],[1,1],[1,1]])
    blocks, visible=center_blocks(xy,radii,1080,1920)
    assert blocks.tolist()==[0,0,134]
    assert visible.tolist()==[False,True,True]
