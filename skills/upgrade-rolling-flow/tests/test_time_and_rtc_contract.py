"""RTC-prefix and packed-token time are contracts, not automatic RTC integration."""
import torch
import pytest
from token_time_torch import (
    as_action_time,prefix_mask_from_lengths,map_action_time_to_tokens,
    apply_clean_prefix,prefix_safe_update
)


def test_scalar_time_equal_token_time_without_copy():
    x=torch.tensor([.3,.7])
    a=as_action_time(x,batch_size=2,horizon=6)
    b=x[:,None].expand(-1,6)
    assert torch.equal(a,b)
    assert as_action_time(b,batch_size=2,horizon=6).shape==(2,6)
    with pytest.raises(ValueError):as_action_time(torch.ones(2,5),batch_size=2,horizon=6)


def test_nonaction_tokens_require_explicit_time():
    time=torch.tensor([[.5,.8],[.1,.3]])
    index=torch.tensor([-1,0,0,1],dtype=torch.long)
    with pytest.raises(ValueError):map_action_time_to_tokens(time,index)
    mapping=map_action_time_to_tokens(time,index,auxiliary_time=torch.full((2,4),.42))
    assert torch.allclose(mapping,torch.tensor([[.42,.5,.5,.8],[.42,.1,.1,.3]]))


@pytest.mark.parametrize('clean_time',[0.0,1.0])
def test_clean_prefix_matches_time_convention_and_freezes(clean_time):
    source=torch.randn(2,8,3)
    target=torch.zeros_like(source)+.25
    mask=prefix_mask_from_lengths(torch.tensor([2,5]),horizon=8)
    conditioned,time=apply_clean_prefix(source,torch.full((2,),.7),prefix_values=target,
        prefix_mask=mask,clean_time=clean_time)
    assert torch.equal(conditioned[mask],target[mask])
    assert torch.equal(time[mask],torch.full_like(time[mask],clean_time))
    # Even with a non-finite unused velocity at an explicitly frozen prefix, update is exact.
    v=torch.ones_like(source)
    v[mask]=float('nan')
    updated=prefix_safe_update(conditioned,v,torch.full((2,8),.1),
        prefix_values=target,prefix_mask=mask)
    assert torch.equal(updated[mask],target[mask])
    assert torch.isfinite(updated).all()


def test_prefix_length_rejects_fractional_and_out_of_bounds():
    with pytest.raises(TypeError):prefix_mask_from_lengths(torch.tensor([1.4]),horizon=8)
    with pytest.raises(ValueError):prefix_mask_from_lengths(torch.tensor([9]),horizon=8)
