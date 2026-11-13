import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from mcse.adaptation import attach_lora,adapter_state,load_adapter


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.freq=torch.nn.Module();self.freq.fc_output_layer=torch.nn.Linear(8,6)
        self.narr=torch.nn.Module();self.narr.fc_output_layer=torch.nn.Linear(6,5)
    def forward(self,x):return self.narr.fc_output_layer(self.freq.fc_output_layer(x))


class AdaptationTests(unittest.TestCase):
    def test_zero_initialization_and_only_low_rank_updates(self):
        torch.manual_seed(81);m=Toy();x=torch.randn(4,8);before=m(x).detach();base={n:p.detach().clone() for n,p in m.named_parameters()}
        self.assertEqual(attach_lora(m),100);torch.testing.assert_close(m(x),before,atol=0,rtol=0)
        opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.1)
        m(x).square().mean().backward();opt.step()
        self.assertGreater(float((m(x)-before).abs().max()),0)
        for n,p in m.named_parameters():
            if n in base:torch.testing.assert_close(p,base[n],atol=0,rtol=0);self.assertIsNone(p.grad)

    def test_adapter_roundtrip_and_bad_state_rejected(self):
        m=Toy();attach_lora(m);state=adapter_state(m);load_adapter(m,state)
        with self.assertRaises(ValueError):load_adapter(m,{})
        broken=dict(state);key=next(iter(broken));broken[key]=torch.full_like(broken[key],float('nan'))
        with self.assertRaises(ValueError):load_adapter(m,broken)
