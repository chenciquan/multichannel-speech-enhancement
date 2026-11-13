"""Thin use of Microsoft loralib on the two registered spatial Linear layers."""
import torch
import loralib

TARGETS=('freq.fc_output_layer','narr.fc_output_layer')


def attach_lora(model,rank=4,alpha=4):
    model.requires_grad_(False)
    for path in TARGETS:
        parent_name,attribute=path.split('.');parent=getattr(model,parent_name);base=getattr(parent,attribute)
        layer=loralib.Linear(base.in_features,base.out_features,r=rank,lora_alpha=alpha,
                             lora_dropout=0,merge_weights=False,bias=base.bias is not None).to(base.weight.device,base.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(base.weight)
            if base.bias is not None:layer.bias.copy_(base.bias)
        setattr(parent,attribute,layer)
    model.requires_grad_(False)
    for name,param in model.named_parameters():
        if name.endswith(('lora_A','lora_B')):param.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_adapter(model,state):
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(state):raise ValueError('Adapter keys do not match the registered trainable parameters')
    with torch.no_grad():
        for name,param in parameters.items():
            if param.shape!=state[name].shape or not torch.isfinite(state[name]).all():raise ValueError('Invalid adapter tensor')
            param.copy_(state[name].to(param.device,param.dtype))


def mask_loss(model,x,target,prediction,data):
    with torch.no_grad():reference_mask=model.io.prepare_target(x=x,yr=target,**data)
    return torch.mean((torch.view_as_real(prediction.to(torch.complex64))-torch.view_as_real(reference_mask))**2)
