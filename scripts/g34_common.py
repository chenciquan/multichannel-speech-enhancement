"""Uniform backend, bounded matched full/LoRA baseline; historical gates stay frozen."""
import hashlib
import json
from pathlib import Path
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g34_matched_full_v1';DATA=ROOT/'data/g29_v1'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def meta(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def status(**kw):dump(OUT/'status.json',{'utc':now(),**kw})
def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous
def guard():
    plan=meta(OUT/'frozen_plan.json')
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h,n
    assert sha(OUT/'sample_indices.npy')==plan['schedule_sha256']
    return plan
def configure(torch):
    assert str(torch.__version__)=='2.5.1+cu121'
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert not torch.are_deterministic_algorithms_enabled()
    return {'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'global_deterministic':torch.are_deterministic_algorithms_enabled(),
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version()}
def state_digest(state,which='all'):
    h=hashlib.sha256()
    for n,t in state.items():
        if which=='base' and 'lora_' in n:continue
        if which=='adapter' and 'lora_' not in n:continue
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
