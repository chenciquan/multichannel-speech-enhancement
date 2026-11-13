"""Verify the entire numerical matrix; do not reclassify the failed original G33 gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g33_precision_audit_v1';OLD=ROOT/'artifacts/g33_full_gradient_probe_v1'
sha,dump,now=common.sha,common.dump,common.now


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    done=meta(OUT/'NUMERICAL_COMPLETE.json');plan=meta(OUT/'frozen_plan.json')
    assert done['status']=='ok' and done['forwards']==240 and done['optimizer_updates']==0 and done['parameters_unchanged']
    assert not (OLD/'FINAL_ACCEPTANCE_AUDIT.json').exists() and done['G33_original_acceptance_still_failed']
    assert meta(OUT/'power_helper.json')['status']=='released'
    for n,h in plan['hashes'].items():assert sha(ROOT/n)==h,n
    for n,h in done['hashes'].items():assert sha(OUT/n)==h,n
    records=meta(OUT/'forward_records.json');pred=np.load(OUT/'predictions.npz');wave=np.load(OUT/'waveforms.npz')
    expected={f"{p['id']}__{i}__{m}__grad{int(g)}__repeat{r}" for p in plan['profiles'] for i in plan['indices'] for m in plan['model_modes'] for g in plan['grad_enabled'] for r in range(2)}
    assert len(records)==len(expected)==len({r['key'] for r in records})==240
    assert set(pred.files)==set(wave.files)=={r['key'] for r in records}==expected
    for r in records:
        key=r['key'];a=pred[key];b=wave[key]
        assert a.shape==(1,257,93) and a.dtype==np.complex64 and b.shape==(23679,) and b.dtype==np.float32
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['prediction_sha256'] and hashlib.sha256(b.tobytes()).hexdigest()==r['waveform_sha256']
    configured={p['id']:p for p in meta(OUT/'actual_profile_settings.json')}
    for p in plan['profiles']:
        a=configured[p['id']];assert a['cudnn_enabled']==p['enabled'] and a['cudnn_tf32']==p['tf32'] and a['cudnn_deterministic']==p['deterministic']
        assert not a['cudnn_benchmark'] and not a['matmul_tf32'] and not a['global_deterministic']
    comparisons=meta(OUT/'comparisons.json');assert len(comparisons)==240
    names={'cross_model_grad','cross_model_no_grad','full_grad_flag','lora_grad_flag'}|{f'repeat_{m}_grad{int(g)}' for m in ('full','lora') for g in (True,False)}
    assert {(r['profile'],r['index'],r['comparison']) for r in comparisons}=={(p['id'],i,n) for p in plan['profiles'] for i in plan['indices'] for n in names}
    def verify(a,b,result):
        error=np.abs(a-b)
        assert result['exact']==bool(np.array_equal(a,b))
        assert result['within_original_tolerance']==bool(np.allclose(a,b,rtol=plan['rtol'],atol=plan['atol']))
        assert result['max_abs']==float(error.max()) and result['failing_elements']==int((error>plan['atol']+plan['rtol']*np.abs(b)).sum())
    for r in comparisons:
        def key(m,g,rp=0):return f"{r['profile']}__{r['index']}__{m}__grad{int(g)}__repeat{rp}"
        n=r['comparison']
        if n=='cross_model_grad':a,b=key('lora',True),key('full',True)
        elif n=='cross_model_no_grad':a,b=key('lora',False),key('full',False)
        elif n=='full_grad_flag':a,b=key('full',True),key('full',False)
        elif n=='lora_grad_flag':a,b=key('lora',True),key('lora',False)
        else:
            mode=n.split('_')[1];enabled=n.endswith('1');a,b=key(mode,enabled,0),key(mode,enabled,1)
        verify(pred[a],pred[b],r['prediction']);verify(wave[a],wave[b],r['waveform'])
    replay=meta(OUT/'old_G33_replay.json');assert len(replay)==12
    for m in ('full','lora'):
        original=np.load(OLD/m/'initial_forward_outputs.npz')
        for r in (v for v in replay if v['mode']==m):verify(pred[f"cudnn_tf32_nondet__{r['index']}__{m}__grad1__repeat0"],original[f"record_{r['index']}"],r)
    stages={}
    for stage in (OUT/'stages').iterdir():
        for r in meta(stage/'archive_manifest.json')['files']:assert sha(ROOT/r['source'])==sha(stage/r['snapshot'])==r['sha256']
        stages[stage.name]=sha(stage/'archive_manifest.json')
    audit={'status':'ok','utc':now(),'forwards':240,'comparisons_independently_recomputed':240,
        'stored_waveforms_verified':240,'old_replays_recomputed':12,'optimizer_updates':0,
        'scope':'matrix integrity/replication, not a guarantee of forwardequivalence or oldG33acceptance',
        'G33_original_gate_still_failed':True,'stages':stages,'numerical_complete_sha256':sha(OUT/'NUMERICAL_COMPLETE.json')}
    dump(OUT/'FINAL_DIAGNOSTIC_AUDIT.json',audit,True)
    report=ROOT/'docs/G33B_数值矩阵独立验收.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G33B 数值矩阵独立验收\n\n240次前向、保存预测/波形、240个比较及12个旧结果复现均从NPZ与逐例日志独立核对，实际后端开关、源文件/代码及阶段快照哈希一致。零参数更新，保持唤醒已释放。\n\n这里验收的是矩阵完整性与对照统计的可复算性，不能将未通过原容差的路径称等价。原G33验收失败仍保留，未补写其通过标记；具体原因和可用执行配置须依据完整矩阵读回解释。\n')
    archive('diagnostic_acceptance',report,[OUT/'FINAL_DIAGNOSTIC_AUDIT.json',ROOT/'scripts/g33b_accept.py',OUT/'power_helper.json'])
    dump(OUT/'status.json',{'utc':now(),'stage':'diagnostic_accepted_G33_original_gate_failed'})
    print({'status':'diagnostic_accepted','forwards':240,'old_G33_gate_still_failed':True},flush=True)


if __name__=='__main__':main()
