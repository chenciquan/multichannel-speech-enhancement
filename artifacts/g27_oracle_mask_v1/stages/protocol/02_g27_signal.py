#!/usr/bin/env python3
"""Oracle reference-power masks as a diagnostic, not an achievable upper bound."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import g8_common as common
from g22_filter_worker import clean_field,audio,initialize
from mcse.beamforming import _input,_stft,_wave,REFERENCE,mask_mvdr
from mcse.wiener import masked_covariances,wiener_weights,mask_mwf
from mcse.components import component_metrics
from mcse.evaluation import MetricEvaluator

ROOT=common.ROOT;OUT=ROOT/'artifacts/g27_oracle_mask_v1';SOURCE=ROOT/'artifacts/g22_matched_rooms_v1'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def meta(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def shape_error(estimated,truth):
    """True-energy-weighted trace-normalized squared Frobenius discrepancy."""
    te=estimated.diagonal(dim1=-2,dim2=-1).real.sum(-1).clamp_min(0)
    tt=truth.diagonal(dim1=-2,dim2=-1).real.sum(-1).clamp_min(0)
    a=estimated/te.clamp_min(1e-30)[:,None,None]
    b=truth/tt.clamp_min(1e-30)[:,None,None]
    return float((((a-b).abs().square().sum((-2,-1)))*tt).sum()/tt.sum().clamp_min(1e-30))


@torch.inference_mode()
def evaluate(job):
    c,direct,save=job
    assert not torch.cuda.is_initialized()
    x=audio(c['noisy_path'],c['input_sha256']);ref=audio(c['reference_path'],c['reference_sha256'])
    s=clean_field(c['source_id'],c['speech_RIR_path'],c['common_scale'])
    assert np.array_equal(s[:,REFERENCE],ref)
    n=x.astype(np.float64)-s.astype(np.float64)
    X,S,N=(_stft(_input(a)) for a in (x,s,n))
    # Exactly the existing masking formula, with the verified true reference replacing E.
    E=_stft(torch.from_numpy(np.ascontiguousarray(ref,dtype=np.float64)))
    mask=E.abs().square()/(E.abs().square()+(X[REFERENCE]-E).abs().square()+1e-10)
    learned=audio(direct['output_path'],direct['output_sha256'])
    L=_stft(torch.from_numpy(np.ascontiguousarray(learned,dtype=np.float64)))
    lm=L.abs().square()/(L.abs().square()+(X[REFERENCE]-L).abs().square()+1e-10)
    ones=torch.ones_like(mask);T=X.shape[-1]
    true_s=AF.psd(S,ones,normalize=False)/T;true_n=AF.psd(N,ones,normalize=False)/T
    ps,pn=masked_covariances(X,mask);ls,ln=masked_covariances(X,lm)
    diagnostics={'true_mask_speech_shape_error':shape_error(ps,true_s),
                 'true_mask_noise_shape_error':shape_error(pn,true_n),
                 'McNet_mask_speech_shape_error':shape_error(ls,true_s),
                 'McNet_mask_noise_shape_error':shape_error(ln,true_n),
                 'mask_mean':float(mask.mean()),'mask_fraction_below_001':float((mask<.01).double().mean()),
                 'partition_relative_max':float((ps+pn-AF.psd(X,ones,normalize=False)/T).abs().max()/(AF.psd(X,ones,normalize=False).abs().max()/T+1e-30))}
    assert diagnostics['partition_relative_max']<1e-10
    evaluator=MetricEvaluator();rows=[]
    for method in ('mvdr','mwf'):
        if method=='mvdr':
            a=AF.psd(X,mask,normalize=True,eps=1e-10);b=AF.psd(X,1-mask,normalize=True,eps=1e-10)
            w=AF.mvdr_weights_souden(a,b,REFERENCE,diagonal_loading=True,diag_eps=1e-7,eps=1e-8)
        else:w,_=wiener_weights(ps,pn,mu=1.)
        y,ys,yn=(_wave(AF.apply_beamforming(w,z),c['frames']) for z in (X,S,N))
        assert np.isfinite(y).all()
        linearity=float(np.max(np.abs(y.astype(float)-ys.astype(float)-yn.astype(float))))
        assert linearity<1e-6*(1+float(np.abs(y).max()))
        if not save:
            check,_=(mask_mvdr(x,ref) if method=='mvdr' else mask_mwf(x,ref,mu=1.))
            assert np.array_equal(check,y),'Oracle plug-in must equal unchanged public filter function'
        quality=evaluator.evaluate(ref,y)
        assert all(quality['status'][k]['status']=='ok' for k in ('si_sdr_db','pesq_wb','stoi'))
        row={**c,'system':'truth_ref_power_mask_'+method,'oracle_clean_information':True,'deployable':False,
             'theoretical_upper_bound':False,'mask_target':'full reverberant quantized reference speech, not dry speech',
             'mask_diagnostics':diagnostics,'components':component_metrics(ref,n[:,REFERENCE],ys,yn),
             'linearity_max_abs':linearity,'metrics':quality,'status':'ok'}
        if save:
            path=OUT/'audio'/row['system']/(c['sample_id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
            z,sr=sf.read(path,dtype='float32');assert sr==16000 and np.array_equal(z,y)
            row.update(output_path=str(path),output_sha256=sha(path))
        rows.append(row)
    return rows


def main():
    assert meta(ROOT/'artifacts/g26_room_ASR_level_v1/FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert (ROOT/'artifacts/g26_room_ASR_level_v1/stages/scientific_readback/archive_manifest.json').exists()
    OUT.mkdir(exist_ok=False)
    names=['scripts/g27_signal.py','scripts/g22_filter_worker.py','scripts/g22_common.py','scripts/g22_prepare.py',
           'src/mcse/beamforming.py','src/mcse/wiener.py','src/mcse/components.py','src/mcse/reverberation.py','src/mcse/evaluation.py',
           'data/g22_v1/manifest.jsonl','artifacts/g22_matched_rooms_v1/evaluation/direct_signals.jsonl',
           'artifacts/g22_matched_rooms_v1/evaluation/truth_signals.jsonl','artifacts/g22_matched_rooms_v1/selected_sources.json',
           'artifacts/g22_matched_rooms_v1/RIR_manifest.json','artifacts/g18_room_stress_v1/RIR_manifest.json',
           'artifacts/g26_room_ASR_level_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'utc':now(),'cases':432,'outputs':864,'max_seconds':2700,'metric_workers':2,'oracle':True,'deployable':False,
          'mask':'abs(Sref)^2/(abs(Sref)^2+abs(Xref-Sref)^2+1e-10), exactly old E-to-mask formula',
          'comparison':'unchanged McNet mask and true covariance outputs from G22, same inputs and solvers',
          'shape_error':'sum_f trace(Rtrue_f)*||Rhat_f/trace(Rhat_f)-Rtrue_f/trace(Rtrue_f)||F^2 / sum_f trace(Rtrue_f), trace floor1e-30',
          'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    protocol=ROOT/'docs/G27A_真值功率比掩码瓶颈诊断协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G27A 真值功率比掩码瓶颈诊断协议\n\n复用 G22 全部432个带噪条件，四位旧说话人各一句、两房间、请求RT=0/.15/.35、三个方向、两噪声及原场型。每例产生真值参考功率比掩码的MVDR和μ=1 MWF，共864输出。不是新独立验证、部署系统、最优掩码或理论上界。\n\n只把原有 E-to-mask 公式中的E替换为已验证的全混响量化参考语音，不改STFT、协方差归一化、求解器或正则。MVDR仍分别按掩码和归一化；MWF仍共同按帧数归一化，不把它们比较解释成只换求解器。权重使用真值，必须标明不可部署。\n\n预检固定首个源语句、OOFFICE、点源相对−80°、目标20°，覆盖两房间和全部RT，共6条件12输出。验证量化参考恢复、与未改动滤波函数输出逐位相等、分量可加和质量指标。通过后归档，再执行全部条件。无新训练、ASR、超参搜索或下载。\n\n保存功率比掩码及原McNet掩码相对真值协方差的真值能量加权形状误差，形状误差为本项目诊断量，不能等同论文MISD或识别正确率。另保存固定滤波器对语音/有效噪声分量的响应指标。原始McNet与真值协方差波形和成绩直接复用。信号阶段通过后再单独决定识别评分。\n')
    archive('protocol',protocol,[OUT/'frozen_plan.json',ROOT/'scripts/g27_signal.py'])
    cases=[r for r in read(ROOT/'data/g22_v1/manifest.jsonl') if r['condition']!='clean'];assert len(cases)==432
    direct={r['sample_id']:r for r in read(SOURCE/'evaluation/direct_signals.jsonl') if r['system']=='mcnet'}
    first=meta(SOURCE/'selected_sources.json')[0]['source_id']
    probes=[c for c in cases if c['source_id']==first and c['noise_id']=='OOFFICE' and c['noise_field']=='point' and c['relative_angle']==-80 and c['target_azimuth']==20]
    assert len(probes)==6
    torch.set_num_threads(1);started=time.monotonic()
    pre=[r for c in probes for r in evaluate((c,direct[c['sample_id']],False))];assert len(pre)==12
    dump(OUT/'preflight.json',pre,True)
    report=ROOT/'docs/G27A_真值掩码预检结果.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G27A 真值掩码预检\n\n固定12项通过真值参考恢复、与现成滤波函数波形逐位相等、固定分量可加与有限指标检查。只证明实现对应冻结的诊断，不代表较好效果或创新。\n')
    archive('preflight',report,[OUT/'preflight.json',OUT/'frozen_plan.json'])
    rows=[]
    with (OUT/'signals.jsonl').open('x',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        jobs=[(c,direct[c['sample_id']],True) for c in cases]
        for offset in range(0,len(jobs),12):
            assert time.monotonic()-started<plan['max_seconds']
            for result in pool.map(evaluate,jobs[offset:offset+12]):
                for r in result:rows.append(r);log.write(json.dumps(r,allow_nan=False)+'\n')
            log.flush();dump(OUT/'status.json',{'utc':now(),'stage':'oracle_mask_signal','done':len(rows),'total':864})
    assert len(rows)==len({(r['system'],r['sample_id']) for r in rows})==864
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items()) and not torch.cuda.is_initialized()
    buckets=defaultdict(list)
    for r in rows:buckets[r['room_id'],r['requested_rt60'],r['system']].append(r)
    summary=[{'room':room,'rt':rt,'system':system,'rows':len(rs),
        **{m:float(np.mean([r['metrics'][m] for r in rs])) for m in ('si_sdr_db','pesq_wb','stoi')},
        **{m:float(np.mean([r['components'][m] for r in rs])) for m in ('speech_error_nmse','target_gain_db','noise_reduction_db')},
        'mask_diagnostics':{m:float(np.mean([r['mask_diagnostics'][m] for r in rs])) for m in rs[0]['mask_diagnostics']}}
        for (room,rt,system),rs in sorted(buckets.items())]
    dump(OUT/'summary.json',summary,True)
    marker={'status':'ok','utc':now(),'rows':864,'cases':432,'oracle':True,'deployable':False,'ASR_performed':False,
        'signals_sha256':sha(OUT/'signals.jsonl'),'summary_sha256':sha(OUT/'summary.json'),'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'SIGNAL_COMPLETE.json',marker,True)
    report=ROOT/'docs/G27B_真值功率比掩码信号诊断.md'
    lines=['# G27B 真值功率比掩码信号诊断','','以下系统含真值，均不可部署。全混响目标，不是干声去混响任务；不称最优或上界。本阶段未运行ASR。','',
           '|房间|请求RT|系统|记录|SI-SDR|PESQ|STOI|语音NMSE|目标增益dB|噪声衰减dB|','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['room']}|{r['rt']}|{r['system']}|{r['rows']}|"+'|'.join(f'{r[m]:.4f}' for m in ('si_sdr_db','pesq_wb','stoi','speech_error_nmse','target_gain_db','noise_reduction_db'))+'|')
    lines+=['','逐例平均，共432个配对条件而非432句。有效噪声包含量化差额。诊断量与原McNet掩码对照逐例保存，不能直接推断词语遗漏或训练能够达到这些分数。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('signal',report,[OUT/'SIGNAL_COMPLETE.json',OUT/'summary.json',ROOT/'scripts/g27_signal.py'])
    dump(OUT/'status.json',{'utc':now(),'stage':'signal_archived_pending_readback'})
    print('G27 oracle-mask signal stage archived',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'utc':now(),'error':repr(e)})
        traceback.print_exc();raise
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('G27 signal job ended\n')
