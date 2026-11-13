#!/usr/bin/env python3
"""Oracle-statistic diagnostic, explicitly unavailable at deployment."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[variable]='1'
import json
import time
import traceback
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import g8_common as common
from mcse.beamforming import _input,_stft,_wave,REFERENCE
from mcse.components import pcm24_field,component_metrics
from mcse.reverberation import full_speech
from mcse.wiener import wiener_weights
from mcse.evaluation import MetricEvaluator

ROOT=common.ROOT
OUT=ROOT/'artifacts/g21_truth_covariance_v1'
OLD=ROOT/'artifacts/g20_room_components_v1'
G18=ROOT/'artifacts/g18_room_stress_v1'
sha,dump,now,read=common.sha,common.dump,common.now,common.read


def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    assert json.loads((OLD/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    decision=ROOT/'docs/G20_分量结论与真值统计诊断决策.md'
    with decision.open('x',encoding='utf-8') as f:
        f.write('# G20 分量结论与下一阶段决策\n\n8640项已验收，原输出全部精确复现。MWF在本批条件中通常有更小的目标语音误差，但残余噪声更多；G19显示它没有自动转化为更好的ASR。\n\n例如请求RT=.35、正确刚性训练组：MVDR/MWF语音NMSE为.4886/.2751，目标增益为−8.7444/−4.2683 dB，残余噪声NMSE为.0518/.1779。均为逐例指标平均；并非词语丢失的判据。\n\n下一阶段用同一六通道量化语音/有效噪声的真值协方差替代掩码协方差，固定两种成熟滤波求解器与原正则。它检验替换统计信息后的条件响应，不是可部署方法，也不是理论上界。有限片段中语音与噪声可能相关，必须保存交叉协方差偏差，不假设Rs+Rn严格等于Rx。不训练、不据此调参；先信号阶段归档，再决定后续ASR或新房间确认。\n')
    previous=common.OUT
    try:common.OUT=OLD;common.archive('next_truth_covariance_decision',decision,[OLD/'FINAL_ACCEPTANCE_AUDIT.json'])
    finally:common.OUT=previous
    OUT.mkdir(exist_ok=False);(OUT/'audio').mkdir()
    names=['scripts/g21_truth_covariance.py','src/mcse/wiener.py','src/mcse/beamforming.py','src/mcse/components.py','src/mcse/reverberation.py','src/mcse/evaluation.py',
           'data/g18_v1/manifest.jsonl','artifacts/g18_room_stress_v1/selected_sources.json',
           'artifacts/g20_room_components_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'cases':432,'new_outputs':864,'mu':1,'max_seconds':1800,
          'oracle_information':True,'deployable':False,'theoretical_upper_bound':False,
          'covariance':'True quantized S and effective N=X-S, common T; finite-sample cross-correlation measured',
          'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    protocol=ROOT/'docs/G21A_真值协方差诊断协议.md'
    with protocol.open('x',encoding='utf-8') as f:
        f.write('# G21A 真值协方差诊断协议\n\n复用全部432个G18带噪条件，每例一个真值协方差Souden MVDR及一个μ=1 MWF，共864输出。真值统计不依赖10个神经前端，不能重复为10个独立结果。Rs和Rn以相同帧数归一化；正则沿用G19。\n\n由已验证的量化六通道语音S和N=X−S估计真值协方差，仅用于诊断。报告有限片段交叉协方差，不能宣称理论上界。先固定首语句/办公室/−80点源/20°/三个RT的6项预检归档，再完整运行。保存波形、信号指标、固定分量响应；此阶段不解码ASR或训练。\n')
    archive('protocol',protocol,[OUT/'frozen_plan.json',ROOT/'scripts/g21_truth_covariance.py'])
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    started=time.monotonic();evaluator=MetricEvaluator()
    sources={r['source_id']:r for r in json.loads((G18/'selected_sources.json').read_text())}
    prior=json.loads((OLD/'verified_files.json').read_text());verified={};monos={}
    def verify(path,expected):
        path=str(path)
        if path not in verified:assert sha(path)==expected;verified[path]=expected
        else:assert verified[path]==expected
    def evaluate(c,save):
        assert time.monotonic()-started<plan['max_seconds']
        src=sources[c['source_id']];path=ROOT/'data/g17_v1/sources'/src['path'];verify(path,src['sha256'])
        if c['source_id'] not in monos:
            monos[c['source_id']],sr=sf.read(path,dtype='float64');assert sr==16000
        verify(c['speech_RIR_path'],prior[c['speech_RIR_path']])
        sq=pcm24_field(full_speech(monos[c['source_id']],np.load(c['speech_RIR_path']))*c['common_scale'])
        verify(c['noisy_path'],c['input_sha256']);verify(c['reference_path'],c['reference_sha256'])
        x,sr=sf.read(c['noisy_path'],dtype='float32');ref,rr=sf.read(c['reference_path'],dtype='float32')
        assert sr==rr==16000 and np.array_equal(sq[:,4],ref)
        n=x.astype(float)-sq.astype(float)
        X,S,N=(_stft(_input(a)) for a in (x,sq,n));frames=X.shape[-1]
        ones=torch.ones(X.shape[1:],dtype=torch.float64)
        ps=AF.psd(S,ones,normalize=False)/frames;pn=AF.psd(N,ones,normalize=False)/frames
        px=AF.psd(X,ones,normalize=False)/frames
        cross=float((px-ps-pn).abs().max()/(px.abs().max()+1e-30))
        for method in ('mvdr','mwf'):
            if method=='mvdr':w=AF.mvdr_weights_souden(ps,pn,REFERENCE,diagonal_loading=True,diag_eps=1e-7,eps=1e-8)
            else:w,_=wiener_weights(ps,pn,mu=1.)
            assert torch.isfinite(w).all()
            y,ys,yn=(_wave(AF.apply_beamforming(w,z),c['frames']) for z in (X,S,N))
            linear=float(np.abs(y.astype(float)-ys.astype(float)-yn.astype(float)).max())
            assert linear<1e-6*(1+float(np.abs(y).max()))
            metrics=component_metrics(ref,n[:,4],ys,yn)
            quality=evaluator.evaluate(ref,y)
            assert all(quality['status'][k]['status']=='ok' for k in ('si_sdr_db','pesq_wb','stoi'))
            system='truth_covariance_'+method
            row={**c,'system':system,'oracle_clean_information':True,'deployable':False,
                 'covariance_cross_relative_max':cross,'linearity_max_abs':linear,'components':metrics,'metrics':quality}
            if save:
                output=OUT/'audio'/(system+'_'+c['sample_id']+'.wav')
                with output.open('xb') as stream:sf.write(stream,y,16000,format='WAV',subtype='FLOAT')
                recovered,rate=sf.read(output,dtype='float32');assert rate==16000 and np.array_equal(recovered,y)
                row.update(output_path=str(output),output_sha256=sha(output))
            yield row
    cases=[c for c in read(ROOT/'data/g18_v1/manifest.jsonl') if c['condition']!='clean'];assert len(cases)==432
    probe=[c for c in cases if c['source_id']==next(iter(sources)) and c['noise_id']=='OOFFICE' and c['noise_field']=='point' and c['relative_angle']==-80 and c['target_azimuth']==20]
    pre=[r for c in probe for r in evaluate(c,False)];assert len(pre)==6
    dump(OUT/'preflight.json',pre,True)
    report=ROOT/'docs/G21A_真值统计预检报告.md'
    with report.open('x',encoding='utf-8') as f:f.write('# G21A 真值统计预检\n\n固定6项均通过参考还原、有限矩阵求解、固定分量叠加和客观指标检查。真值信息在此刻意进入权重，仅供诊断，不能作为部署对照或理论上界。\n')
    archive('preflight',report,[OUT/'preflight.json',OUT/'frozen_plan.json'])
    rows=[]
    with (OUT/'signals.jsonl').open('x',encoding='utf-8') as log:
        for i,c in enumerate(cases,1):
            for r in evaluate(c,True):log.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r)
            log.flush()
            if i%24==0:dump(OUT/'status.json',{'stage':'oracle_signal','cases':i,'total':432,'utc':now()})
    assert len(rows)==864 and len({(r['system'],r['sample_id']) for r in rows})==864
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items()) and not torch.cuda.is_initialized()
    buckets=defaultdict(list)
    for r in rows:buckets[r['requested_rt60'],r['system']].append(r)
    summary=[{'rt':rt,'system':s,'rows':len(rs),**{k:float(np.mean([r['metrics'][k] for r in rs])) for k in ('si_sdr_db','pesq_wb','stoi')},
              **{k:float(np.mean([r['components'][k] for r in rs])) for k in ('speech_error_nmse','target_gain_db','noise_reduction_db')}} for (rt,s),rs in sorted(buckets.items())]
    dump(OUT/'summary.json',summary,True);dump(OUT/'verified_files.json',verified,True)
    complete={'status':'ok','rows':864,'cases':432,'oracle':True,'deployable':False,'completed_utc':now(),
              'signals_sha256':sha(OUT/'signals.jsonl'),'summary_sha256':sha(OUT/'summary.json'),
              'cross_relative_max':max(r['covariance_cross_relative_max'] for r in rows),'elapsed_seconds':time.monotonic()-started}
    dump(OUT/'SIGNAL_COMPLETE.json',complete,True)
    text=['# G21B 真值协方差信号诊断','','以下系统使用真值信息，仅供诊断，非可部署方法或理论上界；未运行ASR。',
          '', '|请求RT|系统|例数|SI-SDR dB|PESQ|STOI|语音NMSE|目标增益dB|噪声衰减dB|','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"|{r['rt']}|{r['system']}|{r['rows']}|"+'|'.join(f"{r[k]:.4f}" for k in ('si_sdr_db','pesq_wb','stoi','speech_error_nmse','target_gain_db','noise_reduction_db'))+'|')
    text+=['','逐例算术平均，dB量逐例计算后平均。有效噪声含量化差额。有限片段的Rs+Rn与Rx交叉协方差偏差逐例保存；不假设信号统计独立。由真值替换统计产生的变化不能直接证明某个神经模块的因果机制。','',now()]
    report=ROOT/'docs/G21B_真值协方差信号报告.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    archive('signal',report,[OUT/'SIGNAL_COMPLETE.json',OUT/'summary.json',OUT/'verified_files.json',OUT/'frozen_plan.json',ROOT/'scripts/g21_truth_covariance.py'])
    dump(OUT/'status.json',{'stage':'signal_archived_pending_readback','utc':now()})


if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':repr(e),'utc':now()})
        traceback.print_exc();raise
    finally:
        if OUT.exists():(OUT/'keep_awake_stop').write_text('truth covariance signal job ended\n')
