#!/usr/bin/env python3
"""Preserve the interrupted room job and independently reproduce legacy geometry."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
import g8_common as common
ROOT=common.ROOT;OUT=ROOT/'artifacts/g13_geometry_audit_v1'
sha=common.sha;dump=common.dump;read=common.read;now=common.now
from mcse.geometry import rotate_azimuth,azimuth_degrees,wrap_degrees
from mcse.spatial_noise import direction,highpass40
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr
from mcse.components import pcm24_field
from g5_prepare import rotate as legacy_rotate


def archive(stage,report,files):
    old=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=old


def main():
    OUT.mkdir(exist_ok=False);old=ROOT/'artifacts/g12_room_stress_v2'
    assert json.loads((old/'power_helper.json').read_text())['status']=='released' and not (old/'evaluation').exists()
    assert not (ROOT/'data/g12_v1/manifest.jsonl').exists()
    preserved={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data/g12_v1').rglob('*') if p.is_file()}
    dump(OUT/'interrupted_G12_file_inventory.json',preserved,True)
    stopped={'status':'stopped_before_any_room_model_scores','stopped_utc':json.loads((old/'status.json').read_text())['utc'],'reason':'Coordinate audit: legacy rotate positive matrix decrements declared azimuth; speech and noise rotated opposite ways',
        'G12A_remains_valid':True,'G12B_incomplete_not_accepted':True,'room_evaluation_exists':False,'partial_files_retained':len(preserved),'power_helper_released':True,
        'pipeline_source_hashes_still_match':all(sha(ROOT/n)==h for n,h in json.loads((old/'PIPELINE_READY.json').read_text())['files'].items())}
    assert stopped['pipeline_source_hashes_still_match'];dump(old/'GEOMETRY_STOP_AUDIT.json',stopped,True)
    geometry=json.loads((ROOT/'configs/g1_geometry.json').read_text());pos=geometry['microphone_positions_m'];effects=[]
    for target in (-40,20,70):
        for rel in (-80,25):
            legacy=legacy_rotate(direction(20+rel,-15),target-20);fixed=rotate_azimuth(direction(20+rel,-15),target-20)
            actual=wrap_degrees(azimuth_degrees(legacy)-target);corrected=wrap_degrees(azimuth_degrees(fixed)-target)
            assert abs(wrap_degrees(corrected-rel))<1e-10
            effects.append({'target':target,'nominal_relative':rel,'actual_legacy_noise_azimuth':azimuth_degrees(legacy),'actual_legacy_relative':actual,'correct_relative':corrected})
    noise_rows=json.loads((ROOT/'data/g3_v1/sources/noise_manifest.json').read_text())+json.loads((ROOT/'artifacts/g7_unseen_noise_v1/noise_sources.json').read_text());noise={}
    def getnoise(name):
        if name not in noise:
            r=next(r for r in noise_rows if r['noise_id']==name);assert sha(r['path'])==r['sha256'];a,sr=sf.read(r['path'],dtype='float64');assert sr==16000;noise[name]=highpass40(a)
        return noise[name]
    stages=[('G5_validation',ROOT/'data/g5_v1/validation/manifest.jsonl',ROOT/'data/g3_v1/sources'),
      ('G5_confirmation',ROOT/'data/g5_v1/confirmation/manifest.jsonl',ROOT/'data/g5_v1/sources'),
      ('G7',ROOT/'data/g7_v1/manifest.jsonl',ROOT/'data/g5_v1/sources'),('G10',ROOT/'data/g10_v1/manifest.jsonl',ROOT/'data/g10_v1/sources')]
    coverage={};probes=[]
    with (OUT/'actual_point_geometry.jsonl').open('x') as log:
        for name,path,sourcebase in stages:
            cases=read(path);points=[r for r in cases if r['condition']=='point'];nwrong=0
            source_manifest=sourcebase/'speech_manifest.jsonl';sources={r['source_id']:r for r in read(source_manifest)}
            selected=[];seen=set()
            for r in points:
                target=r['target_azimuth'];rel=r['relative_angle'];u=direction(20+rel,-15);actualu=legacy_rotate(u,target-20);az=azimuth_degrees(actualu);actualrel=wrap_degrees(az-target);wrong=abs(wrap_degrees(actualrel-rel))>1e-9;nwrong+=wrong
                log.write(json.dumps({'stage':name,'sample_id':r['sample_id'],'target_azimuth':target,'nominal_relative':rel,'actual_relative':actualrel,'actual_noise_azimuth':az,'mislabeled':wrong})+'\n')
                group=(target,rel)
                if group not in seen and r['snr_db']==0:seen.add(group);selected.append(r)
            coverage[name]={'manifest':str(path.relative_to(ROOT)),'manifest_sha256':sha(path),'all_cases':len(cases),'point_cases':len(points),'point_relative_mismatch_cases':int(nwrong),'non_point_legacy_motion_also_opposite':True}
            assert len(selected)==6
            for r in selected:
                srcr=sources[r['source_id']];sp=sourcebase/srcr['path'];assert sha(sp)==srcr['sha256'];mono,sr=sf.read(sp,dtype='float64');assert sr==16000
                speech=render_plane_wave(mono,geometric_delays(pos,direction(r['target_azimuth'],10)));lo,hi=r['noise_intervals'][0];clip=getnoise(r['noise_id'])[lo:hi];assert len(clip)==len(mono)
                u=direction(20+r['relative_angle'],-15);nf=render_plane_wave(clip,geometric_delays(pos,legacy_rotate(u,r['target_azimuth']-20)))
                x,_,_=mix_at_snr(speech,nf,0,speech_activity_mask(speech[:,4]));quant=pcm24_field(x*r['common_scale']);saved,rr=sf.read(r['noisy_path'],dtype='float32',always_2d=True);assert rr==16000 and sha(r['noisy_path'])==r['input_sha256']
                error=float(np.max(np.abs(quant-saved)));assert np.array_equal(quant,saved)
                corrected=render_plane_wave(clip,geometric_delays(pos,rotate_azimuth(u,r['target_azimuth']-20)));correct_x,_,_=mix_at_snr(speech,corrected,0,speech_activity_mask(speech[:,4]));correct_q=pcm24_field(correct_x*r['common_scale'])
                probes.append({'stage':name,'sample_id':r['sample_id'],'legacy_all6ch_max_error':error,'corrected_geometry_vs_saved_max_error':float(np.max(np.abs(correct_q-saved))),
                    'legacy_input_sha256':r['input_sha256'],'corrected_not_saved_as_new_case':True})
    # Reproduce actual stored training arrays, not only their current builder code.
    train=json.loads((ROOT/'data/g5_v1/rotated/train_catalog.json').read_text());srcs={r['source_id']:r for r in read(ROOT/'data/g3_v1/sources/speech_manifest.jsonl')};tx=np.load(ROOT/'data/g5_v1/rotated/train_x.npy',mmap_mode='r')
    trainprobes=[]
    for index,r in [(i,r) for i,r in enumerate(train) if r['field']=='point'][:3]:
        source=srcs[r['source_id']];sp=ROOT/'data/g3_v1/sources'/source['path'];assert sha(sp)==source['sha256'];mono,sr=sf.read(sp,dtype='float64');mono=mono[r['source_crop_start']:r['source_crop_start']+16000]
        s=render_plane_wave(mono,geometric_delays(pos,direction(r['target_azimuth'],10)));lo,hi=r['noise_intervals'][0];nf=render_plane_wave(getnoise(r['noise_id'])[lo:hi],geometric_delays(pos,legacy_rotate(r['base_noise_directions'][0],r['target_azimuth']-20)))
        x,_,_=mix_at_snr(s,nf,r['snr_db'],speech_activity_mask(s[:,4]));reconstructed=np.ascontiguousarray((x*r['common_scale']).T,dtype=np.float32);assert np.array_equal(reconstructed,tx[index])
        trainprobes.append({'index':index,'target':r['target_azimuth'],'actual_relative_noise_azimuth':wrap_degrees(azimuth_degrees(legacy_rotate(r['base_noise_directions'][0],r['target_azimuth']-20))-r['target_azimuth']),
          'base_relative_noise_azimuth':wrap_degrees(r['angle']-20),'all6ch_float32_max_error':float(np.max(np.abs(reconstructed-tx[index])))})
    checks={'all24_saved_waveform_reconstructions_exact':len(probes)==24 and all(r['legacy_all6ch_max_error']==0 for r in probes),'all3_stored_training_reconstructions_exact':len(trainprobes)==3,
        'no_G12_room_model_scores_generated':not (old/'evaluation').exists(),'G12_partial_data_retained':all(sha(ROOT/name)==h for name,h in preserved.items()),'all_corrected_point_offsets_match':all(abs(wrap_degrees(r['correct_relative']-r['nominal_relative']))<1e-10 for r in effects)}
    result={'status':'ok' if all(checks.values()) else 'failed','checks':checks,'coverage':coverage,'table':effects,'saved_waveform_probes':probes,'stored_training_probes':trainprobes,
        'training_policy':'Legacy rotated target moves +delta while noises move -delta in the declared azimuth convention. Not a rigid-scene rotation.',
        'completed_utc':now(),'original_files_modified':False};dump(OUT/'audit.json',result,True);assert result['status']=='ok'
    lines=['# G13A 方位旋转坐标错误审计与历史结论更正','','## 已确认的问题','','2026-11-12在G12代码核查发现并停止：direction采用x=sin(az), y=cos(az)，正az朝右；旧g5_prepare.rotate却是数学xy平面正旋转，会令az减少。旧调用rotate(u,target−20)使噪声朝−delta、语音朝+delta。因此G5训练/验证/确认、G7和G10并非原报告所称的整场景刚性旋转，点噪声相对角标签在目标不为20°时错误。第5麦同波形检查无法发现这一问题：参考通道传播延迟本来就是零。',
      '', '这是我沿用旧辅助函数时漏掉的坐标约定检查。旧测试和哈希证实实现/文件一致，不能证实研究协议的几何含义正确。新增测试独立要求“正90度把正前方转向右方”和“语音与噪声同向刚性旋转”，不靠旧函数自证。',
      '', '## 实际执行证据','','不仅检查当前源码：24个保存的六通道输入（G5验证/确认、G7、G10各6个角度组合）按旧公式全部逐样本重建，最大误差为0；3条实际G5旋转训练数组也逐样本一致。逐例实际点方位旁注另存，不改历史清单或音频。','','|目标|旧名义噪声相对角|实际噪声方位|实际相对角|修正后相对角|','|---:|---:|---:|---:|---:|']
    for r in effects:lines.append(f'|{r["target"]}|{r["nominal_relative"]}|{r["actual_legacy_noise_azimuth"]:.1f}|{r["actual_legacy_relative"]:.1f}|{r["correct_relative"]:.1f}|')
    lines+=['','|阶段|全部条件|点源条件|相对角错误条件|','|---|---:|---:|---:|']
    for name,r in coverage.items():lines.append(f'|{name}|{r["all_cases"]}|{r["point_cases"]}|{r["point_relative_mismatch_cases"]}|')
    lines+=['','## 哪些证据保留、哪些结论撤回','','- G1人工未见明显退步的反馈、未证实稳定吞字的结论不因此改变。G2/G3未使用此旋转函数；G4补充实验直接采用noise_azimuth=target+relative，未落入此处符号错误。',
      '- G5—G11音频和对应信号/双ASR数值仍是实际保存输入的测量；同一输入上直接输出与MVDR的比较、G6推理一致性、G8独立后端计分仍可读。G11条件滤波器分量线性还原也不受方向标签更正影响。',
      '- 撤回“G5旋转分支仅改变整体方位并保持相对几何”的解释；旧rotated实际改变目标与噪声的联合方向分布。−40/20/70的差别混有相对分离变化，不能单独归因为模型目标方向过拟合或刚性旋转的泛化效果。旧模型ID保留作档案标识，今后称legacy counter-rotation，不伪装成修正后重训。',
      '- G7/G10的点源−80/+25是旧基准标签，不是所有目标方向的实际相对角；八源场同样与目标反向移动，不能声称保持目标/噪声相对几何。新说话人隔离事实仍成立，但泛化解释限于这种实际联合分布。',
      '', '## G12处理与下一步','','G12A依赖来源/物理响应/显存/旧输出兼容验证保持有效。G12B运行时发现此问题，模型信号/ASR评分前已终止；99个RIR及部分音频完整保留并逐文件登记，G12B未验收、没有混响效果结论。临时保持唤醒已释放。',
      '', '先完成坐标修正测试和本更正归档，再做单独的修正自由场确认：使用完整旧句/旧噪声与冻结旧模型，逐组明确实际目标/噪声方向、保持相对几何，不增加训练。该检查只能说明旧训练得到的模型在正确测试场景下如何表现；不能证明正确旋转训练的收益。是否重新训练应另立训练/验证/未查看确认协议。混响压力实验暂后移，不把旧误标场景与新混响混在一起。',
      '', '全部旧报告/快照/权重/音频保留；本文件作为其相关几何解释的权威更正入口。后续不得只引用旧整体旋转结论而省略本更正。','', 'UTC：'+now()]
    report=ROOT/'docs/G13A_旋转坐标审计与历史结论更正.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive('audit',report,[OUT/'audit.json',OUT/'actual_point_geometry.jsonl',OUT/'interrupted_G12_file_inventory.json',old/'GEOMETRY_STOP_AUDIT.json',old/'PIPELINE_READY.json',old/'power_helper.json',ROOT/'scripts/g13_geometry_audit.py',ROOT/'src/mcse/geometry.py',ROOT/'tests/test_azimuth_geometry.py'])
    oldout=common.OUT
    try:
        common.OUT=old;common.archive('geometry_stop',report,[old/'GEOMETRY_STOP_AUDIT.json',OUT/'interrupted_G12_file_inventory.json',old/'status.json',old/'PIPELINE_READY.json',ROOT/'artifacts/g12_room_queue_v2.log']+[ROOT/n for n in json.loads((old/'PIPELINE_READY.json').read_text())['files']])
    finally:common.OUT=oldout
    print(json.dumps({'status':result['status'],'checks':checks,'coverage':coverage}))


if __name__=='__main__':main()
