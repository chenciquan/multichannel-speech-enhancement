"""Exact binary gains and independent baseline repetitions for ASR validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import soundfile as sf
import g8_common as common
from g19_scores import waveform

ROOT=common.ROOT;OUT=ROOT/'artifacts/g24_gain_repeat_v1';PARENT=ROOT/'artifacts/g23_gain_controls_v1'
sha,read,dump,now=common.sha,common.read,common.dump,common.now


def archive(stage,report,files):
    previous=common.OUT
    try:common.OUT=OUT;common.archive(stage,report,files)
    finally:common.OUT=previous


def main():
    import json
    assert json.loads((PARENT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok'
    assert (PARENT/'stages/scientific_readback/archive_manifest.json').exists()
    OUT.mkdir(exist_ok=False);(OUT/'audio').mkdir()
    bases=[r for r in read(PARENT/'manifest.jsonl') if r['gain_db']==0];assert len(bases)==480
    keys={(r['base_system'],r['sample_id']) for r in bases};assert len(keys)==480
    names=['scripts/g24_prepare.py','artifacts/g23_gain_controls_v1/manifest.jsonl',
           'artifacts/g23_gain_controls_v1/FINAL_ACCEPTANCE_AUDIT.json']
    plan={'frozen_utc':now(),'base_pairs':480,'cases':48,'wave_gains':[1.,.5,.25],
          'observation_order':['original_first','half','quarter','original_second'],
          'expected_observations_each_ASR':1920,'new_waveforms':960,'fresh_decode_every_observation':True,
          'preserve_historical_cache':True,'exact_inverse_expected':True,
          'selection':'Exactly all G23 zero-dB bases; no new score-based sample selection',
          'limits':'Exploratory ASR reproducibility/amplitude control, no training, enhancement, best-gain optimization or human-label claims',
          'hashes':{n:sha(ROOT/n) for n in names}}
    dump(OUT/'frozen_plan.json',plan,True)
    report=ROOT/'docs/G24A_基准重复与精确二进制增益协议.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G24A 基准重复与精确二进制增益协议\n\n完全保留G23的480个系统/条件基准。对每个基准，后续依次真实解码原幅度第一次、×1/2、×1/4、原幅度第二次，共1920观察/识别器；不使用缓存跳过解码。历史文本只读，不把新重复覆盖旧基准。\n\n缩放音频必须逐样本精确反向还原原float32，原幅度两次使用相同文件。若发生跨次/同次重复不一致，应单列并限制归因。没有按结果挑增益或样本，不运行增强网络或训练，数字控制不证明固定音量下人耳可懂度不变。\n')
    archive('protocol',report,[OUT/'frozen_plan.json',ROOT/'scripts/g24_prepare.py'])
    rows=[]
    with (OUT/'manifest.jsonl').open('x',encoding='utf-8') as log:
        for base in bases:
            original,original_key=waveform(base['output_path'],base['output_sha256'])
            files={1.:(base['output_path'],base['output_sha256'],original_key)}
            for gain in (.5,.25):
                result=(original*gain).astype('float32')
                assert np.array_equal(result/gain,original),'Binary gain did not invert exactly; preserve failure'
                name=f"{base['base_system']}_{base['sample_id']}_{'half' if gain==.5 else 'quarter'}.wav"
                path=OUT/'audio'/name
                with path.open('xb') as f:sf.write(f,result,16000,format='WAV',subtype='FLOAT')
                recovered,key=waveform(str(path),sha(path));assert np.array_equal(recovered,result)
                files[gain]=(str(path),sha(path),key)
            for observation,gain in (('original_first',1.),('half',.5),('quarter',.25),('original_second',1.)):
                path,digest,key=files[gain]
                row={**base,'observation':observation,'gain_exact':gain,'output_path':path,'output_sha256':digest,
                     'output_samples_sha256':key,'baseline_samples_sha256':original_key,
                     'gain':gain,'gain_db':float(20*np.log10(gain)),'exact_inverse':True,
                     'scaled_peak':float(np.abs(original).max())*gain,
                     'relative_roundtrip_error':0.,'si_sdr_change_db':None}
                log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
    assert len(rows)==len({(r['base_system'],r['sample_id'],r['observation']) for r in rows})==1920
    assert all(sha(ROOT/n)==h for n,h in plan['hashes'].items())
    done={'status':'ok','utc':now(),'rows':1920,'base_pairs':480,'new_waveforms':960,
          'all_binary_inverse_samples_exact':True,'manifest_sha256':sha(OUT/'manifest.jsonl'),'ASR_run':False}
    dump(OUT/'PREPARATION_COMPLETE.json',done,True)
    report=ROOT/'docs/G24B_二进制增益精确还原验证.md'
    with report.open('x',encoding='utf-8') as f:
        f.write('# G24B 二进制增益精确还原验证\n\n480个基准的×1/2和×1/4波形均经FLOAT保存/回读，反向缩放后逐样本精确还原原float32；没有发生量化舍入或下溢导致的不等。原幅度两次观察指向同一文件。1920个预定观察键完整，960条新波形已保存。\n\n尚未运行识别器，不能提前声称重复性或幅度敏感性被证实。旧G22/G23音频、缓存和成绩保持原样。\n')
    archive('preparation',report,[OUT/'PREPARATION_COMPLETE.json',OUT/'manifest.jsonl',OUT/'frozen_plan.json',ROOT/'scripts/g24_prepare.py'])
    dump(OUT/'status.json',{'stage':'prepared_fresh_ASR_not_started','utc':now()});print(done)


if __name__=='__main__':main()
