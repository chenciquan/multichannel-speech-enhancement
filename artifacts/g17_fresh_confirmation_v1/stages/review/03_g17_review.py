#!/usr/bin/env python3
"""Four fixed source/angle cards; labelled auxiliary listening, never blind data."""
import html,json
import numpy as np
import soundfile as sf
from g17_common import ROOT,OUT,BASE,now,sha,read,dump,archive


def main():
    assert json.loads((OUT/'MAJOR_STAGE_COMPLETE.json').read_text())['status']=='ok';source=read(BASE/'sources/speech_manifest.jsonl');speakers=list(dict.fromkeys(r['speaker_id'] for r in source));assert len(speakers)==4
    cases=read(BASE/'manifest.jsonl');signals=read(OUT/'evaluation/combined_signals.jsonl');idx={(r['system'],r['sample_id']):r for r in signals};cards=[];waves=[];seed=20261112
    panels=[('noisy','带噪第5麦'),('mcnet','原McNet'),(f'fixed_without_clean_identity_s{seed}','固定训练直接'),(f'rotated_without_clean_identity_s{seed}','反向训练直接'),(f'rigid_without_clean_identity_s{seed}','刚性训练直接'),
      (f'fixed_without_clean_identity_s{seed}_mask_mvdr','固定训练＋MVDR'),(f'rotated_without_clean_identity_s{seed}_mask_mvdr','反向训练＋MVDR'),(f'rigid_without_clean_identity_s{seed}_mask_mvdr','刚性训练＋MVDR')]
    for sp,angle in zip(speakers,(-40,20,70,-40)):
        src=next(r for r in source if r['speaker_id']==sp);r=next(r for r in cases if r['source_id']==src['source_id'] and r['noise_id']=='OOFFICE' and r['condition']=='point' and r['relative_angle']==-80 and r['target_azimuth']==angle)
        entries=[('reference','参考干净语音',r['reference_path'],r['reference_sha256'])]+[(name,label,idx[(name,r['sample_id'])]['output_path'],idx[(name,r['sample_id'])]['output_sha256']) for name,label in panels]
        card={'source_id':src['source_id'],'speaker_id':sp,'target_angle':angle,'transcript':src['transcript'],'sample_id':r['sample_id'],'panels':[]}
        for system,label,path,h in entries:
            assert sha(path)==h;a,sr=sf.read(path,dtype='float32');assert sr==16000 and a.ndim==1 and np.isfinite(a).all();waves.append(a)
            card['panels'].append({'system':system,'label':label,'original_path':path,'original_sha256':h,'frames':len(a),'wave_index':len(waves)-1})
        assert len({p['frames'] for p in card['panels']})==1;cards.append(card)
    gain=min(1.,.95/max(float(np.abs(a).max()) for a in waves));folder=OUT/'review';folder.mkdir(exist_ok=False);(folder/'audio').mkdir();body=[]
    for i,card in enumerate(cards):
        parts=[]
        for j,p in enumerate(card['panels']):
            y=np.asarray(waves[p.pop('wave_index')]*gain,dtype='float32');path=folder/'audio'/f'{i}_{j}.wav'
            with path.open('xb') as f:sf.write(f,y,16000,format='WAV',subtype='FLOAT')
            saved,sr=sf.read(path,dtype='float32');assert sr==16000 and np.array_equal(saved,y) and float(np.abs(y).max())<=.950001
            p.update(review_relative_path='audio/'+path.name,review_sha256=sha(path));parts.append('<div class="panel"><b>'+html.escape(p['label'])+'</b><audio controls preload="none" src="audio/'+path.name+'"></audio></div>')
        body.append('<section><h2>'+html.escape(card['source_id'])+' · '+str(card['target_angle'])+'°</h2><p>'+html.escape(card['transcript'])+'</p><div class="grid">'+''.join(parts)+'</div></section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G17辅助试听</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#182234}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.panel{padding:12px;background:white;border-radius:10px}audio{display:block;width:100%;margin-top:9px}section{margin:28px 0;padding:18px;border:1px solid #d8dfe5;border-radius:12px}h2{font-size:19px}</style><h1>G17 新说话人辅助试听</h1><p>4位新说话人各首句；OOFFICE、0dB、点源相对−80°，目标−40/20/70/−40°。样本规则在G17原协议中固定。八个系统均用首个登记种子20261112作试听，科学报告使用全部三种子。</p><p>标签和参考文本可见，是研究者辅助试听，非盲听。没有假定用户已听过或给出人声细节结论。全页共用播放缩放 '+f'{gain:.6f}'+'，保留模型间音量差异，避免播放削波。重点对照词尾、辅音及听感，ASR不替代实际判断。</p>'+''.join(body)+'</html>'
    (folder/'index.html').write_text(page,encoding='utf-8');dump(folder/'manifest.json',{'created_utc':now(),'cards':cards,'common_page_gain':gain,'players':36,'blind':False,'human_feedback_received':False},True)
    assert page.count('<audio ')==36 and all((folder/p['review_relative_path']).exists() for c in cards for p in c['panels'])
    report=ROOT/'docs/G17_新语音辅助试听入口说明.md';report.write_text('# G17 新语音辅助试听入口说明\n\n入口：[本地对照页](../artifacts/g17_fresh_confirmation_v1/review/index.html)。四说话人各首句，OOFFICE点源−80°、目标−40/20/70/−40°，规则来自取数前G17计划。36个音频：参考、带噪、原模型、固定/反向/刚性三直接与三MVDR。只为试听固定首种子，数值结论仍保留全部三种子，不按成绩选择试听赢家。\n\n标签/文本可见，非盲听；尚无用户对这些新语音的听感或逐词反馈。全页共同缩放'+f'{gain:.6f}'+'，全语句16k单声道FLOAT，无裁剪，保存播放副本SHA并回读逐样本检查；不覆盖原输出。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('review',report,[folder/'manifest.json',folder/'index.html',ROOT/'scripts/g17_review.py']);print('Saved4cards36players,nohumanverdictinferred')


if __name__=='__main__':main()
