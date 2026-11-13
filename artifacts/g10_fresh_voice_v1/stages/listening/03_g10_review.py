#!/usr/bin/env python3
"""Four rule-selected optional listening cards; no human labels inferred."""
import hashlib,html,json
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import soundfile as sf
from g10_common import ROOT,OUT,BASE,now,sha,read,dump,archive


def main():
    assert json.loads((OUT/'completion_audit.json').read_text())['status']=='ok' and (OUT/'stages/completion/archive_manifest.json').exists()
    plan_path=ROOT/'configs/g10_review_plan.json';plan=json.loads(plan_path.read_text());sources=read(BASE/'sources/speech_manifest.jsonl');cases=read(BASE/'manifest.jsonl')
    systems=['noisy','mcnet','fixed_without_clean_identity_s20261112','rotated_without_clean_identity_s20261112','mcnet_mask_mvdr','fixed_without_clean_identity_s20261112_mask_mvdr','rotated_without_clean_identity_s20261112_mask_mvdr']
    names=['干净参考','带噪第5麦','原McNet','固定方向适配器','旋转适配器','原McNet掩码MVDR','固定适配器掩码MVDR','旋转适配器掩码MVDR'];selected=[]
    speakers=list(dict.fromkeys(r['speaker_id'] for r in sources));assert len(speakers)==4
    for sp,t in zip(speakers,(-40,20,70,-40)):
        source=next(r['source_id'] for r in sources if r['speaker_id']==sp)
        matches=[r for r in cases if r['source_id']==source and r['noise_id']=='OOFFICE' and r['condition']=='point' and r['relative_angle']==-80 and r['target_azimuth']==t];assert len(matches)==1;selected+=matches
    idx={(r['system'],r['sample_id']):r for r in read(OUT/'evaluation/combined_signals.jsonl')};asrs={engine:{(r['system'],r['sample_id']):r for r in read(OUT/'evaluation'/engine/'asr_scores.jsonl')} for engine in ('Whisper','CTC')}
    waves={};originals={}
    for r in selected:
        sid=r['sample_id'];paths=[r['reference_path']]+[idx[(s,sid)]['output_path'] for s in systems]
        for role,path in enumerate(paths):
            a,sr=sf.read(path,dtype='float32');assert sr==16000 and a.ndim==1 and np.isfinite(a).all();waves[(sid,role)]=a;originals[(sid,role)]=path
    peak=max(float(np.abs(a).max()) for a in waves.values());gain=np.float32(.95/peak if peak>1 else 1.);folder=OUT/'review';folder.mkdir(exist_ok=False);cards=[];exports=[]
    for r in selected:
        sid=r['sample_id'];players=[]
        for role,label in enumerate(names):
            path=folder/'audio'/f'r{role}'/(sid+'.wav');path.parent.mkdir(parents=True,exist_ok=True);a=waves[(sid,role)]*gain
            with path.open('xb') as f:sf.write(f,a,16000,format='WAV',subtype='FLOAT')
            assert np.array_equal(sf.read(path,dtype='float32')[0],a)
            texts=[]
            for engine,ai in asrs.items():
                record=ai[(systems[role-1] if role else 'noisy',sid)];h=record['hypothesis'] if role else record['reference_hypothesis'];texts.append(html.escape(engine+': '+h))
            players.append('<div class="player"><strong>'+html.escape(label)+'</strong><audio controls preload="none" src="'+path.relative_to(folder).as_posix()+'"></audio><details><summary>展开识别文本（非人工判断）</summary>'+'<br>'.join(texts)+'</details></div>')
            exports.append({'sample_id':sid,'role':label,'source_path':originals[(sid,role)],'source_sha256':sha(originals[(sid,role)]),'export':str(path.relative_to(folder)),'export_sha256':sha(path),'frames':len(a)})
        cards.append('<section><h2>'+html.escape(r['speaker_id']+' / '+sid+' / 目标'+str(r['target_azimuth'])+'°')+'</h2><p>OOFFICE，0dB，点噪声相对−80°。</p><p>参考文本：'+html.escape(r['transcript'])+'</p><div class="grid">'+''.join(players)+'</div></section>')
    markup='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G10 新说话人辅助试听</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;background:#f7f8fb;color:#172434}section{background:white;padding:20px;margin:24px 0;border:1px solid #d9dfe8;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}.player{padding:10px;border-top:2px solid #dce5ee}audio{display:block;width:100%;margin:10px 0}details{font-size:14px}</style><h1>G10 新说话人辅助试听</h1><p>4条固定规则选择的样例，32个全长音频；第一个训练种子，不挑最佳成绩。已显示方法名与参考文本，这是非盲辅助复核，不是正式感知实验。没有推定任何新的人声误删或人工结论。</p><p>同页统一衰减系数：'+str(float(gain))+'；原最大峰值：'+str(peak)+'。不逐方法调音量、不裁剪原片段。原始计分音频和结果保持不变。可先比较音频，再按需展开识别文本。</p>'+''.join(cards)+'</html>'
    (folder/'index.html').write_text(markup,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            if tag=='audio':self.paths.append(dict(attrs)['src'])
    parser=Links();parser.feed(markup);assert len(parser.paths)==32 and all(sf.info(folder/p).channels==1 and sf.info(folder/p).samplerate==16000 for p in parser.paths)
    data={'completed_utc':now(),'plan_sha256':sha(plan_path),'selection':selected,'common_page_gain':float(gain),'source_peak':peak,'exports':exports,'human_feedback_received':False,'blind_listening':False,'checks_32_valid_audio_links':True}
    dump(folder/'selection.json',data,True)
    report=ROOT/'docs/G10_辅助试听入口说明.md';report.write_text('# G10 辅助试听入口\n\n[打开本地试听页](../artifacts/g10_fresh_voice_v1/review/index.html)。四位新说话人各选第一条，固定OOFFICE/0dB/点噪声−80°，目标依次−40/20/70/−40°，8角色共32音频。规则在ASR启动前登记，但当时SE推理已经开始；不声称早于所有计算。没有按指标挑例，也没有新增人工标签。\n\n全长单通道16k FLOAT WAV，仅同页统一必要衰减，原始计分波形不改。页面显示方法和文本，ASR假设可折叠，属于非盲辅助复核。试听不是必须完成的额外实验规模，不以未收到反馈作为成功或失败标签。\n\n逐片原始/导出哈希、统一增益及规则见review/selection.json；32链接/声道/采样率及FLOAT回读检查通过。\n\nUTC：'+now()+'\n',encoding='utf-8')
    archive('listening',report,[plan_path,folder/'selection.json',Path(__file__)]);print(json.dumps({'cards':4,'audio_players':32,'status':'ok'}))


if __name__=='__main__':main()
