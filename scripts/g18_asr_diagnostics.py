#!/usr/bin/env python3
import json
from g18_common import ROOT,OUT,now,read,dump,archive
from mcse.asr_scoring import corpus_score


def main():
    assert json.loads((OUT/'FINAL_ACCEPTANCE_AUDIT.json').read_text())['status']=='ok';w=read(OUT/'evaluation/Whisper/asr_scores.jsonl');c=read(OUT/'evaluation/CTC/asr_scores.jsonl');wi={(r['system'],r['sample_id']):r for r in w};ci={(r['system'],r['sample_id']):r for r in c};pairs=[]
    for r in w:
        if r['requested_rt60']==.15 and r['noise_id']=='OOFFICE' and r['target_azimuth']==70 and r['system'].startswith('rotated_without_clean_identity_') and r['system'].endswith('_mask_mvdr'):
            other='rigid'+r['system'][len('rotated'):];b=wi[(other,r['sample_id'])];cr=ci[(r['system'],r['sample_id'])];cb=ci[(other,r['sample_id'])]
            pairs.append({'sample_id':r['sample_id'],'source_id':r['source_id'],'counter_system':r['system'],'rigid_system':other,'transcript':r['transcript'],
                'Whisper_counter':r['score'],'Whisper_rigid':b['score'],'CTC_counter':cr['score'],'CTC_rigid':cb['score'],
                'counter_hypothesis':r['hypothesis'],'rigid_hypothesis':b['hypothesis'],'Whisper_error_gap_counter_minus_rigid':r['score']['errors']-b['score']['errors']})
    assert len(pairs)==72
    summary={key:corpus_score(r[key] for r in pairs) for key in ('Whisper_counter','Whisper_rigid','CTC_counter','CTC_rigid')};ranked=sorted(pairs,key=lambda r:abs(r['Whisper_error_gap_counter_minus_rigid']),reverse=True)
    result={'status':'ok','created_utc':now(),'selection':'Posthoc targeted explanation of G18requestedRT.15/OOFFICE/70deg,all3counter/rigidMVDRseeds;not new confirmation','pairs':72,'aggregate':summary,'largest_eight_error_gaps':ranked[:8]}
    dump(OUT/'evaluation/posthoc_ASR_insertions.json',result,True)
    lines=['# G18 双识别器异常插入补查','','独立只读复核指出：请求RT0.15、OOFFICE、70°中两个MVDR前端的SI差小，而Whisper分差大。本次在看到结果后定位解释，属于事后分析，不是新确认或模型选择，不重跑识别、不改缓存/指标。保留全部72对（三种子×8句×3场型）。','','|后端/前端|N|S|D|I|WER|','|---|---:|---:|---:|---:|---:|']
    for name,r in summary.items():lines.append(f'|{name}|{r["N"]}|{r["S"]}|{r["D"]}|{r["I"]}|{r["wer"]:.1%}|')
    lines+=['','逐例最大分差、假设文本和完整编辑计数保存在机器记录。不把插入词自动解释为语言模型幻觉：也可能涉及背景声音或其他识别效应，未经人工/波形核查不能判因。不能据Whisper巨大分差宣称人声细节或吞字改善，必须连同CTC和信号指标阅读。',
        '', '后续成熟滤波对照继续固定全部前端和参数，不为本组异常调整模型或解码设置。','', 'UTC：'+now()]
    report=ROOT/'docs/G18_双识别器异常插入补查.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');archive('ASR_diagnostics',report,[OUT/'evaluation/posthoc_ASR_insertions.json',ROOT/'scripts/g18_asr_diagnostics.py']);print(json.dumps(summary))


if __name__=='__main__':main()
