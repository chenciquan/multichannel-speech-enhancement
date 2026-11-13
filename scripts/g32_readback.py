"""Scientific contrasts and fixed researcher listening cards after acceptance."""
import html
import os
from pathlib import Path
from urllib.parse import quote
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from g32_common import ROOT,OUT,sha,read,dump,now,meta,archive


def main():
    assert meta(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    assert meta(ROOT/'artifacts/g31_development_signals_v1/primary_decision.json')['extension_gate_passed'] is False
    s=meta(OUT/'recognition/summary.json');by={(r['source_system'],r['method'],r['clean']):r for r in s['overall']}
    comparisons=[]
    for method in ('direct','mvdr','mwf'):
        a=by['cirm_shape',method,False]
        for comparator in ('cirm','cirm_mask'):
            b=by[comparator,method,False]
            row={'method':method,'comparator':comparator,
                'SI_delta':a['si_sdr_db']-b['si_sdr_db'],'PESQ_delta':a['pesq_wb']-b['pesq_wb'],'STOI_delta':a['stoi']-b['stoi']}
            for engine in ('Whisper','CTC'):
                for regime in ('raw','level_control'):row[engine+'_'+regime+'_WER_pp']=100*(a[engine][regime]['wer']-b[engine][regime]['wer'])
            row['all_four_WER_strictly_lower']=all(row[e+'_'+reg+'_WER_pp']<0 for e in ('Whisper','CTC') for reg in ('raw','level_control'))
            comparisons.append(row)
    dump(OUT/'recognition/scientific_contrasts.json',comparisons,True)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True)
    for i,engine in enumerate(('Whisper','CTC')):
        for j,regime in enumerate(('raw','level_control')):
            ax=axes[i,j]
            for k,comparator in enumerate(('cirm','cirm_mask')):
                values=[next(r for r in comparisons if r['method']==m and r['comparator']==comparator)[engine+'_'+regime+'_WER_pp'] for m in ('direct','mvdr','mwf')]
                ax.bar(np.arange(3)+(-.17 if k==0 else .17),values,width=.32,label='Shape minus '+('cIRM' if k==0 else 'simple mask'),color=('#2b6b91','#c27a31')[k])
            ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(3),['Direct','MVDR','MWF']);ax.set_ylabel('WER delta (percentage points)')
            ax.set_title(engine+' / '+regime);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Supplementary full-utterance development: lower bars mean lower WER\n4 old utterances / 16 original word tokens / 2 speakers / 1 training seed',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9))
    for ext in ('png','pdf'):fig.savefig(OUT/f'ASR_contrasts.{ext}',dpi=180)
    plt.close(fig)
    folder=OUT/'review';folder.mkdir(exist_ok=False);rows=read(OUT/'signals.jsonl');cards=[];links=[]
    sources=meta(OUT/'data_plan.json')['sources']
    def player(label,path):
        path=Path(path);assert path.is_file() and sf.info(path).channels==1
        links.append({'label':label,'path':str(path),'sha256':sha(path)})
        href=quote(os.path.relpath(path,folder).replace(os.sep,'/'),safe='/')
        return '<div class="clip"><b>'+html.escape(label)+'</b><audio controls preload="none" src="'+href+'"></audio></div>'
    for source in sources:
        sid=source['source_id'];candidates=[r for r in rows if r['source_id']==sid and r['requested_rt60']==.28 and r['condition']=='point']
        assert len(candidates)==13
        selected={(r['source_system'],r['method']):r for r in candidates};reference=selected['noisy','direct']
        clips=[player('全混响参考（无背景噪声）',reference['reference_path']),player('带噪第5麦',reference['output_path'])]
        for system,label in (('mcnet','原McNet'),('cirm','原重建适配'),('cirm_mask','简单掩码辅助'),('cirm_shape','统计辅助')):clips.append(player(label+' + MVDR',selected[system,'mvdr']['output_path']))
        other=[]
        for method in ('direct','mwf'):
            for system in ('mcnet','cirm','cirm_mask','cirm_shape'):other.append(player(system+' / '+method,selected[system,method]['output_path']))
        cards.append('<section><h2>'+html.escape(sid)+'</h2><p class="text">'+html.escape(source['transcript'])+'</p><p>固定请求RT=.28、点源；未按ASR分数挑选。</p><div class="grid">'+''.join(clips)+'</div><details><summary>其余固定直接输出与MWF对照</summary><div class="grid">'+''.join(other)+'</div></details></section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G32完整开发语句对照</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;background:#f5f6fa;color:#202938}section{background:white;padding:22px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.clip{background:#f0f3f7;padding:12px;border-radius:8px}audio{width:100%;display:block;margin-top:8px}.text{font-weight:bold}summary{cursor:pointer;padding:12px}</style><h1>G32 完整开发语句对照试听</h1><p>带原转写提示、可见系统名称的研究者辅助试听，不是盲听实验；尚未收到本页人工反馈。播放器均为原电平单通道，请比较具体词语与细节，音量变小本身不等于词被删除。本页不自动记录评价，不产生模型优劣或吞字标签。</p><p>四条旧开发语句，冻结全部1500步权重。G31主门槛仍未通过；这里不替代正式内容保真或未见说话人验证。</p>'+''.join(cards)+'</html>'
    with (folder/'index.html').open('x',encoding='utf-8') as f:f.write(page)
    assert len(links)==56;dump(folder/'audio_links.json',links,True)
    report=ROOT/'docs/G32_科学读回与完整语句试听入口.md'
    lines=['# G32 科学读回与完整语句试听','','原干声已经有明显的识别器文本错误：Whisper两种电平都是6/16错词（37.5%），CTC都是3/16（18.75%），原干声错误均为替换，不存在此处对齐判定的删除/插入。Whisper把OTTO写成AUTO、DEAR写成DEER，两个后端都出现UNFAVOURABLE/UNFAVORABLE的文本差别；这些不能直接解释成声学信息被增强器删除。固定规范化及原转写保持不变。', '',
        '## 相对成熟强对照：带噪24条件', '',
        '|输出|统计辅助−对照|ΔSI|ΔPESQ|ΔSTOI|Whisper raw/电平 WER差百分点|CTC raw/电平 WER差百分点|四条件WER均严格下降|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in comparisons:lines.append(f"|{r['method']}|{r['comparator']}|{r['SI_delta']:.4f}|{r['PESQ_delta']:.4f}|{r['STOI_delta']:.4f}|{r['Whisper_raw_WER_pp']:.3f} / {r['Whisper_level_control_WER_pp']:.3f}|{r['CTC_raw_WER_pp']:.3f} / {r['CTC_level_control_WER_pp']:.3f}|{r['all_four_WER_strictly_lower']}|")
    lines+=['', '负WER差表示统计辅助较低。直接输出在这批极小样本上有两个后端/两条件同向的WER收益；主MVDR对强掩码对照则随后端/电平而变，不能从一个有利单元推断总体内容保真。直接输出的信号质量也未一致领先。G31预登记主比较不因这些补充结果而改判。', '',
        '## 固定试听页', '', '[打开四条完整开发语句对照](../artifacts/g32_complete_dev_v1/review/index.html)。每条固定请求RT=.28/点源，默认展示混响参考、带噪第5麦、四个系统的MVDR；其余直接/MWF可展开，均原电平单声道，共56个经存在性、声道及哈希检查的链接。选择不依据本轮ASR分数。不是盲听，也没有自动新增人工标签。', '',
        '## 后续问题', '',
        '目前的小参数方案只有同位置LoRA及不同目标的比较。若要说明“轻量”在本机的价值，还缺少相同数据上全参数微调的成熟对照。下一项可先进行不更新参数的训练资源预检：相同六条训练输入、原cIRM目标，比较完整微调和标准LoRA的可微性、实际显存与耗时，不读取开发或测试音频。仅验证是否能在8GB内运行，再决定是否登记有限、透明的全参数学习率/预算对照；不能把一个未调好的全参数跑分当作其最佳表现。', '',
        '不继续扩大未过G31门槛的统计辅助训练。所有目标、适配方式与滤波组件均有先例；本项目当前价值在受控证据、强对照与质量/识别权衡，创新性、声学吞字及真实场景泛化尚未证实。', '',
        '![完整语句识别差值](../artifacts/g32_complete_dev_v1/ASR_contrasts.png)', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[OUT/'recognition/scientific_contrasts.json',OUT/'ASR_contrasts.png',OUT/'ASR_contrasts.pdf',folder/'index.html',folder/'audio_links.json',ROOT/'scripts/g32_readback.py'])
    print({'comparisons':len(comparisons),'listening_cards':4,'audio_links':56,'G31gate_unchanged':True})


if __name__=='__main__':main()
