"""Compact scientific readback of accepted room comparisons and error types."""
from g22_common import ROOT,OUT,sha,dump,metadata,archive,now


def main():
    assert metadata(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    rec=OUT/'recognition';summary=metadata(rec/'summary.json')
    index={(r['room'],r['rt'],r['group'],r['method'],r['clean']):r for r in summary['by_rt']}
    selected=[]
    for (room,rt,group,method,clean),r in sorted(index.items()):
        if clean:continue
        selected.append({'room':room,'rt':rt,'group':group,'method':method,
                         **{e:{k:r[e][k] for k in ('N','S','D','I','errors','wer')} for e in ('Whisper','CTC')}})
    rows=[]
    for r in summary['by_system']:
        if not r['clean'] and r['system'].startswith(('rigid_without_','rotated_without_','fixed_without_')):
            rows.append(r)
    dump(rec/'readback_error_types.json',{'errors_all_noisy_groups':selected,'per_system_all_adapters':rows,
          'references':summary['references'],'summary_sha256':sha(rec/'summary.json')},True)
    report=ROOT/'docs/G22_科学结论与下一阶段问题.md'
    lines=['# G22 科学结论与下一阶段问题','','本阶段所有记录、缓存和配对控制已验收。以下结论限定于四位已用说话人各一句、两段旧噪声及两个模拟房间；没有新增训练或人工听感标签。','',
           '## 主要结果','','无反射的2748对输出完全一致。反射条件下，换房间没有给出统一的指标变化：较强反射时，适配器MVDR的SI-SDR可以略升，而两个识别器WER仍上升。','',
           '|正确刚性训练+MVDR|原房间SI-SDR|新房间SI-SDR|Whisper WER：原/新|CTC WER：原/新|','|---|---:|---:|---:|---:|']
    for rt in (.15,.35):
        a,b=index['reference',rt,'rigid','mvdr',False],index['new',rt,'rigid','mvdr',False]
        lines.append(f"|请求RT={rt}|{a['si_sdr_db']:.2f}|{b['si_sdr_db']:.2f}|{a['Whisper']['wer']:.2%} / {b['Whisper']['wer']:.2%}|{a['CTC']['wer']:.2%} / {b['CTC']['wer']:.2%}|")
    lines+=['','完整三种子的组内统计保留，不能把条件记录数当独立语句数。请求RT与实测T20不是同一量，不把这个对照解释为某一尺寸的独立因果效应。','',
            '## 较强反射、新房间的错误组成','','以下均为请求RT=.35，正确刚性训练组三种子汇总。MVDR与MWF共用相同前端和输入。','',
            '|输出|识别器|参考词数|替换S|删除D|插入I|WER|','|---|---|---:|---:|---:|---:|---:|']
    for method in ('mvdr','mwf'):
        r=index['new',.35,'rigid',method,False]
        for e in ('Whisper','CTC'):
            t=r[e];lines.append(f"|{method}|{e}|{t['N']}|{t['S']}|{t['D']}|{t['I']}|{t['wer']:.2%}|")
    lines+=['','S/D/I都是固定识别器的文本错误，不能直接当作声学吞字或听感标签。全部噪声条件组和全部适配器分系统数据另存 readback_error_types.json，未删异常样本或选最佳种子。','',
            '## 下一项应检验的问题','','在设计保护人声模块前，还需区分识别器的幅度敏感性与真正的相对时频结构变化。下一项可以对已保存输出施加预先固定的正标量增益，检查数字波形结构和SI-SDR保持时，两个ASR的S/D/I是否变化。它仅是评估器控制，不是新增强方法，也不能据此选一个测试集最优增益。','',
            '新控制应保留干净参考、带噪输入、正确刚性全部三种子的MVDR/MWF及真值诊断，预先固定条件和增益，不重新运行增强网络或训练。需说明：降低播放电平可以影响人的可听度；数字结构不变不等于固定音量下人耳可懂度不变。','',
            'TSOS来源与幅度约定的补查已独立保存，未将未核对的TSOS实现加入本阶段成绩。声学、识别和辅助试听仍需分别解释。','',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('scientific_readback',report,[rec/'readback_error_types.json',ROOT/'scripts/g22_readback.py',OUT/'FINAL_ACCEPTANCE_AUDIT.json'])
    print('\n'.join(lines[11:29]))


if __name__=='__main__':main()
