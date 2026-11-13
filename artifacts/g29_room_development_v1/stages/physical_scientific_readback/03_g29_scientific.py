"""Measured room descriptor readback; no scores or changes to frozen data."""
import numpy as np
from collections import defaultdict
from g29_common import ROOT,OUT,sha,dump,now,meta,archive


def main():
    assert meta(OUT/'FINAL_ACCEPTANCE_AUDIT.json')['status']=='ok'
    groups=defaultdict(list)
    for r in meta(OUT/'RIR_manifest.json'):groups[r['room_id'],r['rt']].append(r)
    rows=[]
    for (room,request),rs in sorted(groups.items()):
        values=[r['measured_T20_extrapolated_RT60'] for r in rs if r['measured_T20_extrapolated_RT60'] is not None]
        rows.append({'room':room,'requested_RT':request,'RIRs':len(rs),
            'T20_extrapolated_min':min(values) if values else None,'T20_extrapolated_median':float(np.median(values)) if values else None,
            'T20_extrapolated_max':max(values) if values else None,'tail20ms_fraction_max':max(r['last20ms_energy_fraction'] for r in rs)})
    dump(OUT/'physical_readback.json',rows,True)
    report=ROOT/'docs/G29_实际衰减与共享数据范围说明.md'
    lines=['# G29 实际衰减与共享数据范围','','数据已验收，本表只读保存的RIR。请求RT与实测T20外推不是同一数值。每行包含两种目标方向下的语音、两点源与八个噪声位置，共22个RIR；不同位置并非独立房间样本。无反射条件不计算T20。', '',
        '|房间|请求RT秒|RIR数|T20外推最小/中位/最大秒|最大末20ms能量占比|','|---|---:|---:|---|---:|']
    for r in rows:
        text='无反射' if r['T20_extrapolated_median'] is None else ' / '.join(f"{r[k]:.4f}" for k in ('T20_extrapolated_min','T20_extrapolated_median','T20_extrapolated_max'))
        lines.append(f"|{r['room']}|{r['requested_RT']}|{r['RIRs']}|{text}|{r['tail20ms_fraction_max']:.3g}|")
    lines+=['','每段训练干语音1秒，加7680样本RIR全卷积，共23679样本。0.48秒RIR有有限尾长，不能视作无限精确声学仿真；源距1.2m、四个指定房间、固定六麦与八点源噪声也不代表真实阵列或普遍混响场景。', '',
        '训练576个条件实际使用191条源语句、四位旧训练角色说话人，不能把576当独立语句。开发108条件来自12条旧开发角色语句、两人；中心裁剪仅评信号质量/干净控制，不做整句WER。本轮开发不等于未见说话人的最终确认。', '',
        'G30首轮使用单种子、相同抽样与1500步登记做有限开发比较。系数仅从训练输入梯度校准，不能据此保证语音内容或ASR改善。后续进入新完整语句确认必须另排除所有历史训练/开发/反复查看的来源，而非把本轮未抽中的旧训练语句当作未见说话人。', '',now()]
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    archive('physical_scientific_readback',report,[OUT/'physical_readback.json',OUT/'FINAL_ACCEPTANCE_AUDIT.json',ROOT/'scripts/g29_scientific.py'])
    print({'RIR_groups':len(rows),'tail_fraction_max':max(r['tail20ms_fraction_max'] for r in rows)})


if __name__=='__main__':main()
