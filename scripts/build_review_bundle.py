#!/usr/bin/env python3
"""Create an offline, all-case listening aid and plots; assign no human verdict."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import html
import json
import os
from pathlib import Path
import random
from urllib.parse import quote


def rows(path):
    return [json.loads(t) for t in Path(path).read_text().splitlines() if t.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_url(path, base):
    return quote(os.path.relpath(Path(path).resolve(),base.resolve()).replace(os.sep,"/"),safe="/.-_")


def plot_case(case, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    from scipy.signal import stft
    signals=[]
    for role in ("reference","noisy","enhanced"):
        data,sr=sf.read(case[role+"_path"],dtype="float64",always_2d=True)
        if sr!=16000:
            raise ValueError("Plotting must not silently resample")
        signals.append(data[:,4] if data.shape[1]==6 else data[:,0])
    if len({len(s) for s in signals})!=1:
        raise ValueError("Plotting must not silently align or crop")
    spectra=[stft(s,fs=16000,nperseg=256,noverlap=192,boundary="zeros") for s in signals]
    anchor=max(float(np.abs(spectra[0][2]).max()),1e-12)
    ymax=max(max(float(np.abs(s).max()) for s in signals),1e-6)*1.05
    figure,axes=plt.subplots(3,2,figsize=(12,8),layout="constrained")
    image=None
    for index,(role,signal,spectrum) in enumerate(zip(("Reference","Noisy mic 5","McNet"),signals,spectra)):
        axes[index,0].plot(np.arange(len(signal))/16000,signal,linewidth=.55)
        axes[index,0].set(xlim=(0,len(signal)/16000),ylim=(-ymax,ymax),ylabel=role)
        f,t,z=spectrum
        image=axes[index,1].pcolormesh(t,f/1000,20*np.log10(np.maximum(np.abs(z),1e-12)/anchor),
                                     shading="auto",vmin=-70,vmax=0,cmap="magma")
        axes[index,1].set(xlim=(0,len(signal)/16000),ylim=(0,8),ylabel="kHz")
    axes[-1,0].set_xlabel("Time (s); common physical waveform scale")
    axes[-1,1].set_xlabel("Time (s); common reference-based spectral scale")
    figure.colorbar(image,ax=axes[:,1],label="dB relative to reference maximum (display only)")
    figure.suptitle(case["sample_id"]+"\nNo alignment, gain normalization or missing-word verdict")
    figure.savefig(destination,dpi=130)
    plt.close(figure)


def main():
    import numpy as np
    import soundfile as sf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--asr",type=Path,required=True)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to replace an existing review bundle")
    cases=rows(args.cases)
    asr={r["sample_id"]:r for r in rows(args.asr)}
    scores={r["sample_id"]:r for r in rows(args.scores)}
    ids={r["sample_id"] for r in cases}
    if len(ids)!=len(cases) or ids!=set(asr) or ids!=set(scores):
        raise ValueError("Case/ASR/metric IDs or denominators do not match")
    if any(c["status"]!="ok" or asr[c["sample_id"]]["status"]!="ok" or scores[c["sample_id"]]["status"]!="ok" for c in cases):
        raise ValueError("Resolve partial failures before making a complete-stage bundle")
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"plots").mkdir()
    (args.output_dir/"audio").mkdir()
    ranked=sorted(cases,key=lambda c:(not asr[c["sample_id"]]["asr_regression_review_candidate"],
                                      -asr[c["sample_id"]]["delta_deletions_enhanced_minus_noisy"],c["sample_id"]))
    candidate_ids=[c["sample_id"] for c in ranked if asr[c["sample_id"]]["asr_regression_review_candidate"]]
    controls=[c["sample_id"] for c in ranked if c["sample_id"] not in candidate_ids][:2]
    random_ids=random.Random(20261111).sample(sorted(ids),min(3,len(ids)))
    plot_ids=list(dict.fromkeys(candidate_ids+controls+random_ids))
    plot_ids=plot_ids[:16]
    cards=[]
    review_rows=[]
    all_metrics=[]
    playback=[]
    for index,c in enumerate(ranked):
        sid=c["sample_id"]
        a=asr[sid]
        s=scores[sid]
        is_candidate=a["asr_regression_review_candidate"]
        labels=[]
        for role,label in (("reference","干净参考"),("noisy","带噪参考麦"),("enhanced","McNet输出")):
            entry=a["asr"][role]
            score=entry["score"]
            audio_path=Path(c[role+"_path"])
            if role=="noisy":
                data,sr=sf.read(audio_path,dtype="float32",always_2d=True)
                if sr!=16000 or data.shape[1]!=6:
                    raise ValueError("Listening copy requires the original six-channel 16k input")
                selected=data[:,4]
                audio_path=args.output_dir/"audio"/(hashlib.sha256(sid.encode()).hexdigest()[:16]+"_mic5.wav")
                sf.write(audio_path,selected,sr,subtype="FLOAT")
                restored,rsr=sf.read(audio_path,dtype="float32")
                if rsr!=sr or not np.array_equal(selected,restored):
                    raise ValueError("Listening copy changed reference-microphone samples")
                playback.append({"sample_id":sid,"source_path":c["noisy_path"],
                                 "source_sha256":sha(c["noisy_path"]),
                                 "selected_channel_zero_based":4,"path":str(audio_path.resolve()),
                                 "sha256":sha(audio_path),"format":"WAV/FLOAT",
                                 "samples_equal_to_source_channel":True,
                                 "resampling_or_gain_or_alignment":False})
            labels.append(f'<div><b>{label}</b> WER={score["wer"]:.1%}；S/D/I={score["S"]}/{score["D"]}/{score["I"]}'
                          f'<p>{html.escape(entry["hypothesis_raw"])}</p>'
                          f'<audio controls preload="none" src="{local_url(audio_path,args.output_dir)}"></audio></div>')
        illustration=""
        if sid in plot_ids:
            filename=f"{index:03d}_{hashlib.sha256(sid.encode()).hexdigest()[:10]}.png"
            plot_case(c,args.output_dir/"plots"/filename)
            illustration=f'<img loading="lazy" src="plots/{filename}" alt="同步波形和声谱图；不是吞字判定">'
        deltas={key:s["metrics"]["enhanced"][key]-s["metrics"]["noisy"][key]
                if s["metrics"]["enhanced"][key] is not None and s["metrics"]["noisy"][key] is not None else None
                for key in ("si_sdr_db","stoi","pesq_wb")}
        row={"sample_id":sid,"source_id":c["source_id"],"speaker_id":c["speaker_id"],
             "noise_id":c["noise_id"],"snr_db":c["snr_db"],
             "asr_review_candidate":is_candidate,"delta_wer":a["delta_wer_enhanced_minus_noisy"],
             "delta_ASR_deletions":a["delta_deletions_enhanced_minus_noisy"],
             "clean_ASR_has_errors":a["clean_reference_asr_has_errors"],
             **deltas,"listened":False,"human_verdict":"not_assessed",
             "suspect_word_and_time":"","alignment_scale_checked":"","review_notes":""}
        review_rows.append(row)
        all_metrics.append(row)
        desc="优先核查：ASR发生退步" if is_candidate else "普通样本：保留于全部分母"
        cards.append(f'<details class="case" {"open" if is_candidate else ""}><summary>{html.escape(sid)} — {desc}</summary>'
                     f'<p>原文：{html.escape(c["transcript"])}</p><p>说话人 {html.escape(c["speaker_id"])}；'
                     f'{html.escape(c["noise_id"])}；{c["snr_db"]} dB；人工判定：未核查。</p>'
                     f'<p>ΔSI-SDR={deltas["si_sdr_db"]:.3f}dB；ΔSTOI={deltas["stoi"]:.4f}；'
                     f'ΔPESQ={deltas["pesq_wb"] if deltas["pesq_wb"] is not None else "未取得"}</p>'
                     f'<div class="triple">{"".join(labels)}</div>{illustration}</details>')
    with (args.output_dir/"review_all_cases.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(review_rows[0]))
        writer.writeheader();writer.writerows(review_rows)
    with (args.output_dir/"paired_metrics.csv").open("x",newline="") as stream:
        fields=["sample_id","source_id","speaker_id","noise_id","snr_db","si_sdr_db","stoi","pesq_wb","delta_wer","delta_ASR_deletions"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(all_metrics)
    grouped={}
    for label,key in (("noise_snr",lambda c:f'{c["noise_id"]}|{c["snr_db"]}dB'),("speaker",lambda c:c["speaker_id"])):
        groups=defaultdict(list)
        for c in cases:groups[key(c)].append(c["sample_id"])
        grouped[label]={}
        for name,members in sorted(groups.items()):
            item={"n":len(members)}
            for metric in ("si_sdr_db","stoi","pesq_wb"):
                for role in ("noisy","enhanced"):
                    values=[scores[sid]["metrics"][role][metric] for sid in members if scores[sid]["metrics"][role][metric] is not None]
                    item[role+"_"+metric]={"n":len(values),"mean":float(np.mean(values)) if values else None}
            for role in ("reference","noisy","enhanced"):
                values=[asr[sid]["asr"][role]["score"] for sid in members]
                n=sum(x["N"] for x in values)
                item[role+"_asr"]={"N":n,**{k:sum(x[k] for x in values) for k in ("S","D","I","errors")}}
                item[role+"_asr"]["wer"]=item[role+"_asr"]["errors"]/n
            grouped[label][name]=item
    summary={"created_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"cases":len(cases),
             "unique_sources":len({c["source_id"] for c in cases}),"unique_speakers":len({c["speaker_id"] for c in cases}),
             "asr_regression_candidates":candidate_ids,
             "increased_ASR_deletion_candidates":[sid for sid in sorted(ids) if asr[sid]["delta_deletions_enhanced_minus_noisy"]>0],
             "plots":plot_ids,"plot_policy":"all ASR regressions + first two nonregressions + seeded random three; max16; all audio cases retained",
             "human_judgements_completed":0,"confirmed_acoustic_deletions":None,"grouped":grouped,
             "listening_copy_policy":"noisy input selects zero-based channel4; float32 samples exactly preserved; reference/enhanced link original mono files",
             "playback_manifest":"playback_manifest.json",
             "evidence_sha256":{"cases":sha(args.cases),"asr":sha(args.asr),"scores":sha(args.scores)},
             "training_performed":False}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (args.output_dir/"playback_manifest.json").write_text(json.dumps(playback,ensure_ascii=False,indent=2)+"\n")
    header='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G1 首轮诊断核查</title>
<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#18232b}h1{font-size:28px}.notice{background:#fff4d6;padding:18px;border-radius:8px}.case{background:white;border:1px solid #d7dce1;border-radius:8px;margin:18px 0;padding:16px}summary{cursor:pointer;font-weight:650}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}audio{width:100%}img{width:100%;max-width:1200px}input{padding:10px;width:70%}@media(max-width:800px){.triple{grid-template-columns:1fr}}</style>
<h1>G1 首轮诊断：全部样本试听与证据</h1>
<p class="notice">简化自由场仿真，不是官方CHiME测试集。ASR退步和幅度变化都不能直接证明吞字。全部人工判定仍为“未核查”；请对照原文、三路声音和谱图，记录具体字词/时间，再判断原因。没有逐句音量归一化；勿以“更响”当作保真证据。</p>
<p>带噪试听只播放第5麦的单通道副本，已逐采样核对与原输入一致；其余两路链接原始单通道文件。请固定系统与播放器音量，不要使用自动响度均衡。CSV需另存后填写，本页面不会保存人工意见。</p>
'''
    header+=f'<p>{len(cases)}个混合，{len({c["source_id"] for c in cases})}条源语句、{len({c["speaker_id"] for c in cases})}位说话人；ASR退步候选{len(candidate_ids)}个。<a href="review_all_cases.csv">下载全部核查表</a>。本页仅本地文件，不上传音频。</p>'
    header+='<p><input id="filter" placeholder="按样本、说话人或文本查找" oninput="document.querySelectorAll(\'.case\').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))"></p>'
    (args.output_dir/"index.html").write_text(header+"\n".join(cards)+"</html>",encoding="utf-8")
    print(json.dumps({"cases":len(cases),"asr_review_candidates":len(candidate_ids),"plots":len(plot_ids),"index":str((args.output_dir/"index.html").resolve()),"human_verdict":"not_assessed"}))


if __name__=="__main__":main()
