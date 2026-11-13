#!/usr/bin/env python3
"""Freeze and build reference-matched spatial-noise diagnostic inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mcse.simulation import render_plane_wave,geometric_delays,speech_activity_mask,mix_at_snr,snr_db
from mcse.spatial_noise import noise_fields,spatial_coherence,highpass40


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--plan",type=Path,default=ROOT/"configs/g2_spatial_probe_plan.json")
    parser.add_argument("--source-manifest",type=Path,help="Optional preselected new complete utterances; paths relative to this file")
    args=parser.parse_args()
    if args.output_dir.exists():parser.error("Refusing to overwrite diagnostic data")
    plan_path=args.plan.resolve()
    plan=read(plan_path)
    frozen=read(ROOT/"data/diagnostic/g1_v1/frozen_inputs.json")
    old=[json.loads(t) for t in (ROOT/"data/diagnostic/g1_v1/manifest.jsonl").read_text().splitlines()]
    geometry=read(ROOT/"configs/g1_geometry.json")
    sources=[]
    if args.source_manifest:
        sources=[json.loads(t) for t in args.source_manifest.read_text().splitlines() if t.strip()]
        for r in sources:r["path"]=str((args.source_manifest.resolve().parent/r["path"]).resolve())
    else:
        for speaker in sorted({r["speaker_id"] for r in frozen["sources"]}):
            sources.append(sorted([r for r in frozen["sources"] if r["speaker_id"]==speaker],key=lambda r:r["source_id"])[0])
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"noisy").mkdir();(args.output_dir/"reference").mkdir()
    write(args.output_dir/"frozen_plan.json",{"plan":plan,"plan_sha256":sha(plan_path),"sources":sources,
          "generation_script_sha256":sha(Path(__file__)),"spatial_noise_module_sha256":sha(ROOT/"src/mcse/spatial_noise.py"),
          "source_manifest_sha256":sha(args.source_manifest or ROOT/"data/sources/g1_v1/source_manifest.jsonl"),"geometry":geometry,
          "noise_sources":frozen["noises"],"role":"discovery_not_final_test","frozen_before_generation":True})
    noise_map={n["noise_id"]:n for n in frozen["noises"]}
    noise_audio={}
    for noise_id,entry in noise_map.items():
        if sha(entry["path"])!=entry["sha256"]:raise ValueError("Noise bytes changed")
        raw,_=sf.read(entry["path"],dtype="float64")
        noise_audio[noise_id]={"raw":raw,"highpass40_then_snr0":highpass40(raw)}
    spectral_kinds=plan.get("noise_spectrum_conditions",["raw"])
    spatial_kinds=plan.get("spatial_conditions",plan.get("noise_fields"))
    blocked={noise_id:[(r["noise_segment_start_sample"],r["noise_segment_start_sample"]+r["noise_segment_samples"])
                        for r in old if r["noise_id"]==noise_id] for noise_id in plan["noises"]}
    records=[];details=[]
    for index,source in enumerate(sources):
        if sha(source["path"])!=source["sha256"]:raise ValueError("Source bytes changed")
        mono,sr=sf.read(source["path"],dtype="float64")
        if sr!=16000:raise ValueError("No implicit resampling")
        speech=render_plane_wave(mono,geometric_delays(geometry["microphone_positions_m"],geometry["target_direction_unit"]))
        activity=speech_activity_mask(speech[:,4])
        conditions=[]
        for ni,noise_id in enumerate(plan["noises"]):
            source_noise=noise_map[noise_id]
            if sha(source_noise["path"])!=source_noise["sha256"]:raise ValueError("Noise bytes changed")
            previous=next((r for r in old if r["source_id"]==source["source_id"] and r["noise_id"]==noise_id),None)
            keyseed=int.from_bytes(hashlib.sha256((source["source_id"]+noise_id+str(plan["seed"])).encode()).digest()[:8],"little")
            if previous:
                start=previous["noise_segment_start_sample"]
            else:
                rng=np.random.default_rng(keyseed)
                for _ in range(10000):
                    start=int(rng.integers(4000,len(noise_audio[noise_id]["raw"])-len(mono)-4000))
                    if all(start+len(mono)+4000<=a or start-4000>=b for a,b in blocked[noise_id]):break
                else:raise ValueError("No nonoverlapping new-noise segment within the recording")
                blocked[noise_id].append((start,len(mono)+start))
            for spectral in spectral_kinds:
                segment=noise_audio[noise_id][spectral][start:start+len(mono)]
                for replica in plan.get("seed_replicas",[0]):
                    seed=plan["seed"]+index*100+ni+replica*10000
                    fields,field_detail=noise_fields(segment,geometry["microphone_positions_m"],geometry["target_direction_unit"],seed)
                    for kind in spatial_kinds:
                        mixture,scaled_noise,gain=mix_at_snr(speech,fields[kind],float(plan["snr_db"]),activity)
                        conditions.append((noise_id,spectral,kind,replica,mixture,gain,start,seed,spatial_coherence(scaled_noise)))
                    details.append({"source_id":source["source_id"],"noise_id":noise_id,"spectrum":spectral,"replica":replica,**field_detail})
        scale=min(1.0,.95/max(float(np.abs(speech).max()),*(float(np.abs(c[4]).max()) for c in conditions)))
        refrel="reference/"+source["source_id"]+".wav"
        sf.write(args.output_dir/refrel,speech[:,4]*scale,16000,subtype="PCM_24")
        reference,_=sf.read(args.output_dir/refrel)
        for noise_id,spectral,kind,replica,mixture,gain,start,seed,coherence in conditions:
            suffix=("__"+spectral+"__seed"+str(replica)) if "noise_spectrum_conditions" in plan or "seed_replicas" in plan else ""
            sid=source["source_id"]+"__"+noise_id+"__"+kind+suffix
            relative="noisy/"+sid+".wav"
            sf.write(args.output_dir/relative,mixture*scale,16000,subtype="PCM_24")
            actual,_=sf.read(args.output_dir/relative,always_2d=True)
            records.append({"sample_id":sid,"source_id":source["source_id"],"speaker_id":source["speaker_id"],
                "split":plan.get("split","discovery"),"noisy_path":relative,"reference_path":refrel,"transcript":source["transcript"],
                "noise_id":noise_id,"snr_db":plan["snr_db"],"sample_rate":16000,"ref_channel":4,
                "noise_field":kind,"noise_spectrum":spectral,"seed_replica":replica,"mix_seed":seed,"noise_segment_start_sample":start,"noise_gain":gain,"common_scale":scale,
                "coherence":coherence,"achieved_snr_db":snr_db(reference,actual[:,4]-reference,activity),
                "noisy_sha256":sha(args.output_dir/relative),"reference_sha256":sha(args.output_dir/refrel)})
        print(json.dumps({"source":index+1,"total":len(sources),"source_id":source["source_id"]}),flush=True)
    group_keys={(r["source_id"],r["noise_id"],r["noise_spectrum"]) for r in records}
    for source_id,noise_id,spectral in group_keys:
        group=[r for r in records if r["source_id"]==source_id and r["noise_id"]==noise_id and r["noise_spectrum"]==spectral]
        channels=[sf.read(args.output_dir/r["noisy_path"],always_2d=True)[0][:,4] for r in group]
        if len(channels)!=len(spatial_kinds)*len(plan.get("seed_replicas",[0])) or any(not np.array_equal(channels[0],c) for c in channels[1:]):
            raise ValueError("Reference mixtures are not bit-identical across spatial conditions/seeds")
    if len(records)!=plan["expected_mixtures"]:raise ValueError("Unexpected denominator")
    with (args.output_dir/"manifest.jsonl").open("x",encoding="utf-8") as stream:
        for r in records:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    write(args.output_dir/"field_details.json",details)
    write(args.output_dir/"summary.json",{"status":"complete","sources":len(sources),"mixtures":len(records),
          "reference_matched_groups":len(group_keys),"all_spatial_noisy_references_bit_identical":True,
          "max_snr_error_db":max(abs(r["achieved_snr_db"]-plan["snr_db"]) for r in records),
          "manifest_sha256":sha(args.output_dir/"manifest.jsonl"),"training":False})


if __name__=="__main__":main()
