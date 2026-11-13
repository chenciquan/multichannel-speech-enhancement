#!/usr/bin/env python3
"""Fetch only a bounded speech archive prefix and two ZIP members.

Selection is fixed before any enhancement/ASR output. No full corpus download,
archive extraction paths, speech cropping, model training or invented transcripts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class BoundedStream:
    def __init__(self, raw, limit):
        self.raw, self.limit, self.count = raw, limit, 0

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded read is forbidden")
        if self.count + size > self.limit:
            raise RuntimeError("Speech archive prefix reached the predeclared byte cap")
        block = self.raw.read(size)
        self.count += len(block)
        return block


class RangeReader(io.RawIOBase):
    """Seekable HTTP object. Each request is bounded and must be HTTP 206."""
    def __init__(self, session, url, max_bytes=32 * 1024**2):
        self.session, self.url, self.position = session, url, 0
        r = session.head(url, allow_redirects=True, timeout=40)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.received = 0
        self.max_bytes = max_bytes
        self.cache_start, self.cache = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= pos <= self.size:
            raise ValueError("HTTP seek outside object")
        self.position = pos
        return pos

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        start = self.position
        if self.cache_start <= start and start + size <= self.cache_start + len(self.cache):
            result = self.cache[start-self.cache_start:start-self.cache_start+size]
        else:
            count = min(max(size, 65536), self.size-start)
            if self.received + count > self.max_bytes:
                raise RuntimeError("ZIP member download exceeded byte cap")
            with self.session.get(self.url, headers={"Range": f"bytes={start}-{start+count-1}"},
                                  stream=True, timeout=90) as response:
                if response.status_code != 206:
                    raise RuntimeError("Server did not honor Range; refusing a full ZIP download")
                expected = f"bytes {start}-{start+count-1}/{self.size}"
                if response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Unexpected Content-Range")
                data = response.raw.read(count+1)
                if len(data) != count:
                    raise RuntimeError("Incomplete or oversized range response")
            self.received += len(data)
            self.cache_start, self.cache = start, data
            result = data[:size]
        self.position += len(result)
        return result


def fetch_speech(session, output, plan):
    config = plan["speech"]
    folder = output / "speech"
    folder.mkdir()
    transcripts, choices, complete = {}, {}, []
    response = session.get(config["url"], stream=True, timeout=90)
    response.raise_for_status()
    bounded = BoundedStream(response.raw, config["max_stream_bytes"])
    scanned, stop = 0, False
    try:
        with tarfile.open(fileobj=bounded, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                leaf = PurePosixPath(member.name).name
                if leaf.endswith(".trans.txt"):
                    text = archive.extractfile(member).read().decode("utf-8")
                    for line in text.splitlines():
                        key, value = line.split(" ", 1)
                        transcripts[key] = value
                elif leaf.endswith(".flac") and re.fullmatch(r"\d+-\d+-\d+\.flac", leaf):
                    source_id = leaf[:-5]
                    speaker = source_id.split("-")[0]
                    selected = choices.setdefault(speaker, [])
                    if len(selected) >= 4:
                        continue
                    data = archive.extractfile(member).read()
                    info = sf.info(io.BytesIO(data))
                    scanned += 1
                    if info.channels == 1 and info.samplerate == 16000 and 1.0 <= info.duration <= 3.8:
                        path = folder / leaf
                        path.write_bytes(data)
                        selected.append({"source_id": source_id, "speaker_id": speaker,
                                         "path": "speech/"+leaf, "sample_rate": 16000,
                                         "duration_seconds": info.duration, "samples": info.frames,
                                         "sha256": sha256(data), "archive_member": member.name,
                                         "source_url": config["url"], "dataset": "LibriSpeech dev-clean",
                                         "license": config["license"]})
                        print(f"SPEECH {speaker}: {len(selected)}/4", flush=True)
                    if len(selected) == 4 and speaker not in complete:
                        complete.append(speaker)
                if len(complete) >= 6:
                    final = [r for spk in complete[:6] for r in choices[spk]]
                    if all(r["source_id"] in transcripts for r in final):
                        stop = True
                        break
    finally:
        response.close()
    if not stop:
        raise RuntimeError(f"Insufficient complete eligible speakers; complete={complete}")
    rows = [dict(r, transcript=transcripts[r["source_id"]]) for r in final]
    (output / "source_manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    return {"archive_url": config["url"], "compressed_bytes_streamed": bounded.count,
            "whole_archive_saved": False, "whole_archive_hash_verified": False,
            "saved_file_integrity": "per-file SHA256; TLS archive stream; full official archive MD5 not verified",
            "eligible_selection_policy": config["selection"], "scanned_audio_headers": scanned,
            "selected_sources": len(rows), "selected_speakers": complete[:6]}


def fetch_noises(session, output):
    folder = output / "noise"
    folder.mkdir()
    rows, reports = [], []
    for name, category in (("DWASHING", "washing_machine"), ("DKITCHEN", "kitchen_activity")):
        url = f"https://zenodo.org/records/1227121/files/{name}_16k.zip?download=1"
        remote = RangeReader(session, url)
        with zipfile.ZipFile(remote) as archive:
            members = [n for n in archive.namelist() if PurePosixPath(n).name.lower() == "ch01.wav"]
            if len(members) != 1:
                raise RuntimeError(f"Expected one ch01.wav in {name}: {members}")
            member = members[0]
            data = archive.read(member)  # ZipFile verifies the member CRC.
            crc = archive.getinfo(member).CRC
        info = sf.info(io.BytesIO(data))
        if info.channels != 1 or info.samplerate != 16000:
            raise RuntimeError(f"Unexpected DEMAND audio format: {info}")
        filename = f"{name}_ch01.wav"
        (folder / filename).write_bytes(data)
        rows.append({"noise_id": name+"_ch01", "path": "noise/"+filename,
                     "noise_type": category, "sample_rate": 16000,
                     "duration_seconds": info.duration, "sha256": sha256(data),
                     "archive_member": member, "source_url": url,
                     "dataset": "DEMAND v1.0", "license": "CC-BY-SA-3.0"})
        reports.append({"archive": url, "archive_size": remote.size,
                        "bytes_downloaded": remote.received, "member": member, "crc32": crc,
                        "member_sha256": sha256(data), "whole_archive_downloaded": False})
        print(f"NOISE {name}: {remote.received} range bytes", flush=True)
    (output / "noise_manifest.jsonl").write_text(
        "".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/sources/g1_v1")
    parser.add_argument("--plan", type=Path, default=ROOT/"configs/g1_source_plan.json")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output exists; refusing to overwrite source evidence")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "source_plan_frozen.json").write_bytes(plan_bytes)
    report = {"plan_sha256": sha256(plan_bytes), "status": "running", "training_performed": False}
    try:
        with requests.Session() as session:
            report["speech"] = fetch_speech(session, args.output_dir, plan)
            report["noise"] = fetch_noises(session, args.output_dir)
        report["status"] = "ok"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output_dir / "fetch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
