#!/usr/bin/env python3
import json,os,subprocess,sys
import g13_geometry_audit as audit
from g13_geometry_audit import ROOT,OUT,sha,now,dump,archive


def main():
    result=json.loads((OUT/'audit.json').read_text());assert result['status']=='ok'
    log=OUT/'full96_unit_tests.log'
    with log.open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=f,stderr=subprocess.STDOUT,timeout=300,check=True)
    assert 'Ran 96 tests' in log.read_text() and '\nOK' in log.read_text()
    checked=0;valid=True
    for path in list((OUT/'stages').glob('*/archive_manifest.json'))+[ROOT/'artifacts/g12_room_stress_v2/stages/geometry_stop/archive_manifest.json']:
        for r in json.loads(path.read_text())['files']:valid&=sha(path.parent/r['snapshot'])==r['sha256'];checked+=1
    assert valid
    version_files=['scripts/g5_prepare.py','scripts/g7_build.py','scripts/g10_build.py','src/mcse/spatial_noise.py','src/mcse/simulation.py']
    final={'status':'ok','completed_utc':now(),'full_tests':96,'prior_snapshot_files_checked':checked,'all_prior_snapshots_match':True,
        'legacy_sources_preserved_sha256':{name:sha(ROOT/name) for name in version_files},'no_legacy_code_audio_weights_or_scores_changed':True,
        'next':'Corrected/freefield paired diagnostic with frozen legacy weights; room test deferred'}
    report=ROOT/'docs/G13_几何审计验收与后续决策.md';report.write_text('# G13 几何审计验收与后续决策\n\n24个保存六通道输入与3条训练数组逐样本还原通过；96项完整测试通过，审计与G12中止快照SHA核验通过。旧代码、音频、权重和评分未修改，G12B没有增强或ASR成绩。\n\n此前的整场景旋转解释须以[G13A更正](G13A_旋转坐标审计与历史结论更正.md)为准。接下来冻结已有15系统，使用同一批源句/片段做旧反向旋转与正确刚性旋转的配对自由场诊断；共同尺度与参考波形一致，目标20°为两协议相同的负控制，不重新训练。该实验评估测试几何修正的影响，不能证明正确旋转训练的效果。\n\n混响实验后移，避免把几何错误与新混响混为一谈。若需训练，另立正确训练/验证及未查看确认数据，不覆盖已有模型。\n\nUTC：'+now()+'\n',encoding='utf-8')
    dump(OUT/'FINAL_ACCEPTANCE_AUDIT.json',final,True);archive('acceptance',report,[OUT/'FINAL_ACCEPTANCE_AUDIT.json',log,ROOT/'scripts/g13_accept.py']+[ROOT/name for name in version_files]);print(json.dumps(final))


if __name__=='__main__':main()
