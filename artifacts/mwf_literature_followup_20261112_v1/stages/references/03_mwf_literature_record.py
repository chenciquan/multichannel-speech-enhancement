#!/usr/bin/env python3
import inspect,json
from pathlib import Path
import torch,torchaudio
import g8_common as common


def main():
    root=common.ROOT;out=root/'artifacts/mwf_literature_followup_20261112_v1';out.mkdir(exist_ok=False)
    source=inspect.getsource(torchaudio.functional.psd);(out/'installed_psd_source.txt').write_text(source,encoding='utf-8')
    common.dump(out/'verification.json',{'verified_utc':common.now(),'Exa_queries':1,'search_candidates':5,'primary_PDFs_checked':2,'systematic_review':False,
        'query':'"speech distortion weighted" "multichannel Wiener" Doclo Moonen covariance',
        'primary_sources':[{'url':'https://ftp.esat.kuleuven.be/pub/stadius/doclo/reports/05-114.pdf','year':2005,'volume':12,'issue':12,'pages':'809-811','DOI':'10.1109/LSP.2005.859530','verified':'originalPDFfirstpage'},
           {'url':'https://uol.de/f/6/dept/mediphysik/ag/sigproc/download/papers/doclo/journal_SDWMWF.pdf','year':2007,'volume':49,'pages':'636-656','DOI':'10.1016/j.specom.2007.02.001','verified':'originalPDFfirstpage'}],
        'local_source_file':inspect.getsourcefile(torchaudio.functional.psd),'source_function_sha256':common.sha(out/'installed_psd_source.txt'),'torch':torch.__version__,'torchaudio':torchaudio.__version__,
        'new_algorithm_implemented':False,'G18_code_changed':False},True)
    old=common.OUT
    try:
        common.OUT=out;common.archive('references',root/'docs/研究补查_失真加权Wiener成熟对照_20261112.md',[out/'verification.json',out/'installed_psd_source.txt',Path(__file__)])
    finally:common.OUT=old
    print('Saved primary-source and installed PSD convention verification')


if __name__=='__main__':main()
