import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from g14_finish import protocol_effect


class GeometryContrastTests(unittest.TestCase):
    def rows(self):
        rows=[];wi={};ci={}
        for i,n in enumerate((10,10,10,100)):
            for protocol in ('legacy_counterrotation','rigid_rotation'):
                rigid=protocol=='rigid_rotation';sid=f'{i}_{protocol}';key=('m',sid)
                rows.append({'speaker_id':str(i),'system':'m','sample_id':sid,'geometry_protocol':protocol,'metrics':{'si_sdr_db':float(rigid)}})
                def score(errors):return {'N':n,'S':errors,'D':0,'I':0,'correct':n-errors,'errors':errors,'wer':errors/n}
                wi[key]={'score':score(5 if rigid and i==0 else 0)};ci[key]={'score':score(2 if rigid and i==1 else 0)}
        return rows,wi,ci

    def test_paired_micro_errors_are_counted_before_dividing(self):
        rows,wi,ci=self.rows();r=protocol_effect(rows,wi,ci,['m'])
        self.assertAlmostEqual(r['delta']['si_sdr_db'],1)
        self.assertAlmostEqual(r['delta']['Whisper_WER'],5/130)
        self.assertAlmostEqual(r['delta']['CTC_WER'],2/130)
        self.assertEqual(len(r['per_speaker']),4)

    def test_missing_protocol_pair_is_rejected(self):
        rows,wi,ci=self.rows()
        with self.assertRaises(AssertionError):protocol_effect(rows[:-1],wi,ci,['m'])
