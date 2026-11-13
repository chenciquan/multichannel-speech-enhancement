import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from g10_finish import cluster_effect


class SpeakerClusterEffectTests(unittest.TestCase):
    def fixture(self):
        rows=[];scores={}
        for i,(n,a,b) in enumerate([(1,0,1),(9,9,0),(2,1,1),(8,4,0)]):
            for system,error in [('a',a),('b',b)]:
                row={'system':system,'sample_id':str(i),'source_id':str(i),'speaker_id':str(i),'metrics':{'si_sdr_db':10 if system=='a' else 0}}
                rows.append(row);scores[(system,str(i))]={'score':{'N':n,'errors':error,'S':error,'D':0,'I':0,'correct':n-error}}
        return rows,scores

    def test_micro_WER_speaker_effect_and_reproducible_interval(self):
        rows,scores=self.fixture();value=cluster_effect(rows,scores,scores,['a'],['b'])
        self.assertAlmostEqual(value['delta']['Whisper_WER'],.6)
        self.assertAlmostEqual(value['delta']['si_sdr_db'],10.)
        self.assertAlmostEqual(value['per_speaker']['0']['CTC_WER'],-1.)
        self.assertEqual(value,cluster_effect(rows,scores,scores,['a'],['b']))

    def test_four_speaker_contract(self):
        rows,scores=self.fixture()
        with self.assertRaises(AssertionError):cluster_effect(rows[:-2],scores,scores,['a'],['b'])


if __name__=='__main__':unittest.main()
