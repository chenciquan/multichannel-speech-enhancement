import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from g8_finish import paired


class PairedWERTest(unittest.TestCase):
    def rows(self):
        return [{'system':name+'_s1','sample_id':str(i),'source_id':str(i),'transcript':'same',
                 'score':{'N':n,'errors':e}} for name,counts in [('a',[(1,0),(9,9)]),('b',[(1,1),(9,0)])]
                for i,(n,e) in enumerate(counts)]

    def test_micro_not_mean_sentence_wer(self):
        value=paired(self.rows(),['a_s1'],['b_s1'])
        self.assertAlmostEqual(value['delta_wer'],.8)
        self.assertEqual(value['N_per_method'],10)
        self.assertEqual(value['sources'],2)

    def test_missing_pair_rejected(self):
        with self.assertRaises(AssertionError):paired(self.rows()[:-1],['a_s1'],['b_s1'])


if __name__=='__main__':unittest.main()
