import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from g11_finish import compare,METRICS,aggregate


class ComponentSummaryTests(unittest.TestCase):
    def test_equal_speaker_effect_interval_has_known_answer(self):
        rows=[]
        for sp in range(4):
            for group,offset in [('a',1.),('b',0.)]:
                for repeated in range(3 if group=='a' else 1):
                    rows.append({'group':group,'speaker_id':str(sp),'metrics':{k:sp+offset for k in METRICS}})
        value=compare(rows,'a','b')
        for k in METRICS:
            self.assertEqual(value['delta'][k],1.);self.assertEqual(value['speaker_cluster_bootstrap2000_ci95'][k],[1.,1.])

    def test_undefined_SI_and_power_floors_remain_counted(self):
        score={'N':10,'S':1,'D':0,'I':0,'correct':9,'errors':1,'wer':.1}
        row={'source_id':'s','speaker_id':'p','metrics':{**dict.fromkeys(METRICS,1.),'target_component_si_sdr_db':None,
             'output_speech_power_floored':True,'output_noise_power_floored':False,'component_input_snr_db':0.},'Whisper':score,'CTC':score}
        value=aggregate([row]);self.assertEqual(value['records'],1);self.assertEqual(value['target_component_SI_valid_count'],0)
        self.assertEqual(value['speech_output_power_floored'],1);self.assertEqual(value['Whisper']['wer'],.1)


if __name__=='__main__':unittest.main()
