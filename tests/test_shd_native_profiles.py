import copy
import unittest
import constants


class SHDProfiles(unittest.TestCase):
    def test_data_and_budget_remain_pp3(self):
        for method in constants.SHD_NATIVE_PROFILES:
            config = constants.make_config(method, 'shd')
            self.assertEqual(config['data'], constants.DATASETS['shd'])
            self.assertEqual(config['model']['epochs'], constants.EPOCHS)
            self.assertEqual(config['model']['batch_size'], constants.BATCH_SIZE)
            self.assertEqual(config['data']['steps'], 2000)
        self.assertEqual(constants.make_config('spsn', 'shd')['model']['time_step'],
                         constants.DATASETS['shd']['dt'])
        self.assertEqual(constants.DATASETS['shd']['dt'], .0005)

    def test_shd_only_and_copy_isolation(self):
        self.assertEqual(constants.make_config('espp', 'shd')['model']['readout_lr'], .0003)
        self.assertEqual(constants.make_config('espp', 'texture')['model']['readout_lr'], .001)
        before = copy.deepcopy(constants.METHODS)
        constants.make_config('espp', 'shd')['model']['beta'][0] = 0
        self.assertEqual(before, constants.METHODS)
        self.assertNotIn('hyperparameter_protocol', constants.make_config('gatezo', 'shd'))
        self.assertNotIn('hyperparameter_protocol', constants.make_config('tp', 'texture'))

    def test_native_tp_script_matches_profile(self):
        from pathlib import Path
        import shlex
        script = (Path(__file__).resolve().parents[1] /
                  'traces_propagation-main/experiments/shd.sh').read_text()
        block = script.split('# python main.py', 1)[1].split('# ### EXPECTED RESULTS', 1)[0]
        tokens = shlex.split(' '.join(line.removeprefix('#').replace('\\', '')
                                     for line in block.splitlines()))
        values = dict(zip(tokens, tokens[1:]))
        profile = constants.make_config('tp', 'shd')['model']
        for source, target in [('learning_rate', 'lr'), ('hidden_layers_size', 'hidden_size'),
                               ('hidden_layers', 'layers'), ('l_vth', 'l_vth'),
                               ('l_leak_m', 'l_leak_m'), ('l_leak_t', 'l_leak_t'), ('T', 'T')]:
            self.assertEqual(float(values['--'+source]), profile[target])
        self.assertEqual(values['--optim'], profile['optim'])
        self.assertEqual(values['--scheduler_name'], profile['scheduler_name'])


if __name__ == '__main__':
    unittest.main()
