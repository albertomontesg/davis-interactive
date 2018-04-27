import unittest

import pandas as pd

from davisinteractive.common import Path, patch
from davisinteractive.dataset import Davis
from davisinteractive.evaluation import EvaluationService
from davisinteractive.utils.scribbles import annotated_frames, is_empty


class TestEvaluationService(unittest.TestCase):

    @patch.object(Davis, 'check_files', return_value=True)
    def test_start(self, mock_davis):
        assert mock_davis.call_count == 0
        service = EvaluationService('train', davis_root='/tmp/DAVIS')
        assert mock_davis.call_count == 1

        samples, max_t, max_i = service.get_samples()

        assert max_t is None
        assert max_i is None

        for seq in Davis.sets['train']:
            nb_scribbles = Davis.dataset[seq]['num_scribbles']

            for i in range(nb_scribbles):
                assert (seq, i + 1) in samples

        service = EvaluationService('val', davis_root='/tmp/DAVIS')
        samples, max_t, max_i = service.get_samples()
        assert mock_davis.call_count == 2

        assert max_t is None
        assert max_i is None

        for seq in Davis.sets['val']:
            nb_scribbles = Davis.dataset[seq]['num_scribbles']

            for i in range(nb_scribbles):
                assert (seq, i + 1) in samples

    @patch.object(Davis, 'check_files', return_value=True)
    def test_starting_scribble(self, mock_davis):
        dataset_dir = Path(__file__).parent.parent.joinpath(
            'dataset', 'test_data', 'DAVIS')

        service = EvaluationService('train', davis_root=dataset_dir)
        service.get_samples()

        scribble = service.get_scribble('bear', 1)
        assert scribble['sequence'] == 'bear'
        assert not is_empty(scribble)
        assert annotated_frames(scribble) == [39]

    # @patch.object(Davis, 'check_files', return_value=True)
    # def test_report(self, mock_davis):
    #     service = EvaluationService(davis_root='/tmp/DAVIS')

    #     assert service.get_report() is None

    #     assert mock_davis.call_count == 0
    #     service.start('train')
    #     assert mock_davis.call_count == 1

    #     report = service.get_report()
    #     assert isinstance(report, pd.DataFrame)
