import unittest

from davisinteractive.common import patch
from davisinteractive.dataset import Davis
from davisinteractive.evaluation import EvaluationService


class TestEvaluationService(unittest.TestCase):

    @patch.object(Davis, 'check_files', return_value=None)
    def test_start(self, mock_davis):
        service = EvaluationService(davis_root='/tmp/DAVIS')

        assert mock_davis.call_count == 0
        samples, max_t, max_i = service.start('train')
        assert mock_davis.call_count == 1

        assert max_t is None
        assert max_i is None

        for seq in Davis.sets['train']:
            nb_scribbles = Davis.dataset[seq]['num_scribbles']

            for i in range(nb_scribbles):
                assert (seq, i + 1) in samples
