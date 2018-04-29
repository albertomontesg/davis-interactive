from __future__ import absolute_import, division

import json
import os
import tempfile
import time
import unittest
from functools import wraps

import numpy as np
import pandas as pd
import pytest

from davisinteractive.common import Path, patch
from davisinteractive.connector.local import LocalConnector
from davisinteractive.dataset import Davis
from davisinteractive.session import DavisInteractiveSession
from davisinteractive.utils.scribbles import annotated_frames, is_empty

EMPTY_SCRIBBLE = {
    'scribbles': [[] for _ in range(69)],
    'sequence': 'test-sequence'
}


def dataset(subset, **samples):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            sequence_set = Davis.sets[subset]
            new_set = list(samples.keys())
            new_set.sort()  # Necessary as the keys order is not garanteed
            original = Davis.dataset.copy()

            for seq in samples:
                Davis.dataset[seq].update(samples[seq])
            Davis.sets[subset] = new_set

            result = func(*args, **kwargs)

            # Recover orignal state at Davis class
            Davis.sets[subset] = sequence_set
            Davis.dataset = original

            return result

        return wrapper

    return decorator


class TestDavisInteractiveSession(unittest.TestCase):

    @patch.object(Davis, 'check_files', return_value=None)
    def test_subset(self, mock_davis):
        davis_root = '/tmp/DAVIS'

        session = DavisInteractiveSession(subset='val', davis_root=davis_root)
        session.__enter__()
        session = DavisInteractiveSession(subset='train', davis_root=davis_root)
        session.__enter__()

        session1 = DavisInteractiveSession(
            subset='test-dev', davis_root=davis_root)
        session2 = DavisInteractiveSession(subset='xxxx', davis_root=davis_root)
        with pytest.raises(ValueError):
            session1.__enter__()
        with pytest.raises(ValueError):
            session2.__enter__()

        assert mock_davis.call_count == 2

    @patch.object(
        LocalConnector, 'post_predicted_masks', return_value=EMPTY_SCRIBBLE)
    @patch.object(LocalConnector, 'get_report', return_value=pd.DataFrame())
    @patch.object(LocalConnector, 'get_scribble', return_value=EMPTY_SCRIBBLE)
    @patch.object(
        LocalConnector,
        'get_samples',
        return_value=([('bear', 2), ('bear', 1)], 5, None))
    def test_interactions_limit(self, mock_start_session, mock_get_scribble,
                                mock_get_report, mock_submit_masks):
        davis_root = '/tmp/DAVIS'

        with DavisInteractiveSession(
                davis_root=davis_root,
                max_nb_interactions=5,
                report_save_dir=tempfile.mkdtemp(),
                max_time=None) as session:

            assert mock_start_session.call_count == 1

            for i in range(7):
                assert session.next()
                seq, scribbles, new_seq = session.get_scribbles()
                assert seq == 'bear'
                assert is_empty(scribbles)
                if i % 5 == 0:
                    assert new_seq, i

                session.submit_masks(None)

        assert mock_get_scribble.call_count == 2
        assert mock_submit_masks.call_count == 7

    @dataset('train', bear={'num_frames': 2, 'num_scribbles': 1})
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_integration_single(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')

        tmp_dir = Path(tempfile.mkdtemp())

        with DavisInteractiveSession(
                davis_root=dataset_dir,
                subset='train',
                max_nb_interactions=5,
                report_save_dir=tmp_dir,
                max_time=None) as session:
            count = 0

            temp_csv = tmp_dir / ("%s.tmp.csv" % session.report_name)
            final_csv = tmp_dir / ("%s.csv" % session.report_name)

            while session.next():
                seq, scribble, new_seq = session.get_scribbles()
                assert new_seq == (count == 0)
                assert seq == 'bear'
                if count == 0:
                    with dataset_dir.joinpath('Scribbles', 'bear',
                                              '001.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                assert not is_empty(scribble)

                # Simulate model predicting masks
                pred_masks = np.zeros((2, 480, 854))

                session.submit_masks(pred_masks)

                count += 1

                assert not final_csv.exists()
                assert temp_csv.exists()

                df = pd.read_csv(temp_csv, index_col=0)
                assert df.shape == (count * 2, 8)
                assert df.sequence.unique() == ['bear']
                assert np.all(
                    df.interaction.unique() == [i + 1 for i in range(count)])
                assert np.all(df.object_id.unique() == [1])

            assert count == 5
            assert final_csv.exists()
            assert not temp_csv.exists()

        assert mock_davis.call_count == 0

    @dataset('train', bear={'num_frames': 2, 'num_scribbles': 2})
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_integration_multiple(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')

        with DavisInteractiveSession(
                davis_root=dataset_dir,
                subset='train',
                max_nb_interactions=5,
                report_save_dir=tempfile.mkdtemp(),
                max_time=None) as session:
            count = 0

            while session.next():
                seq, scribble, new_seq = session.get_scribbles()
                assert new_seq == (count == 0 or count == 5)
                assert seq == 'bear'
                if count == 0:
                    with dataset_dir.joinpath('Scribbles', 'bear',
                                              '001.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                if count == 5:
                    with dataset_dir.joinpath('Scribbles', 'bear',
                                              '002.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                assert not is_empty(scribble)

                # Simulate model predicting masks
                pred_masks = np.zeros((2, 480, 854))

                session.submit_masks(pred_masks)

                count += 1

            assert count == 10

        assert mock_davis.call_count == 0

    @dataset(
        'train',
        bear={
            'num_frames': 2,
            'num_scribbles': 2
        },
        tennis={
            'num_frames': 2,
            'num_scribbles': 1
        })
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_integration_multiple_sequences(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')

        with DavisInteractiveSession(
                davis_root=dataset_dir,
                subset='train',
                max_nb_interactions=4,
                report_save_dir=tempfile.mkdtemp(),
                max_time=None) as session:
            count = 0

            for seq, scribble, new_seq in session.scribbles_iterator():
                assert new_seq == (count == 0 or count == 4 or count == 8)
                if count < 8:
                    assert seq == 'bear', count
                else:
                    assert seq == 'tennis', count
                if count == 0:
                    with dataset_dir.joinpath('Scribbles', 'bear',
                                              '001.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                if count == 4:
                    with dataset_dir.joinpath('Scribbles', 'bear',
                                              '002.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                if count == 8:
                    with dataset_dir.joinpath('Scribbles', 'tennis',
                                              '001.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                assert not is_empty(scribble)

                # Simulate model predicting masks
                pred_masks = np.zeros((2, 480, 854))

                session.submit_masks(pred_masks)

                count += 1

            assert count == 12
            df = session.get_report()

        assert mock_davis.call_count == 0

        assert df.shape == (2 * 4 * 2 * 1 + 4 * 2 * 2, 8)
        assert np.all(df.jaccard == 0.)

    @dataset('train', bear={'num_frames': 2, 'num_scribbles': 1})
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_integration_single_only_last(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')

        with DavisInteractiveSession(
                davis_root=dataset_dir,
                subset='train',
                max_nb_interactions=4,
                report_save_dir=tempfile.mkdtemp(),
                max_time=None) as session:
            count = 0

            scribble_iter = None

            while session.next():
                seq, scribble, new_seq = session.get_scribbles(only_last=True)
                assert new_seq == (count == 0)
                assert seq == 'bear'
                if count == 0:
                    with dataset_dir.joinpath('Scribbles', 'bear',
                                              '001.json').open() as fp:
                        sc = json.load(fp)
                        assert sc == scribble
                elif count == 1:
                    scribble_iter = scribble
                else:
                    assert annotated_frames(scribble) == annotated_frames(
                        scribble_iter)
                assert not is_empty(scribble)

                # Simulate model predicting masks
                pred_masks = np.zeros((2, 480, 854))

                session.submit_masks(pred_masks)

                count += 1

            assert count == 4

        assert mock_davis.call_count == 0

    @dataset('train', bear={'num_frames': 2, 'num_scribbles': 1})
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_integration_single_timeout(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')

        with DavisInteractiveSession(
                davis_root=dataset_dir,
                subset='train',
                max_nb_interactions=None,
                max_time=1,
                report_save_dir=tempfile.mkdtemp()) as session:
            count = 0

            while session.next():
                seq, scribble, new_seq = session.get_scribbles(only_last=True)
                assert new_seq == (count == 0)
                assert seq == 'bear'
                with dataset_dir.joinpath('Scribbles', 'bear',
                                          '001.json').open() as fp:
                    sc = json.load(fp)
                    assert sc == scribble
                assert not is_empty(scribble)

                # Simulate model predicting masks
                pred_masks = np.zeros((2, 480, 854))

                time.sleep(1.2)

                session.submit_masks(pred_masks)

                count += 1

            assert count == 1

        assert mock_davis.call_count == 0

    @dataset('train', bear={'num_frames': 2, 'num_scribbles': 1})
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_report_folder_creation(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')
        tmp_dir = Path(tempfile.mkdtemp()) / 'test'
        assert not tmp_dir.exists()

        session = DavisInteractiveSession(
            davis_root=dataset_dir, subset='train', report_save_dir=tmp_dir)
        assert tmp_dir.exists()
        assert mock_davis.call_count == 0

    @dataset(
        'train',
        bear={
            'num_frames': 2,
            'num_scribbles': 2
        },
        tennis={
            'num_frames': 2,
            'num_scribbles': 1
        })
    @patch.object(Davis, '_download_scribbles', return_value=None)
    def test_shuffle(self, mock_davis):
        dataset_dir = Path(__file__).parent.joinpath('test_data', 'DAVIS')

        with DavisInteractiveSession(
                davis_root=dataset_dir,
                subset='train',
                shuffle=True,
                report_save_dir=tempfile.mkdtemp()) as session:
            assert ('bear', 1) in session.samples
            assert ('bear', 2) in session.samples
            assert ('tennis', 1) in session.samples
        assert mock_davis.call_count == 0
