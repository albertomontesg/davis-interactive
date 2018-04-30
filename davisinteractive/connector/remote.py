from __future__ import absolute_import, division

import os
import random

import pandas as pd
import requests

from ..dataset import Davis
from ..third_party import mask_api
from .abstract import AbstractConnector


class RemoteConnector(AbstractConnector):  # pragma: no cover
    """ Proxy class to run the EvaluationService on a remote server.
    """

    VALID_SUBSETS = ['test-dev']

    HEALTHCHECK_URL = 'api/healthcheck'
    GET_SAMPLES_URL = 'api/dataset/samples'
    GET_SCRIBBLE_URL = 'api/dataset/scribbles/{sequence}/{scribble_idx:03d}'
    POST_PREDICTED_MASKS_URL = 'api/evaluation/interaction'
    GET_REPORT_URL = 'api/evaluation/report'

    def __init__(self, user_key, session_key, host):
        self.user_key = user_key
        self.session_key = session_key
        self.host = host
        self.headers = {
            'User-Key': self.user_key,
            'Session-Key': self.session_key
        }

        if user_key is None or session_key is None:
            raise ValueError('user_key and session_key must be specified')
        r = requests.get(os.path.join(self.host, self.HEALTHCHECK_URL))
        if r.status_code != 200:
            raise NameError('Server {} not found'.format(self.host))

    def get_samples(self, subset, davis_root=None):
        if subset not in self.VALID_SUBSETS:
            raise ValueError('subset must be a valid one: {}'.format(
                self.VALID_SUBSETS))
        r = requests.get(
            os.path.join(self.host, self.GET_SAMPLES_URL), headers=self.headers)
        assert r.status_code == 200
        response = r.json()

        samples, max_t, max_i = response
        random.shuffle(samples)

        return samples, max_t, max_i

    def get_scribble(self, sequence, scribble_idx):
        r = requests.get(
            os.path.join(self.host,
                         self.GET_SCRIBBLE_URL.format(
                             sequence=sequence, scribble_idx=scribble_idx)),
            headers=self.headers)
        assert r.status_code == 200
        scribble = r.json()
        return scribble

    def post_predicted_masks(self, sequence, scribble_idx, pred_masks, timing,
                             interaction):
        nb_objects = Davis.dataset[sequence]['num_objects']
        pred_masks_enc = mask_api.encode_batch_masks(
            pred_masks, nb_objects=nb_objects)

        body = {
            'sequence': sequence,
            'scribble_idx': scribble_idx,
            'pred_masks': pred_masks_enc,
            'timing': timing,
            'interaction': interaction,
        }

        r = requests.post(
            os.path.join(self.host, self.POST_PREDICTED_MASKS_URL),
            json=body,
            headers=self.headers)
        assert r.status_code == 200
        response = r.json()
        return response

    def get_report(self):
        r = requests.get(
            os.path.join(self.host, self.GET_REPORT_URL), headers=self.headers)
        df = pd.DataFrame.from_dict(r.json())
        return df
