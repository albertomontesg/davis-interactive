from __future__ import absolute_import, division

import os
import random

import pandas as pd
import requests
from six.moves.urllib.parse import urlparse

from ..third_party import mask_api
from .abstract import AbstractConnector


class RemoteConnector(AbstractConnector):  # pragma: no cover
    """ Proxy class to run the EvaluationService on a remote server.
    """

    VALID_SUBSETS = ['test-dev']
    GET_SAMPLES_URL = 'api/dataset/samples'
    GET_SCRIBBLE_URL = 'api/dataset/scribbles/{sequence}/{scribble_idx:03d}'
    POST_PREDICTED_MASKS_URL = 'api/evaluation/interaction'
    GET_REPORT_URL = 'api/evaluation/report'

    def __init__(self, user_key, session_key, host):
        self.user_key = user_key
        self.session_key = session_key
        o = urlparse(host)
        self.host = "{}://{}".format(o.scheme, o.netloc)
        if user_key is None or session_key is None:
            raise ValueError('user_key and session_key must be specified')
        self.session = requests.Session()

    def get_samples(self, subset, davis_root=None):
        if subset not in self.VALID_SUBSETS:
            raise ValueError('subset must be a valid one: {}'.format(
                self.VALID_SUBSETS))
        r = self.session.get(os.path.join(self.host, self.GET_SAMPLES_URL))
        assert r.status_code == 200
        response = r.json()

        samples, max_t, max_i = response
        random.shuffle(samples)

        return samples, max_t, max_i

    def get_scribble(self, sequence, scribble_idx):
        r = self.session.get(
            os.path.join(self.host,
                         self.GET_SCRIBBLE_URL.format(
                             sequence=sequence, scribble_idx=scribble_idx)))
        assert r.status_code == 200
        scribble = r.json()
        return scribble

    def post_predicted_masks(self, sequence, scribble_idx, pred_masks, timing,
                             interaction):
        pred_masks_enc = mask_api.encode_batch_masks(pred_masks)

        body = {
            'sequence': sequence,
            'scribble_idx': scribble_idx,
            'pred_masks': pred_masks_enc,
            'timing': timing,
            'interaction': interaction,
        }

        headers = {'User-Key': self.user_key, 'Session-Key': self.session_key}
        r = self.session.post(
            os.path.join(self.host, self.POST_PREDICTED_MASKS_URL),
            json=body,
            headers=headers)
        assert r.status_code == 200
        response = r.json()
        return response

    def get_report(self):
        headers = {'User-Key': self.user_key, 'Session-Key': self.session_key}
        r = self.session.get(
            os.path.join(self.host, self.GET_REPORT_URL), headers=headers)
        df = pd.DataFrame.from_dict(r.json())
        return df
