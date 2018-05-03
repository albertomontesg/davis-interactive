from __future__ import absolute_import, division

import os
import random

import pandas as pd
import requests

from .. import logging
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
        if self._handle_response(r):
            raise NameError('Server {} not found'.format(self.host))

    def get_samples(self, subset, davis_root=None):
        if subset not in self.VALID_SUBSETS:
            raise ValueError('subset must be a valid one: {}'.format(
                self.VALID_SUBSETS))
        r = requests.get(
            os.path.join(self.host, self.GET_SAMPLES_URL), headers=self.headers)
        self._handle_response(r, raise_error=True)
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
        self._handle_response(r, raise_error=True)
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
        self._handle_response(r, raise_error=True)
        response = r.json()
        return response

    def get_report(self):
        r = requests.get(
            os.path.join(self.host, self.GET_REPORT_URL), headers=self.headers)
        self._handle_response(r, raise_error=True)
        df = pd.DataFrame.from_dict(r.json())
        return df

    def _handle_response(self, response, raise_error=False):
        """ Checks the status code of the response and log the error if any.
        """
        if response.status_code >= 400 and response.status_code < 500:
            logging.error('Remote server error')
            e_body = response.json()
            error_name, error_msg = e_body['error'], e_body['message']
            logging.error('{}: {}'.format(error_name, error_msg))
            if raise_error:
                # Reconstruct error
                error_class = eval(error_name)
                error = error_class(*error_msg)
                raise error
            return True
        elif response.status_code == 500:
            logging.error('Uknown Error')
            logging.fatal(response.json())
            return True
        return False
