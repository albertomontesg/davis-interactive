from __future__ import absolute_import, division

import json
import os
import random

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .. import logging
from ..dataset import Davis
from ..third_party import mask_api
from .abstract import AbstractConnector


def _requests_retry_session(
        retries=3,
        backoff_factor=2,
        status_forcelist=(500, 502, 503, 504),
        session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class RemoteConnector(AbstractConnector):  # pragma: no cover
    """ Proxy class to run the EvaluationService on a remote server.
    """

    VALID_SUBSETS = ['test-dev']

    HEALTHCHECK_URL = 'api/healthcheck'
    GET_SAMPLES_URL = 'api/dataset/samples'
    GET_SCRIBBLE_URL = 'api/dataset/scribbles/{sequence}/{scribble_idx:03d}'
    POST_PREDICTED_MASKS_URL = 'api/evaluation/interaction'
    GET_REPORT_URL = 'api/evaluation/report'
    POST_FINISH = 'api/evaluation/finish'

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
        r = _requests_retry_session().get(
            os.path.join(self.host, self.HEALTHCHECK_URL))
        if self._handle_response(r):
            raise NameError('Server {} not found'.format(self.host))

    def get_samples(self,
                    subset,
                    max_t,
                    max_i,
                    davis_root=None,
                    metric_to_optimize='J_AND_F'):
        # This will be set in the server side
        del metric_to_optimize
        del max_t
        del max_i

        if subset not in self.VALID_SUBSETS:
            raise ValueError('subset must be a valid one: {}'.format(
                self.VALID_SUBSETS))
        r = _requests_retry_session().get(
            os.path.join(self.host, self.GET_SAMPLES_URL), headers=self.headers)
        self._handle_response(r, raise_error=True)
        response = r.json()

        samples, max_t, max_i = response
        random.shuffle(samples)

        return samples, max_t, max_i

    def get_scribble(self, sequence, scribble_idx):
        r = _requests_retry_session().get(
            os.path.join(
                self.host,
                self.GET_SCRIBBLE_URL.format(
                    sequence=sequence, scribble_idx=scribble_idx)),
            headers=self.headers)
        self._handle_response(r, raise_error=True)
        scribble = r.json()
        return scribble

    def post_predicted_masks(self,
                             sequence,
                             scribble_idx,
                             pred_masks,
                             timing,
                             interaction,
                             next_scribble_frame_candidates=None):
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
        if next_scribble_frame_candidates is not None:
            body[
                'next_scribble_frame_candidates'] = next_scribble_frame_candidates

        r = _requests_retry_session().post(
            os.path.join(self.host, self.POST_PREDICTED_MASKS_URL),
            json=body,
            headers=self.headers)
        self._handle_response(r, raise_error=True)
        response = r.json()
        return response

    def get_report(self):
        r = _requests_retry_session().get(
            os.path.join(self.host, self.GET_REPORT_URL), headers=self.headers)
        self._handle_response(r, raise_error=True)
        df = pd.DataFrame.from_dict(r.json())
        return df

    def post_finish(self):
        r = _requests_retry_session().post(
            os.path.join(self.host, self.POST_FINISH),
            json={},
            headers=self.headers)
        self._handle_response(r, raise_error=True)
        summary = r.json()
        summary['session_key'] = self.session_key
        return summary

    def _handle_response(self, response, raise_error=False):
        """ Checks the status code of the response and log the error if any.
        """
        if response.status_code >= 400 and response.status_code < 500:
            logging.error('Remote server error')
            e_body = response.content
            logging.error(response.status_code)
            logging.error(response.content)
            e_body = response.json()
            error_name, error_msg = e_body.get('error', ''), e_body.get(
                'message', '')
            logging.error('{}: {}'.format(error_name, error_msg))
            if raise_error and response.status_code == 400:
                # Reconstruct error
                error_class = eval(error_name)
                error = error_class(*error_msg)
                raise error
            elif raise_error:
                raise Exception(error_name)
            return True
        elif response.status_code == 500:
            logging.error('Uknown Error')
            logging.fatal(response.json())
            return True
        elif response.status_code == 503:
            logging.error('Server Unavailable')
            if raise_error:
                raise Exception('Servier Unavailable.')
        elif response.status_code > 500:
            logging.error('Error in server')
            logging.error(response.content)
        # Tries to decode the JSON response and if error print conent.
        if response.status_code == 200:
            try:
                response.json()
            except json.decoder.JSONDecodeError:
                logging.error(response.content)
        return False
