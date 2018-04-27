from __future__ import absolute_import, division

import getpass
import json
import os
import random
from glob import glob

import pandas as pd

from ..evaluation import EvaluationService
from .abstract import AbstractConnector


class LocalConnector(AbstractConnector):
    """ Proxy class to run the EvaluationService locally.
    """

    VALID_SUBSETS = ['train', 'val', 'trainval']

    def __init__(self, user_key, session_key):
        self.service = None
        self.user_key = user_key or getpass.getuser()
        self.session_key = session_key

    def get_samples(self, subset, davis_root=None):
        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                'For local connector, `subset` must be a valid subset: {}'.
                format(self.VALID_SUBSETS))

        self.service = EvaluationService(subset, davis_root=davis_root)
        return self.service.get_samples()

    def get_scribble(self, sequence, scribble_idx):
        return self.service.get_scribble(sequence, scribble_idx)

    def post_predicted_masks(self, sequence, scribble_idx, pred_masks, timming,
                             interaction):
        return self.service.post_predicted_masks(
            sequence, scribble_idx, pred_masks, timming, interaction,
            self.user_key, self.session_key)

    def get_report(self):
        return self.service.get_report(
            user_id=self.user_key, session_id=self.session_key)
