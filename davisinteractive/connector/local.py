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

    VALID_SUBSETS = ['train', 'val']

    def __init__(self):
        self.service = None

    def start_session(self, subset, davis_root=None):
        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f'For local connector, `subset` must be a valid subset: {self.VALID_SUBSETS}'
            )

        self.service = EvaluationService(davis_root=davis_root)
        return self.service.start(subset)

    def get_starting_scribble(self, sequence, scribble_idx):
        return self.service.get_starting_scribble(sequence, scribble_idx)

    def submit_masks(self, sequence, scribble_idx, pred_masks, timming,
                     interaction):
        return self.service.submit_masks(sequence, scribble_idx, pred_masks,
                                         timming, interaction)

    def get_report(self):
        return self.service.get_report()

    def close(self):
        self.service.close()
