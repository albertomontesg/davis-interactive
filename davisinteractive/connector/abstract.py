from __future__ import absolute_import, division


class AbstractConnector:

    def start_session(self, subset, davis_root=None):
        raise NotImplementedError('This is an abstract class')

    def get_starting_scribble(self, sequence):
        raise NotImplementedError('This is an abstract class')

    def submit_masks(self, pred_masks):
        raise NotImplementedError('This is an abstract class')

    def get_report(self):
        raise NotImplementedError('This is an abstract class')
