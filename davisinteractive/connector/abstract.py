from __future__ import absolute_import, division


class AbstractConnector:

    def get_samples(self, subset, davis_root=None):
        raise NotImplementedError('This is an abstract class')

    def get_scribble(self, sequence, scribble_idx):
        raise NotImplementedError('This is an abstract class')

    def post_predicted_masks(self, sequence, scribble_idx, pred_masks, timming,
                             interaction):
        raise NotImplementedError('This is an abstract class')

    def get_report(self):
        raise NotImplementedError('This is an abstract class')
