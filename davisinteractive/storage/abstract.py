from __future__ import absolute_import, division


class AbstractStorage:
    """ Abstract class to store all evaluation results"""

    def store_interactions_results(self, user_id, session_id, sequence,
                                   scribble_idx, timing, objects_idx, frames,
                                   jaccard):
        raise NotImplementedError('This is an abstract class')

    def get_repot(self, user_id=None, session_id=None):
        raise NotImplementedError('This is an abstract class')
