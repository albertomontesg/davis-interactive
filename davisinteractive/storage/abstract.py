from __future__ import absolute_import, division


class AbstractStorage:
    """ Abstract class to store all evaluation results"""

    COLUMNS = [
        'session_id', 'sequence', 'scribble_idx', 'interaction', 'object_id',
        'frame', 'jaccard', 'contour', 'j_and_f', 'timing'
    ]

    def store_interactions_results(self, user_id, session_id, sequence,
                                   scribble_idx, timing, objects_idx, frames,
                                   jaccard):
        raise NotImplementedError('This is an abstract class')

    def get_report(self, user_id=None, session_id=None):
        raise NotImplementedError('This is an abstract class')

    def get_annotated_frames(self, session_id, sequence, scribble_idx):
        raise NotImplementedError('This is an abstract class')

    def store_annotated_frame(self, session_id, sequence, scribble_idx,
                              annotated_frame, override):
        raise NotImplementedError('This is an abstract class')
