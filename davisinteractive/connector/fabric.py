from __future__ import absolute_import, division

from absl import logging

from .local import LocalConnector


class ServerConnectionFabric:

    @staticmethod
    def get_connector(host, user_key, session_key):
        if host == 'localhost':
            logging.info('Created connector to localhost service')
            return LocalConnector(user_key=user_key, session_key=session_key)
        else:
            raise NotImplementedError('Remote connection not implemented')
