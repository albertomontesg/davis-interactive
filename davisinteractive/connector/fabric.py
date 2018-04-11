from __future__ import absolute_import, division

from absl import logging

from .local import LocalConnector


class ServerConnectionFabric:

    @staticmethod
    def get_connector(host, key):
        if host == 'localhost':
            logging.info('Created connector to localhost service')
            return LocalConnector()
        else:
            raise NotImplementedError('Remote connection not implemented')
