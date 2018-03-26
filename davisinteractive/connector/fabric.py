from .local import LocalConnector


class ServerConnectionFabric:
    @staticmethod
    def get_connector(host, key):
        if host == 'localhost':
            return LocalConnector()
        else:
            raise NotImplementedError('Remote connection not implemented')
