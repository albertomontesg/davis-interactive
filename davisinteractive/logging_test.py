import unittest

from davisinteractive import logging


class TestLogging:

    def test_level(self, caplog):

        logging.set_verbosity(logging.WARN)
        logging.info('Test info')

        assert not caplog.records

        logging.warn('Test warn')
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == 'WARNING'
        assert record.msg == 'Test warn'

    def test_verbose(self, caplog):
        logging.set_verbosity(logging.WARN)
        logging.verbose('Test verbose')

        assert not caplog.records

        logging.set_verbosity(logging.INFO)
        logging.verbose('Test verbose')

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == 'INFO'
        assert record.msg == 'Test verbose'

        caplog.clear()

        logging.set_info_level(1)
        logging.verbose('Test verbose 2', 2)
        assert not caplog.records
