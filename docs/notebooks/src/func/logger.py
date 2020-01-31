import os
import logging
from datetime import datetime


class Logger(object):
    def __init__(self, dir_out):
        self._logger = logging.getLogger('runner')
        self._logger.setLevel(logging.DEBUG)

        self._handler = None
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        self.set_handler(dir_out)

    def set_handler(self, dir_out):
        start = datetime.now()
        timestr = start.strftime('%Y%m%d-%H%M%S')
        self._handler = logging.FileHandler(f'{dir_out}/log_{timestr}.log', mode='w')
        self._handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self._handler.setFormatter(formatter)
        self._logger.addHandler(self._handler)

    def remove_handler(self):
        self._handler.close()
        self._logger.removeHandler(self._handler)

    def log_param(self, param):
        self._logger.info('=' * 33 + 'PARAM' + '=' * 33)

        def transform_v(v):
            if hasattr(v, '__iter__') and not isinstance(v, str):
                v = [str(_) for _ in v]
                v = ', '.join(v)
            elif isinstance(v, float):
                v = f'{v:.2f}'
            else:
                v = str(v)
            return v

        for k1, v1 in param.items():
            if isinstance(v1, dict):
                self._logger.info(f'{k1}:')
                for k2, v2 in v1.items():
                    if isinstance(v2, dict):
                        self._logger.info(f'\t{k2}:')
                        for k3, v3 in v2.items():
                            v3 = transform_v(v3)
                            self._logger.info(f'\t\t{k3}: {v3}')
                    else:
                        v2 = transform_v(v2)
                        self._logger.info(f'\t{k2}: {v2}')
            else:
                v1 = transform_v(v1)
                self._logger.info(f'{k1}: {v1}')

    def log_process(self, level='info', msg='', header=False):
        if header is True:
            self._logger.info('=' * 33 + 'PROC' + '=' * 33)
        else:
            getattr(self._logger, level)(msg)

    def log_err(self, msg, exc_info=False):
        self._logger.error(msg, exc_info=exc_info)

    def end(self):
        self._logger.info('=' * 33 + 'END' + '=' * 33)
        self.remove_handler()
