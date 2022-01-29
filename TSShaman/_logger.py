import logging
from typing import Union


class CustomFormatter(logging.Formatter):

    grey = "\x1b[0;37m"
    white = "\x1b[0;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"

    reset = "\x1b[0m"

    format = "%(asctime)s %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.default_time_format = '%H:%M:%S'
        formatter.default_msec_format = '%s.%03d'
        return formatter.format(record)


# Initialise logger
logger = logging.getLogger("TSShaman")
handler = logging.StreamHandler()
handler.setLevel(logging.NOTSET)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)


def set_verbosity(verbosity: Union[str, int, bool] = 'info'):
    levels = {'CRITICAL': 50, 'FATAL': 50, 'Fatal': 50, 'fatal': 50, 1: 50,
              'ERROR': 40, 'Error': 40, 'error': 40, 2: 40, False: 40,
              'WARNING': 30, 'WARN': 30, 'Warning': 30, 'warning': 30, 3: 30,
              'INFO': 20, 'Info': 20, 'info': 20, 4: 20, True: 20,
              'DEBUG': 10, 'Debug': 10, 'debug': 10, 5: 10,
              'NOTSET': 0, 0: 0}

    logger.setLevel(levels[verbosity])
