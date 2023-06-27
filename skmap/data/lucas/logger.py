'''
LUCAS-related logging.
'''

import os
import logging
import logging.config

class LucasRequestLogger(logging.getLoggerClass()):
    pass

def logger():
    """Return a logger.
    """
    logging.config.fileConfig(
        os.path.join(os.path.dirname(__file__), 'logging.conf')
    )

    logging.setLoggerClass(LucasRequestLogger)
    logger = logging.getLogger('LUCAS')

    return logger

Logger = logger()
