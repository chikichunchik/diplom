import time
import json
import traceback
from loguru import logger


class Logger:
    """
    Logger which saves logs to a local file.

    --Example usage

    logger = Logger('logs.txt')
    logger.add_log("Something happened", "INFO", {"key": "value"})
    """
    def __init__(self, log_file):
        self.log_file = log_file
        self.__setup_logging(log_file)

    def add_log(
            self,
            message: str = '',
            level: str = 'INFO',
            data: dict = None
    ):
        try:
            log_entry = {
                'Timestamp': time.time(),
                'Message': message,
                'Level': level,
            }
            if data:
                log_entry['Data'] = data

            logger.info(json.dumps(log_entry))
        except Exception as e:
            raise Exception from e

    @staticmethod
    def __setup_logging(log_file):
        logger.remove()
        logger.add(
            log_file,
            rotation="2 hours",
            retention="4 hours",
            format="{message}",
            enqueue=False,
        )