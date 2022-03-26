import sys
from loguru import logger
from .observer import LoggerSubscriber, LoggerObserver

logger.remove()

class BaseTextLogger(LoggerSubscriber):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html
    
    Usage:
        from modules.logger import BaseTextLogger
        LOGGER = BaseTextLogger.init_logger(__name__)

    """

    message_format = """<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <light-black>{file}</light-black>:<light-black>{function}</light-black>:<light-black>{line}</light-black> - <level>{message}</level>"""

    def __init__(self, name):
        self.name = name

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        if level == LoggerObserver.WARN:
            logger.warning(value)

        if level == LoggerObserver.INFO:
            logger.info(value)

        if level == LoggerObserver.ERROR:
            logger.error(value)

        if level == LoggerObserver.DEBUG:
            logger.debug(value)

        if level == LoggerObserver.SUCCESS:
            logger.success(value)
        

class FileLogger(BaseTextLogger):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html
    
    Usage:
        from modules.logger import FileLogger
        LOGGER = FileLogger.init_logger(__name__)

    """

    def __init__(self, name, logdir, rotation="10 MB", debug=False):
        self.logdir = logdir
        self.filename = f'{self.logdir}/log.txt'
        super().__init__(name)

        if debug:
            level = "DEBUG"
        else:
            level = "INFO"

        logger.add(
            self.filename, 
            rotation=rotation, 
            level=level,
            filter=lambda record: "filelog" in record["extra"]
        )
    
    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        with logger.contextualize(filelog=True):
            return super().log_text(tag, value, level, **kwargs)

class StdoutLogger(BaseTextLogger):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html
    
    Usage:
        from modules.logger import StdoutLogger
        LOGGER = StdoutLogger.init_logger(__name__)

    """

    def __init__(self, name, debug=False):
        super().__init__(name)

        if debug:
            level = "DEBUG"
        else:
            level = "INFO"

        logger.add(
            sys.stdout, 
            backtrace=True, 
            diagnose=True,
            level=level, 
            format = self.message_format,
            filter=lambda record: "stdout" in record["extra"]
        )

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        with logger.contextualize(stdout=True):
            return super().log_text(tag, value, level, **kwargs)