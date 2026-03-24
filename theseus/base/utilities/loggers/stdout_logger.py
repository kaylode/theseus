import sys

from loguru import logger

from .observer import LoggerObserver, LoggerSubscriber

logger.remove()


class BaseTextLogger(LoggerSubscriber):
    """
    Logger class for showing text in prompt and file
    For more documents, look into https://docs.python.org/3/library/logging.html

    Usage:
        from modules.logger import BaseTextLogger
        LOGGER = BaseTextLogger.init_logger(__name__)

    """

    message_format = """<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <light-black>{extra[filename]}</light-black>:<light-black>{extra[funcname]}</light-black>:<light-black>{extra[lineno]}</light-black> - <level>{message}</level>"""

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
        os.makedirs(self.logdir, exist_ok=True)
        self.filename = f"{self.logdir}/log.txt"
        super().__init__(name)

        if debug:
            level = "DEBUG"
        else:
            level = "INFO"

        logger.add(
            self.filename,
            rotation=rotation,
            level=level,
            filter=lambda record: (
                record["extra"].get("filelog") and record["extra"].get("name") == self.name
            ),
        )

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        filename = kwargs.get("filename")
        funcname = kwargs.get("funcname")
        lineno = kwargs.get("lineno")
        with logger.contextualize(
            filelog=True, name=self.name, filename=filename, funcname=funcname, lineno=lineno
        ):
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
            format=self.message_format,
            filter=lambda record: (
                record["extra"].get("stdout") and record["extra"].get("name") == self.name
            ),
        )

    def log_text(self, tag, value, level=LoggerObserver.DEBUG, **kwargs):
        filename = kwargs.get("filename")
        funcname = kwargs.get("funcname")
        lineno = kwargs.get("lineno")
        with logger.contextualize(
            stdout=True, name=self.name, filename=filename, funcname=funcname, lineno=lineno
        ):
            return super().log_text(tag, value, level, **kwargs)
