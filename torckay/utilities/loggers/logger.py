import os
import logging

class CustomFormatter(logging.Formatter):
    """
    Color schemes longging formater
    https://docs.microsoft.com/en-us/windows/terminal/customize-settings/color-schemes
    """
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    bold_red = "\x1b[31;21m"
    red = "\x1b[31;1m"
    grey2 = "\x1b[1;30m"
    white = "\x1b[1;37m"
    reset = "\x1b[0m"
    cyan = "\x1b[1;36m"
    purple = "\x1b[35m"

    FORMATS = {
        logging.DEBUG: green,
        logging.INFO: cyan,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self, text_format, date_format):
        self.text_format = text_format
        self.date_format = date_format

    def format(self, record):
        log_fmt = self.text_format.format(
            level_color=self.FORMATS.get(record.levelno),
            time_color=self.grey2, msg_color=self.white, 
            path_color=self.purple)

        formatter = logging.Formatter(log_fmt, datefmt=self.date_format)
        return formatter.format(record)


class LoggerManager:
    """
    Logger class
    For more documents, look into https://docs.python.org/3/library/logging.html
    
    Usage:
        from modules.logger import LoggerManager
        LOGGER = LoggerManager.init_logger(__name__)

    """
    current_dir = os.path.abspath(os.getcwd())
    filename = f'{current_dir}/log.txt'
    date_format = '%d-%m-%y %H:%M:%S'
    message_format = '[%(asctime)s][%(pathname)s::%(lineno)d][%(levelname)s]: %(message)s'
    color_message_format = '{time_color}[%(asctime)s]\x1b[0m{path_color}[%(pathname)s::%(lineno)d]\x1b[0m{level_color}[%(levelname)s]\x1b[0m: {msg_color}%(message)s\x1b[0m'
    level = logging.INFO        

    @staticmethod
    def init_logger(name):
        # Init logger
        logger = logging.getLogger(name)
        logger.setLevel(LoggerManager.level)

        # Create handlers
        handlers = LoggerManager.init_handlers()

        # Add handlers
        LoggerManager.add_handlers(logger, handlers=handlers)
        return logger

    @staticmethod
    def init_handlers():
        # Create one file logger and one stream logger
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(LoggerManager.filename)
        
        # Create formatters and add it to handlers
        format = logging.Formatter(LoggerManager.message_format, datefmt=LoggerManager.date_format)
        custom_format = CustomFormatter(LoggerManager.color_message_format, date_format=LoggerManager.date_format)
        stream_handler.setFormatter(custom_format)
        file_handler.setFormatter(format)
        
        return stream_handler, file_handler

    @staticmethod
    def add_handlers(logger, handlers):
        # Add handlers to the logger
        for handler in handlers:
            logger.addHandler(handler)

    @staticmethod
    def set_debug_mode(toggle="off"):
        if toggle == "on":
            LoggerManager.level = logging.DEBUG
        else:
            LoggerManager.level = logging.INFO