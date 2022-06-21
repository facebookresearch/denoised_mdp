import sys
import os
import logging
import tqdm
import contextlib


__all__ = ['logger', 'configure', 'disable']


# logger = logging.getLogger(__name__)
logger = logging.getLogger()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa E722
            self.handleError(record)


class MultiLineFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style='%'):
        assert style == '%'
        super(MultiLineFormatter, self).__init__(fmt, datefmt, style)
        self.multiline_fmt = fmt

    def format(self, record):
        r"""
        This is mostly the same as logging.Formatter.format except for the splitlines() thing.
        This is done so (copied the code) to not make logging a bottleneck. It's not lots of code
        after all, and it's pretty straightforward.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        if '\n' in record.message:
            splitted = record.message.splitlines()
            output = self._fmt % dict(record.__dict__, message=splitted.pop(0))
            output += ' \n' + '\n'.join(
                self.multiline_fmt % dict(record.__dict__, message=line)
                for line in splitted
            )
        else:
            output = self._fmt % record.__dict__

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            output += ' \n'
            try:
                output += '\n'.join(
                    self.multiline_fmt % dict(record.__dict__, message=line)
                    for index, line in enumerate(record.exc_text.splitlines())
                )
            except UnicodeError:
                output += '\n'.join(
                    self.multiline_fmt % dict(record.__dict__, message=line)
                    for index, line
                    in enumerate(record.exc_text.decode(sys.getfilesystemencoding(), 'replace').splitlines())
                )
        return output


# this should replace `sys.excepthook`
# Shamelessly taken from https://stackoverflow.com/a/16993115
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logging.shutdown()


def configure(logging_file, log_level=logging.INFO, level_prefix='', prefix='', write_to_stdout=True):
    logger.setLevel(logging.INFO)  # set to info first to allow outputing in this function

    sys.excepthook = handle_exception  # automatically log uncaught errors

    handlers = []

    if write_to_stdout:
        handlers.append(TqdmLoggingHandler())

    delayed_logging = []  # log after we set the handlers with the nice formatter

    if logging_file is not None:
        delayed_logging.append((logging.info, 'Logging to {}'.format(logging_file)))
        if os.path.isfile(logging_file):
            delayed_logging.append((logging.warning, "Log file already exists, will append"))
        handlers.append(logging.FileHandler(logging_file))

    formatter = MultiLineFormatter("{}%(asctime)s [{}%(levelname)-5s]  %(message)s".format(prefix, level_prefix),
                                   "%Y-%m-%d %H:%M:%S")
    logger.handlers = []
    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    logger.setLevel(log_level)

    # flush cached message
    for fn, msg in delayed_logging:
        fn(msg)

    return logger


@contextlib.contextmanager
def disable(level):
    # disables any level leq to :attr:`level`
    prev_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(prev_level)
