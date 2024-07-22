import logging


def pytest_configure():
  loggers = {"numba"}
  for l in loggers:
    logger = logging.getLogger(l)
    logger.disabled = True
    logger.propagate = False
