import logging


def pytest_configure():
  loggers = {"tensorflow", "absl", "urllib3"}
  for l in loggers:
    logger = logging.getLogger(l)
    logger.disabled = True
    logger.propagate = False

  # gpus = tf.config.list_physical_devices('GPU')
  # for gpu in gpus:
  #   try:
  #     tf.config.experimental.set_memory_growth(gpu, True)
  #   except RuntimeError as e:
  #     print(e)

  main_logger = logging.getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  console = logging.StreamHandler()
  console.setLevel(logging.DEBUG)
  main_logger.addHandler(console)
