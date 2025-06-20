# birdnet/logging_utils.py
from __future__ import annotations

import logging
import sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

PKG_NAME = "birdnet"

def get_module_logger():
  return logging.getLogger(PKG_NAME)

# ------------------------------------------------------------ #
# 1) Im HAUPTprozess aufrufen → Listener starten
# ------------------------------------------------------------ #
def start_queue_listener(
  loglevel: str | int = "INFO",
  fmt: str = "%(asctime)s (%(processName)s) %(levelname).1s: %(message)s",
) -> tuple[Queue, QueueListener]:
  """
  Richtet einen QueueListener ein, der Records aus *allen* Prozessen
  entgegennimmt und in die Konsole (oder andere Handler) schreibt.

  Rückgabe:
      log_queue  – an Worker weiterreichen
      listener   – nach `join()` der Worker mit `.stop()` beenden
  """
  log_queue: Queue = Queue()

  console = logging.StreamHandler(sys.stdout)
  console.setFormatter(logging.Formatter(fmt))
  console.setLevel(loglevel)

  listener = QueueListener(log_queue, console)
  listener.start()
  return log_queue, listener


# ------------------------------------------------------------ #
# 2) In JEDEM Prozess (Haupt + Worker) aufrufen
# ------------------------------------------------------------ #
def enable_package_queue_logging(
  log_queue: Queue,
) -> None:
  """
  Hängt einen QueueHandler **nur** an den birdnet-Stamm-Logger.
  Unter-Logger propagieren automatisch dorthin.
  """
  pkg_log = logging.getLogger(PKG_NAME)

  # Doppeltes Anhängen vermeiden
  if any(
    isinstance(h, QueueHandler) and h.queue is log_queue for h in pkg_log.handlers
  ):
    return

  qh = QueueHandler(log_queue)
  qh.setLevel(pkg_log.level)  # respektiert Paket-Level
  pkg_log.addHandler(qh)

  # WICHTIG: Keine Weiterleitung an Root, damit wir Root nicht verändern
  pkg_log.propagate = False
