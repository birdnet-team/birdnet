"""An exception class that only a subprocess under test may import.

Tests assert this module is absent from the parent's `sys.modules` after a
child raised the exception, so nothing in the parent may import it.
"""


class ForeignLoadError(Exception):
  pass
