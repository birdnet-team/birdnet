from typing import Literal

import numpy as np

# flag for "can be written to"
WRITABLE_FLAG = np.uint8(0)

WRITING_FLAG = np.uint8(1)

# flag for "can be read from"
READABLE_FLAG = np.uint8(2)

# flag for "busy", i.e., currently being processed
READING_FLAG = np.uint8(3)

# flag for "done"
DONE_FLAG = np.uint8(4)

DEVICES = Literal["CPU", "GPU"]
