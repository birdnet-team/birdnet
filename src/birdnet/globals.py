
import numpy as np

PKG_NAME = "birdnet"

# flag for "can be written to" = free
WRITABLE_FLAG = np.uint8(0)

# flag for "currently being written to"
WRITING_FLAG = np.uint8(1)

# flag for "can be read from" = preloaded
READABLE_FLAG = np.uint8(2)

# flag for "busy", i.e., currently being processed
READING_FLAG = np.uint8(3)

# flag for "done"
DONE_FLAG = np.uint8(4)

