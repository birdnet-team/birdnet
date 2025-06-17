import numpy as np


def test():
  thres = np.array(
    [
      [0.1, 0.2, np.inf],
      [0.1, 0.2, np.inf],
    ]
  )
  preds = np.array(
    [
      [0.01, 0.3, 0.4],
      [0.11, 0.02, 0],
    ]
  )

