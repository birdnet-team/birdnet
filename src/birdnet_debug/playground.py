import numpy as np

res = np.array([683], dtype=np.float16)
x = res * 3.0
print(x)

print(isinstance(3, float))


import numbers
if not isinstance(3, numbers.Number):