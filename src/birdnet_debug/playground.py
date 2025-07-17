
import numpy as np

# Testdaten erstellen
n = 1_000_000
file_paths = [f"file_{i:06d}.wav" for i in range(n)]
start_times = np.random.random(n).astype(np.float64)
end_times = start_times + 3.0
species_names = [f"Species_{i % 100:03d}" for i in range(n)]
confidences = np.random.random(n).astype(np.float32)

# 1. Dictionary mit separaten Arrays
dict_arrays = {
  "file_path": np.array(file_paths),
  "start_time": start_times,
  "end_time": end_times,
  "species_name": np.array(species_names),
  "confidence": confidences,
}

# 2. Strukturiertes Array
dtype = [
  ("file_path", dict_arrays["file_path"].dtype),
  ("start_time", dict_arrays["start_time"].dtype),
  ("end_time", dict_arrays["end_time"].dtype),
  ("species_name", dict_arrays["species_name"].dtype),
  ("confidence", dict_arrays["confidence"].dtype),
]
structured = np.empty(n, dtype=dtype)
structured["file_path"] = dict_arrays["file_path"]
structured["start_time"] = dict_arrays["start_time"]
structured["end_time"] = dict_arrays["end_time"]
structured["species_name"] = dict_arrays["species_name"]
structured["confidence"] = dict_arrays["confidence"]

# Memory-Verbrauch messen
dict_size = sum(arr.nbytes for arr in dict_arrays.values())
structured_size = structured.nbytes

print(f"Dictionary Arrays: {dict_size / 1024**2:.1f} MB")
print(f"Structured Array:  {structured_size / 1024**2:.1f} MB")
print(f"Verhältnis: {structured_size / dict_size:.2f}x")
