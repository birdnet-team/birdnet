import numpy as np, tensorflow as tf, time, os

from birdnet_v2.local_data import get_local_model_root_dir

MODEL_DIR = get_local_model_root_dir("acoustic", "2.4", "pb") / "model"  # dein SavedModel-Ordner
BATCH     = np.empty((2, 48000*3), np.float32)  # exakt dieselbe Shape wie später

tf.debugging.set_log_device_placement(True)   # zeigt jedes Kernel-Mapping
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"      # sämtliche TF-Logs

model = tf.saved_model.load(MODEL_DIR)
print("== Signatures ==", model.signatures.keys())

basic = model.signatures.get("basic")
print("== basic input spec ==", basic.structured_input_signature)

t0 = time.perf_counter()
with tf.device("/device:GPU:0"):  # explizit auf CPU setzen, damit es nicht auf GPU läuft
  out = basic(inputs=BATCH)["scores"]
print("ms 1st run:", (time.perf_counter()-t0)*1000)
