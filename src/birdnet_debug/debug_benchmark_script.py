
from birdnet.benchmark_script import run_benchmark_from_args

if __name__ == "__main__":
  import logging


  # Set up logging
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  args = ["example/soundscape.wav", "/tmp/soundscape.csv", "-b", "tf", "-w", "1"]

  # Run the benchmark
  try:
    run_benchmark_from_args(args)
  except Exception as e:
    logger.error(f"Benchmark failed: {e}")
    raise
  else:
    logger.info("Benchmark completed successfully.")
