import birdnet


def test_issue_29() -> None:
  target = "example/soundscape.wav"
  top_k = None
  batch_size = 1
  prefetch_ratio = 3
  overlap_duration_s = 0.0
  bandpass_fmin = 0
  bandpass_fmax = 15000
  sigmoid_sensitivity = 1.0
  speed = 1.0
  default_confidence_threshold = 0.25
  custom_species_list = None
  progress_callback = None
  show_stats = "progress"
  n_workers = None
  n_producers = 1
  apply_sigmoid = True

  model = birdnet.load(
    "acoustic",
    "2.4",
    "tf",
  )

  model.predict(
    target,
    top_k=top_k,
    batch_size=batch_size,
    prefetch_ratio=prefetch_ratio,
    overlap_duration_s=overlap_duration_s,
    bandpass_fmin=bandpass_fmin,
    bandpass_fmax=bandpass_fmax,
    sigmoid_sensitivity=sigmoid_sensitivity,
    speed=speed,
    default_confidence_threshold=default_confidence_threshold,
    custom_species_list=custom_species_list,
    progress_callback=progress_callback,
    show_stats=show_stats,
    n_workers=n_workers,
    n_producers=n_producers,
    apply_sigmoid=apply_sigmoid,
  )
