from birdnet.model_loader import load


def test_component():
  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  result = model.predict(
    20,
    50,
    week=1,
    min_confidence=0.03,
    half_precision=True,
  )
  assert result is not None
