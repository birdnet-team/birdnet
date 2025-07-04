from pathlib import Path

if __name__ == "__main__":
  url = "https://zenodo.org/records/7525805/files/soundscape_data.zip?download=1"
  destination = Path("test-dataset/HSN")

  destination.mkdir(parents=True, exist_ok=True)
  print(f"Downloading dataset from {url} to {destination}...")
  import requests

  response = requests.get(url, stream=True)
  if response.status_code == 200:
    with open(destination / "soundscape_data.zip", "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
    print("Download completed successfully.")
  else:
    print(f"Failed to download dataset. Status code: {response.status_code}")

  # unizp
  import zipfile

  with zipfile.ZipFile(destination / "soundscape_data.zip", "r") as zip_ref:
    zip_ref.extractall(destination)
  print(f"Extracted dataset to {destination}.")
