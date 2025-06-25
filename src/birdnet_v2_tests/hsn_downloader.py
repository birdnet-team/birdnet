import shutil
import subprocess
import time
from pathlib import Path
from zipfile import BadZipFile

URL = "https://zenodo.org/records/7525805/files/soundscape_data.zip?download=1"
DEST_DIR = Path("test-dataset/HSN")
ZIP_FILE = DEST_DIR / "soundscape_data.zip"


def download_with_wget(url: str, dest: Path, *, retries: int = 3) -> float:
  """Lädt *url* mit **wget** nach *dest* und gibt die Dauer in Sekunden zurück.

  Zeigt den gewohnten wget‑Fortschrittsbalken an und versucht es bei Fehlern
  (HTTP‑Fehler, Netzabbrüche …) bis zu *retries*‑mal erneut.
  """
  dest.parent.mkdir(parents=True, exist_ok=True)

  for attempt in range(1, retries + 1):
    start = time.perf_counter()
    try:
      subprocess.run(
        [
          "wget",
          "--progress=bar:force:noscroll",  # einzeiliger Balken
          "--timeout=15",
          "--tries=1",  # internes wget‑Retry wollen wir hier nicht
          "-O",
          str(dest),
          url,
        ],
        check=True,
      )
      return time.perf_counter() - start  # Erfolg ⇒ Dauer zurückgeben
    except subprocess.CalledProcessError as exc:
      print(f"wget‑Versuch {attempt} fehlgeschlagen: {exc}")
      if attempt == retries:
        raise  # nach letztem Versuch Exception weiterreichen
      time.sleep(2**attempt)  # exponentielles Back‑off


def extract_archive(archive: Path, target: Path) -> None:
  """Entpackt *archive* in *target* (unterstützt zip, tar.* u. a.)."""
  target.mkdir(parents=True, exist_ok=True)
  try:
    shutil.unpack_archive(str(archive), str(target))
  except (BadZipFile, shutil.ReadError) as exc:
    raise RuntimeError(f"Entpacken fehlgeschlagen: {exc}")


def get_hsn_file_paths() -> list[Path]:
  """Gibt eine Liste der Pfade zu den Dateien im Zielverzeichnis zurück."""
  return list(DEST_DIR.glob("**/*.flac"))


def download_and_extract_hsn_data() -> None:
  try:
    if ZIP_FILE.exists():
      print(f"Verwende vorhandenes Archiv: {ZIP_FILE}")
      duration = 0.0
    else:
      print("Starte Download mit wget …")
      duration = download_with_wget(URL, ZIP_FILE)
      size_mb = ZIP_FILE.stat().st_size / (1024**2)
      if duration > 0:
        speed = size_mb / duration
        print(
          f"Download abgeschlossen: {size_mb:.2f} MiB in {duration:.1f} s "
          f"({speed:.2f} MiB/s)"
        )

    print("Entpacke Archiv …")
    extract_archive(ZIP_FILE, DEST_DIR)
    print("Fertig.")
  except Exception as e:
    print(f"Fehler: {e}")


if __name__ == "__main__":
  res = get_hsn_file_paths()
  print(len(res))
