#!/usr/bin/env bash
# convert_audio.sh ─ wandelt example.wav in gängige Audio-Formate um
# Benötigt: FFmpeg (>= 4.x) im PATH

# Aufruf:  ./convert_audio.sh [input.wav] [ausgabe-ordner]
# src/birdnet_v2_tests/create_audio_formats.sh example/soundscape.wav src/birdnet_v2_tests/audio_formats

set -euo pipefail

# ------------------------------------------------------------ #
# Eingabe- & Ausgabe-Pfad bestimmen                            #
# ------------------------------------------------------------ #
INFILE="${1:-example.wav}"
OUTDIR="${2:-$(dirname "$INFILE")}"

if [[ ! -f "$INFILE" ]]; then
  echo "❌  Eingabedatei '$INFILE' nicht gefunden." >&2
  exit 1
fi

mkdir -p "$OUTDIR"
BASENAME="$(basename "${INFILE%.*}")"   # example

# Helper-Funktion
encode() {
  echo -e "\033[1;36mffmpeg $*\033[0m"
  ffmpeg -loglevel error -y "$@"
}

# ------------------------------------------------------------ #
# Konvertierungen                                              #
# ------------------------------------------------------------ #
# MP3 – CBR 192 kb/s
encode -i "$INFILE" -c:a libmp3lame -b:a 192k       "$OUTDIR/$BASENAME.mp3"

# AAC (.m4a) – 192 kb/s
encode -i "$INFILE" -c:a aac -b:a 192k              "$OUTDIR/$BASENAME.m4a"

# Ogg/Vorbis – VBR q5 (~192 kb/s)
encode -i "$INFILE" -c:a libvorbis -q:a 5           "$OUTDIR/$BASENAME.ogg"

# Opus – 160 kb/s
encode -i "$INFILE" -c:a libopus -b:a 160k          "$OUTDIR/$BASENAME.opus"

# FLAC – lossless, Kompression 5
encode -i "$INFILE" -c:a flac -compression_level 5  "$OUTDIR/${BASENAME}.flac"

# WAV 24-bit PCM
encode -i "$INFILE" -c:a pcm_s24le                  "$OUTDIR/${BASENAME}_24bit.wav"

# WAV A-Law 8 kHz
encode -i "$INFILE" -ar 8000 -c:a pcm_alaw          "$OUTDIR/${BASENAME}_alaw.wav"

# WAV µ-Law 8 kHz
encode -i "$INFILE" -ar 8000 -c:a pcm_mulaw         "$OUTDIR/${BASENAME}_ulaw.wav"

echo -e "\033[1;32m✔️  Alle Konvertierungen abgeschlossen.\033[0m"
