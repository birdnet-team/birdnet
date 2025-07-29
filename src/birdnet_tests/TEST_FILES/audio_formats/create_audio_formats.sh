#!/usr/bin/env bash
# Benötigt: FFmpeg (>= 4.x) im PATH

# Aufruf:  ./convert_audio.sh [input.wav] [ausgabe-ordner]
# src/birdnet_tests/TEST_FILES/audio_formats/create_audio_formats.sh example/soundscape.wav src/birdnet_tests/TEST_FILES/audio_formats

set -euo pipefail

INFILE="${1:-example.wav}"
OUTDIR="${2:-$(dirname "$INFILE")}"

if [[ ! -f "$INFILE" ]]; then
  echo "❌  Eingabedatei '$INFILE' nicht gefunden." >&2
  exit 1
fi

mkdir -p "$OUTDIR"
BASENAME="$(basename "${INFILE%.*}")"   # example

encode() {
  echo -e "\033[1;36mffmpeg $*\033[0m"
  ffmpeg -loglevel error -y "$@"
}

cp "$INFILE" "$OUTDIR/$BASENAME.wav"

#  stereo
encode -i "$INFILE" -filter_complex "[0:a]pan=stereo|c0=c0|c1=c0" "$OUTDIR/${BASENAME}_stereo.wav"

# three channels
encode -i "$INFILE" -filter_complex "[0:a]pan=3c|c0=c0|c1=c0|c2=c0" "$OUTDIR/${BASENAME}_3ch.wav"

# verlustbehaftet (komprimiert)                                #
encode -i "$INFILE" -c:a libmp3lame    -b:a 192k  "$OUTDIR/$BASENAME.mp3"   # MP3
encode -i "$INFILE" -c:a aac           -b:a 192k  "$OUTDIR/$BASENAME.aac"   # AAC „roher“ Stream
encode -i "$INFILE" -c:a aac           -b:a 192k  "$OUTDIR/$BASENAME.m4a"   # AAC im MP4-Container
encode -i "$INFILE" -c:a libvorbis     -q:a 5     "$OUTDIR/$BASENAME.ogg"   # Ogg/Vorbis
encode -i "$INFILE" -c:a libopus       -b:a 160k  "$OUTDIR/$BASENAME.opus"  # Opus
encode -i "$INFILE" -c:a wmav2         -b:a 192k  -f asf "$OUTDIR/$BASENAME.wma"  # WMA-2 (ASF)

# verlustfrei / PCM                                            #
encode -i "$INFILE" -c:a flac                -compression_level 5 "$OUTDIR/${BASENAME}.flac"          # FLAC
encode -i "$INFILE" -c:a pcm_s24le                          "$OUTDIR/${BASENAME}_24bit.wav"          # WAV 24-bit
encode -i "$INFILE" -ar 8000 -c:a pcm_alaw                  "$OUTDIR/${BASENAME}_alaw.wav"           # WAV A-Law
encode -i "$INFILE" -ar 8000 -c:a pcm_mulaw                 "$OUTDIR/${BASENAME}_ulaw.wav"           # WAV µ-Law
encode -i "$INFILE" -c:a pcm_s16be -f aiff                  "$OUTDIR/${BASENAME}.aiff"               # AIFF
encode -i "$INFILE" -c:a adpcm_ima_qt -f aiff "$OUTDIR/${BASENAME}.aifc" # AIFC
encode -i "$INFILE" -c:a pcm_s16be -f au                    "$OUTDIR/${BASENAME}.au"                 # AU / Sun

echo -e "\033[1;32m✔️  Konvertierung abgeschlossen.\033[0m"