#!/bin/bash
set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/kanji_data"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "=== Kanji Practice Board: Full Database Installer ==="
echo "[-] Accessing data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 1. Download KanjiVG paths (all 6,000+ characters)
if [ ! -f "kanjivg.xml.gz" ]; then
    echo "[-] Downloading KanjiVG stroke database..."
    curl -L -o kanjivg.xml.gz "https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz"
else
    echo "[+] kanjivg.xml.gz already exists, skipping download."
fi

# 2. Download KanjiDic meanings & readings
if [ ! -f "kanjidic2.xml.gz" ]; then
    echo "[-] Downloading KANJIDIC2 dictionary..."
    curl -L -o kanjidic2.xml.gz "https://www.edrdg.org/kanjidic/kanjidic2.xml.gz"
else
    echo "[+] kanjidic2.xml.gz already exists, skipping download."
fi

# 3. Run the compiler script
echo "[-] Compiling binary database & metadata JSON..."
python3 download_and_pack.py

echo "=== Installation Successful! ==="
echo "[+] Packed database files copied to: $FRONTEND_DIR"
echo "[+] Please refresh your browser page at http://localhost:8000"
