#!/bin/bash
set -e

echo "[+] Compiling Y Language Compiler..."
cargo build --release

PORTABLE_FLAG=""
if [[ "$1" == "--portable" ]]; then
    PORTABLE_FLAG="--portable"
    echo "[+] Building portable release (no AVX/AVX-512 requirements)..."
fi

echo "[+] Compiling ShadowPlay Y-Lang source to native client..."
./target/release/Y shadowplay/shadowplay.ysu $PORTABLE_FLAG

echo "[+] Creating release structure..."
rm -rf Y_ShadowPlay
mkdir -p Y_ShadowPlay

# Copy executable and config
if [[ "$1" == "--portable" ]]; then
    # Strip the GNU property note so it can run on CPUs with lower ISA levels than the build host
    strip --remove-section=.note.gnu.property shadowplay/shadowplay || true
fi
cp shadowplay/shadowplay Y_ShadowPlay/
if [[ "$1" != "--portable" ]]; then
    cp .ysu_hw_profile Y_ShadowPlay/ || true
fi

# 1. Create a self-contained installer for the friend (no Rust/Y compiler needed)
cat << 'EOF' > Y_ShadowPlay/install.sh
#!/bin/bash
set -e

# Check if run as root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "[!] Please run this script with sudo:"
  echo "    sudo ./install.sh"
  exit 1
fi

echo "[+] Installing Y ShadowPlay HUD globally..."

# Create install directories
mkdir -p /usr/local/bin
mkdir -p /usr/local/share/Y_ShadowPlay
mkdir -p /usr/share/applications

# Copy executable and hardware profile
rm -f /usr/local/share/Y_ShadowPlay/shadowplay
cp shadowplay /usr/local/share/Y_ShadowPlay/
cp .ysu_hw_profile /usr/local/share/Y_ShadowPlay/ || true

# Create global wrapper script in /usr/local/bin
cat << 'INNER_EOF' > /usr/local/bin/y-shadowplay
#!/bin/bash
cd /usr/local/share/Y_ShadowPlay
exec ./shadowplay "$@"
INNER_EOF
chmod +x /usr/local/bin/y-shadowplay

# Create global desktop entry
cat << 'INNER_EOF' > /usr/share/applications/Y_ShadowPlay.desktop
[Desktop Entry]
Type=Application
Name=Y ShadowPlay HUD
Comment=Hardware-Sentient Screen Recorder Driven by Y-Lang
Exec=/usr/local/bin/y-shadowplay
Icon=video-display
Terminal=true
Categories=Utility;AudioVideo;Recorder;
INNER_EOF

chmod +x /usr/share/applications/Y_ShadowPlay.desktop

echo "=========================================================="
echo " [OK] ShadowPlay successfully installed globally!"
echo "=========================================================="
echo " You can now run it from anywhere using the command:"
echo "    y-shadowplay"
echo "=========================================================="
EOF
chmod +x Y_ShadowPlay/install.sh

# 2. Create a self-contained uninstaller for the friend
cat << 'EOF' > Y_ShadowPlay/uninstall.sh
#!/bin/bash
set -e

if [ "$EUID" -ne 0 ]; then
  echo "[!] Please run this script with sudo:"
  echo "    sudo ./uninstall.sh"
  exit 1
fi

echo "[+] Uninstalling Y ShadowPlay HUD..."
rm -f /usr/local/bin/y-shadowplay
rm -rf /usr/local/share/Y_ShadowPlay
rm -f /usr/share/applications/Y_ShadowPlay.desktop

echo "=========================================================="
echo " [OK] ShadowPlay successfully uninstalled!"
echo "=========================================================="
EOF
chmod +x Y_ShadowPlay/uninstall.sh

# 3. Create a README file for the friend
cat << 'EOF' > Y_ShadowPlay/README.txt
==========================================================
 Y ShadowPlay HUD - Pre-compiled Release Package
==========================================================

This package contains the pre-compiled binary for the Y ShadowPlay HUD.
Your system does not need Rust or the Y Compiler installed to run this.

Requirements:
-------------
- An X11 desktop session (KDE Plasma, GNOME, etc.)
- 'gpu-screen-recorder' (recommended for NVENC/VAAPI) or 'ffmpeg'
- 'pactl' (for desktop audio capture support)

Installation:
-------------
To install it globally on your system:
  1. Open a terminal in this directory.
  2. Run the installer:
       sudo ./install.sh
  3. You can now launch it from your desktop applications menu,
     or run it from any terminal with:
       y-shadowplay

Uninstallation:
---------------
To remove it from your system:
  sudo ./uninstall.sh
==========================================================
EOF

# Create tarball
echo "[+] Archiving into Y_ShadowPlay.tar.gz..."
tar -czf Y_ShadowPlay.tar.gz Y_ShadowPlay

echo "=========================================================="
echo " [OK] Portable ShadowPlay package created: Y_ShadowPlay.tar.gz"
echo "=========================================================="
echo " Share the 'Y_ShadowPlay.tar.gz' file with your friend."
echo " They just need to extract it and run: sudo ./install.sh"
echo "=========================================================="
