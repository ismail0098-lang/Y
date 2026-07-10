#!/bin/bash
set -e

# Check if run as root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "[!] Please run this script with sudo:"
  echo "    sudo ./install.sh"
  exit 1
fi

echo "[+] Installing Y ShadowPlay HUD globally..."

# Compile the package if it hasn't been built yet
if [ ! -f shadowplay/shadowplay ]; then
    echo "[-] shadowplay executable not found. Compiling first..."
    cargo build --release
    ./target/release/Y shadowplay/shadowplay.ysu
fi

# Create install directories
mkdir -p /usr/local/bin
mkdir -p /usr/local/share/Y_ShadowPlay
mkdir -p /usr/share/applications

# Copy executable and hardware profile
rm -f /usr/local/share/Y_ShadowPlay/shadowplay
cp shadowplay/shadowplay /usr/local/share/Y_ShadowPlay/
cp .ysu_hw_profile /usr/local/share/Y_ShadowPlay/ || true

# Create global wrapper script in /usr/local/bin
cat << 'EOF' > /usr/local/bin/y-shadowplay
#!/bin/bash
cd /usr/local/share/Y_ShadowPlay
exec ./shadowplay "$@"
EOF
chmod +x /usr/local/bin/y-shadowplay

# Create global desktop entry
cat << 'EOF' > /usr/share/applications/Y_ShadowPlay.desktop
[Desktop Entry]
Type=Application
Name=Y ShadowPlay HUD
Comment=Hardware-Sentient Screen Recorder Driven by Y-Lang
Exec=/usr/local/bin/y-shadowplay
Icon=video-display
Terminal=true
Categories=Utility;AudioVideo;Recorder;
EOF

chmod +x /usr/share/applications/Y_ShadowPlay.desktop

echo "=========================================================="
echo " [OK] ShadowPlay successfully installed globally!"
echo "=========================================================="
echo " You can now run it from anywhere using the command:"
echo "    y-shadowplay"
echo " Or find it in your KDE Plasma application menu!"
echo "=========================================================="
