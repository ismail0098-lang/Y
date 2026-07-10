#!/bin/bash
set -e

# Check if run as root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "[!] Please run this script with sudo:"
  echo "    sudo ./uninstall.sh"
  exit 1
fi

echo "[+] Uninstalling Y ShadowPlay HUD globally..."

# Remove global files and directories
rm -f /usr/local/bin/y-shadowplay
rm -rf /usr/local/share/Y_ShadowPlay
rm -f /usr/share/applications/Y_ShadowPlay.desktop

# Clean up local build artifact
rm -f shadowplay/shadowplay

echo "=========================================================="
echo " [OK] ShadowPlay successfully uninstalled!"
echo "=========================================================="
