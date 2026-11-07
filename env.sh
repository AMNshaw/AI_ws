#!/bin/bash

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
    source "$SCRIPT_DIR/$VENV_DIR/bin/activate"
    echo "[+] AI_ws environment activated."
else
    echo "[-] Error: $VENV_DIR not found. Please create it first."
fi
