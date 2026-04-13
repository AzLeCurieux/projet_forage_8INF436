#!/bin/bash
# Lancement du dashboard Dash depuis la racine du projet
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Erreur : venv introuvable. Lancez d'abord : python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo "Dashboard disponible sur http://127.0.0.1:8050"
python dashboard/app.py
