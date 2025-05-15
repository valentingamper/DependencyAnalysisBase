#!/usr/bin/env bash

# Array mit 7 unterschiedlichen Strings
STRINGS=(
  "keycloak/keycloak"
  "apache/tomcat"
  "zaproxy/zaproxy"
  "Checkmk/checkmk"
  "cve-search/cve-search"
  "intelowlproject/IntelOwl"
  "yeti-platform/yeti"
  "smicallef/spiderfoot"
  "SigmaHQ/sigma"
  "honeynet/honeyscanner"
  "mitre/caldera"
)

# Pfad zu deinem Python-Skript
PYTHON_SCRIPT="Dependency_analysis_notebook.py"

# For-Loop über alle Strings
for s in "${STRINGS[@]}"; do
  echo "Rufe Python-Skript mit Argument: $s auf"
  python "$PYTHON_SCRIPT" "$s"
done
