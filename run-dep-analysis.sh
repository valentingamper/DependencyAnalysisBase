#!/usr/bin/env bash

STRINGS=(
  "keycloak/keycloak"
  "apache/tomcat"
  "zaproxy/zaproxy"
  "Checkmk/checkmk"
  "cve-search/cve-search"
  "intelowlproject/IntelOwl"
  "yeti-platform/yeti"
  "smicallef/spiderfoot"
  "mitre/caldera"
)

PYTHON_SCRIPT="Dependency_analysis_notebook.py"

for s in "${STRINGS[@]}"; do
  echo "Rufe Python-Skript mit Argument: $s auf"
  python "$PYTHON_SCRIPT" "$s"
done
