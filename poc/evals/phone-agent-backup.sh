#!/bin/sh
set -eu
# Stable state directory: never derive live data paths from the release symlink.
exec /opt/phone-agent/venv/bin/python \
  /opt/phone-agent/state/maintenance/production-backup-v1/production_backup.py backup \
  --state-dir /opt/phone-agent/state/runtime \
  --backup-dir /opt/phone-agent/state/backups
