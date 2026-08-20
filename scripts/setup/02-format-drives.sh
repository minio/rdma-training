#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Format every data NVMe on this storage node with XFS and mount it at
# /mnt/data<N>, then persist the mounts in /etc/fstab by label.
#
#   DESTRUCTIVE. Every matching drive is wiped.
#
# Data drives are identified by model string so the boot device (Dell BOSS-N1)
# can never be selected. Override with DRIVE_MODEL= if your hardware differs.
#
# Usage:
#   ./02-format-drives.sh              # dry run: list what would be formatted
#   ./02-format-drives.sh --apply      # actually format
#
set -euo pipefail

DRIVE_MODEL="${DRIVE_MODEL:-SDS6BA176PSP9X3}"
MOUNT_BASE="${MOUNT_BASE:-/mnt}"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

mapfile -t DRIVES < <(
    lsblk -dn -o NAME,MODEL |
        awk -v m="$DRIVE_MODEL" '$0 ~ m {print $1}' |
        sort -V
)

if [[ ${#DRIVES[@]} -eq 0 ]]; then
    echo "no drives matching model '$DRIVE_MODEL' found" >&2
    exit 1
fi

echo "== $(hostname): ${#DRIVES[@]} data drives matching '$DRIVE_MODEL' =="
for i in "${!DRIVES[@]}"; do
    printf '  %-12s -> %s/data%d\n' "/dev/${DRIVES[$i]}" "$MOUNT_BASE" "$((i + 1))"
done

# Refuse to touch anything that currently holds the root filesystem.
root_src=$(findmnt -no SOURCE / | sed 's/p\?[0-9]*$//')
for d in "${DRIVES[@]}"; do
    if [[ "/dev/$d" == "$root_src" ]]; then
        echo "REFUSING: /dev/$d backs the root filesystem" >&2
        exit 1
    fi
done

if [[ $APPLY -ne 1 ]]; then
    echo
    echo "dry run only. re-run with --apply to format (THIS WIPES THE DRIVES)."
    exit 0
fi

echo
echo "== formatting =="
for i in "${!DRIVES[@]}"; do
    dev="/dev/${DRIVES[$i]}"
    n=$((i + 1))
    label="data${n}"
    mnt="${MOUNT_BASE}/data${n}"

    # A previous run may have left it mounted.
    if mountpoint -q "$mnt"; then
        sudo umount "$mnt"
    fi

    sudo wipefs -a "$dev" >/dev/null
    # -K: skip discard, much faster on these drives and unnecessary for a benchmark.
    # -L: label, so fstab entries survive NVMe enumeration order changes.
    sudo mkfs.xfs -f -K -L "$label" "$dev" >/dev/null
    sudo mkdir -p "$mnt"
    sudo mount -o noatime,nodiratime "LABEL=$label" "$mnt"

    # AIStor writes into a subdirectory of the mount, never the mount root, so a
    # missing/unmounted drive can never be silently written to the boot disk.
    sudo mkdir -p "$mnt/minio"
    sudo chown -R minio:minio "$mnt"

    sudo sed -i "\|LABEL=$label |d" /etc/fstab
    echo "LABEL=$label $mnt xfs noatime,nodiratime,defaults 0 2" |
        sudo tee -a /etc/fstab >/dev/null

    printf '  %-12s %-14s ok\n' "$dev" "$mnt"
done

echo
echo "== result =="
findmnt -no SOURCE,TARGET,FSTYPE,SIZE --list | grep -E "${MOUNT_BASE}/data[0-9]+ " | sort -V -k2
echo "mounted: $(findmnt -no TARGET --list | grep -cE "${MOUNT_BASE}/data[0-9]+$") drives"
