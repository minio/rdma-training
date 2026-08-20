#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Add /etc/hosts aliases for the AIStor nodes.
#
# Why aliases rather than real hostnames: MinIO expands one server-pool argument
# with an ellipsis (`http://host{1...2}/...`), so the node names must be
# contiguous. Our own two nodes were named coe02 and coe04, and `coe{02...04}`
# would have expanded to coe02, coe03, coe04 -- pulling in a third machine we do
# not own. Writing two separate arguments instead would create two single-node
# pools, so erasure sets would not stripe across nodes and we would lose half the
# aggregate throughput. Contiguous aliases give us one pool spanning both nodes.
#
# Set these for YOUR hardware. Rail 0 / rail 1 are the two NIC ports per node;
# if your nodes are single-railed, leave the RAIL1 values empty.
#
#   AISTOR1_RAIL0=10.0.0.11 AISTOR2_RAIL0=10.0.0.12 \
#   AISTOR1_RAIL1=10.0.1.11 AISTOR2_RAIL1=10.0.1.12 ./03-configure-hosts.sh
#
# Run on every storage node AND on the training client.
set -euo pipefail

AISTOR1_RAIL0="${AISTOR1_RAIL0:-}"
AISTOR2_RAIL0="${AISTOR2_RAIL0:-}"
AISTOR1_RAIL1="${AISTOR1_RAIL1:-}"
AISTOR2_RAIL1="${AISTOR2_RAIL1:-}"

if [[ -z "$AISTOR1_RAIL0" || -z "$AISTOR2_RAIL0" ]]; then
    cat >&2 <<'MSG'
ERROR: set the node addresses for your own cluster, e.g.

  AISTOR1_RAIL0=10.0.0.11 AISTOR2_RAIL0=10.0.0.12 \
  AISTOR1_RAIL1=10.0.1.11 AISTOR2_RAIL1=10.0.1.12 ./03-configure-hosts.sh

AISTOR{1,2}_RAIL0 are required; the RAIL1 pair is optional (dual-rail NICs only).
MSG
    exit 2
fi

MARKER="# aistor-rdma-benchmark"

ENTRIES="$AISTOR1_RAIL0   aistor1 aistor1.rail0
$AISTOR2_RAIL0   aistor2 aistor2.rail0"
[[ -n "$AISTOR1_RAIL1" ]] && ENTRIES+="
$AISTOR1_RAIL1   aistor1.rail1"
[[ -n "$AISTOR2_RAIL1" ]] && ENTRIES+="
$AISTOR2_RAIL1   aistor2.rail1"

echo "== configuring /etc/hosts on $(hostname) =="

# Drop any previous block so this is idempotent.
sudo sed -i "/${MARKER}\$/d" /etc/hosts

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    echo "$line $MARKER" | sudo tee -a /etc/hosts >/dev/null
done <<<"$ENTRIES"

echo "-- resolved --"
for h in aistor1 aistor2 aistor1.rail1 aistor2.rail1; do
    addr="$(getent hosts "$h" | awk '{print $1}')"
    [[ -z "$addr" ]] && continue
    printf '  %-14s %s\n' "$h" "$addr"
done
