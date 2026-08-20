#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Apply the host side of a lossless RoCE configuration: PFC on priority 3,
# DSCP 26 -> priority 3, ECN on priority 3, on every RoCE netdev.
#
# This is HOST-ONLY. Losslessness must be configured identically on every switch
# in the path or it does nothing — a host-only config cannot stop a switch from
# dropping. See docs/troubleshooting.md, and MinIO's published AIStor RDMA
# deployment guidance for the switch-side ("lossless fabric") requirements.
#
# Run on every storage node AND on the training client. Not reboot-persistent
# unless --install-service is passed.
set -euo pipefail

INSTALL_SERVICE=0
[[ "${1:-}" == "--install-service" ]] && INSTALL_SERVICE=1

# Every RoCE netdev on this host, derived from the ibverbs devices so we never
# guess at interface names.
mapfile -t DEVS < <(
    for d in /sys/class/infiniband/*/device/net/*; do
        [[ -e "$d" ]] && basename "$d"
    done | sort -u
)

if [[ ${#DEVS[@]} -eq 0 ]]; then
    echo "no RoCE netdevs found" >&2
    exit 1
fi

echo "== $(hostname): applying lossless RoCE host QoS to: ${DEVS[*]} =="

for dev in "${DEVS[@]}"; do
    echo "-- $dev --"
    # trust dscp: honour the DSCP the application marks, so the class survives
    # routed (L3) RoCEv2 hops. Without this the NIC would use PCP instead.
    sudo mlnx_qos -i "$dev" --trust dscp >/dev/null 2>&1 || echo "   warn: --trust dscp failed"
    # PFC on priority 3 only. Priority 3 / DSCP 26 is the NVIDIA convention.
    sudo mlnx_qos -i "$dev" --pfc 0,0,0,1,0,0,0,0 >/dev/null 2>&1 || echo "   warn: --pfc failed"
    sudo mlnx_qos -i "$dev" --dscp2prio set,26,3 >/dev/null 2>&1 || echo "   warn: --dscp2prio failed"
    # Map priority 3 onto the lossless buffer.
    sudo mlnx_qos -i "$dev" --prio2buffer 0,0,0,1,0,0,0,0 >/dev/null 2>&1 || echo "   warn: --prio2buffer failed"
    # DCQCN: reaction point + notification point on priority 3.
    echo 1 | sudo tee "/sys/class/net/$dev/ecn/roce_rp/enable/3" >/dev/null 2>&1 || true
    echo 1 | sudo tee "/sys/class/net/$dev/ecn/roce_np/enable/3" >/dev/null 2>&1 || true

    printf '   pfc:  '
    sudo mlnx_qos -i "$dev" 2>/dev/null | grep -A1 'enabled' | tail -1 || echo "?"
    printf '   trust: %s\n' "$(sudo mlnx_qos -i "$dev" 2>/dev/null | grep -i 'trust' | head -1 | xargs || echo '?')"
    printf '   ecn rp/np prio3: %s/%s\n' \
        "$(cat "/sys/class/net/$dev/ecn/roce_rp/enable/3" 2>/dev/null || echo '?')" \
        "$(cat "/sys/class/net/$dev/ecn/roce_np/enable/3" 2>/dev/null || echo '?')"
done

if [[ $INSTALL_SERVICE -eq 1 ]]; then
    echo
    echo "== installing roce-lossless.service (re-applies at boot) =="
    sudo cp "$0" /usr/local/sbin/roce-lossless.sh
    sudo chmod +x /usr/local/sbin/roce-lossless.sh
    sudo tee /etc/systemd/system/roce-lossless.service >/dev/null <<'EOF'
[Unit]
Description=RoCE lossless host QoS (PFC prio3 + DSCP26 + ECN)
After=network-online.target openibd.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/roce-lossless.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable roce-lossless.service >/dev/null
    echo "enabled: $(systemctl is-enabled roce-lossless.service)"
fi

echo
echo "NOTE: switch-side PFC/ECN is NOT configured by this script. Verify"
echo "empirically with scripts/setup/06-check-fabric.sh under load."
