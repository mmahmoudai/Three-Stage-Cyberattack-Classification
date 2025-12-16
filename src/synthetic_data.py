from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticSpec:
    protocol_values: List[str]
    benign_label: str
    attack_label: str
    attack_classes: List[str]
    numeric_feature_names: List[str]


def default_synthetic_spec(benign_label: str = "Benign", attack_label: str = "Attack") -> SyntheticSpec:
    protocol_values = ["TCP", "UDP", "LDAP", "NetBIOS", "ICMP", "DNS", "HTTP", "HTTPS"]
    attack_classes = [
        "SYN Flood",
        "UDP Flood",
        "LDAP Attack",
        "NetBIOS Attack",
        "PortScan",
        "Botnet",
        "DDoS",
        "Brute Force",
        "SQL Injection",
        "XSS",
        "Infiltration",
        "Malware",
        "Ransomware",
        "MITM",
        "Data Exfiltration",
        "Heartbleed",
    ]

    top_named = [
        "SYN Flag Count",
        "Flow Duration",
        "Flow Packets/s",
        "Flow Bytes/s",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Protocol",
        "Fwd IAT Mean",
        "Packet Length Variance",
        "ACK Flag Count",
        "Subflow Fwd Packets",
        "Init_Win_bytes_forward",
        "Bwd IAT Mean",
        "Packet Length Mean",
        "Active Mean",
        "Idle Mean",
        "Fwd Header Length",
        "Bwd Packets/s",
        "URG Flag Count",
        "Subflow Bwd Bytes",
    ]

    filler = [f"feat_{i}" for i in range(80 - (len(top_named) - 1))]
    numeric_feature_names = [n for n in top_named if n != "Protocol"] + filler

    return SyntheticSpec(
        protocol_values=protocol_values,
        benign_label=benign_label,
        attack_label=attack_label,
        attack_classes=attack_classes,
        numeric_feature_names=numeric_feature_names,
    )


def generate_synthetic_dataset(
    path: str,
    n_rows: int,
    seed: int,
    protocol_col: str,
    class_col: str,
    category_col: str,
    benign_label: str,
    attack_label: str,
    spec: Optional[SyntheticSpec] = None,
) -> pd.DataFrame:
    if spec is None:
        spec = default_synthetic_spec(benign_label=benign_label, attack_label=attack_label)

    rng = np.random.default_rng(seed)

    class_probs = np.array([0.7] + [0.3 / len(spec.attack_classes)] * len(spec.attack_classes), dtype=float)
    classes = [benign_label] + list(spec.attack_classes)
    y_class = rng.choice(classes, size=n_rows, p=class_probs)
    y_category = np.where(y_class == benign_label, benign_label, attack_label)

    protocol = np.empty(n_rows, dtype=object)
    protocol[:] = rng.choice(spec.protocol_values, size=n_rows)

    protocol_map = {
        "SYN Flood": "TCP",
        "UDP Flood": "UDP",
        "LDAP Attack": "LDAP",
        "NetBIOS Attack": "NetBIOS",
    }
    for attack, proto in protocol_map.items():
        m = y_class == attack
        protocol[m] = proto

    data = {}

    flow_duration = rng.lognormal(mean=10.0, sigma=1.0, size=n_rows)
    syn_count = rng.poisson(lam=2.0, size=n_rows).astype(float)
    pkt_rate = rng.lognormal(mean=2.0, sigma=0.6, size=n_rows)
    byte_rate = pkt_rate * rng.lognormal(mean=4.0, sigma=0.4, size=n_rows)

    m_syn = y_class == "SYN Flood"
    syn_count[m_syn] = rng.poisson(lam=80.0, size=m_syn.sum()).astype(float)
    pkt_rate[m_syn] = rng.lognormal(mean=5.0, sigma=0.4, size=m_syn.sum())
    flow_duration[m_syn] = rng.lognormal(mean=8.5, sigma=0.7, size=m_syn.sum())

    m_udp = y_class == "UDP Flood"
    pkt_rate[m_udp] = rng.lognormal(mean=5.2, sigma=0.4, size=m_udp.sum())
    byte_rate[m_udp] = pkt_rate[m_udp] * rng.lognormal(mean=3.8, sigma=0.4, size=m_udp.sum())

    fwd_len_mean = rng.lognormal(mean=4.0, sigma=0.3, size=n_rows)
    bwd_len_mean = rng.lognormal(mean=4.2, sigma=0.3, size=n_rows)

    fwd_iat_mean = rng.lognormal(mean=1.5, sigma=0.5, size=n_rows)
    bwd_iat_mean = rng.lognormal(mean=1.6, sigma=0.5, size=n_rows)

    ack_count = rng.poisson(lam=10.0, size=n_rows).astype(float)
    urg_count = rng.poisson(lam=0.5, size=n_rows).astype(float)

    pkt_len_var = rng.lognormal(mean=3.0, sigma=0.6, size=n_rows)
    pkt_len_mean = rng.lognormal(mean=4.1, sigma=0.25, size=n_rows)

    active_mean = rng.lognormal(mean=2.5, sigma=0.5, size=n_rows)
    idle_mean = rng.lognormal(mean=2.2, sigma=0.6, size=n_rows)

    subflow_fwd_pkts = rng.poisson(lam=20.0, size=n_rows).astype(float)
    subflow_bwd_bytes = rng.lognormal(mean=6.0, sigma=0.8, size=n_rows)

    init_win_fwd = rng.lognormal(mean=10.0, sigma=0.7, size=n_rows)

    bwd_pkts_rate = rng.lognormal(mean=1.8, sigma=0.6, size=n_rows)
    fwd_header_len = rng.lognormal(mean=3.6, sigma=0.4, size=n_rows)

    data.update(
        {
            "Flow Duration": flow_duration,
            "SYN Flag Count": syn_count,
            "Flow Packets/s": pkt_rate,
            "Flow Bytes/s": byte_rate,
            "Fwd Packet Length Mean": fwd_len_mean,
            "Bwd Packet Length Mean": bwd_len_mean,
            "Fwd IAT Mean": fwd_iat_mean,
            "Bwd IAT Mean": bwd_iat_mean,
            "ACK Flag Count": ack_count,
            "URG Flag Count": urg_count,
            "Packet Length Variance": pkt_len_var,
            "Packet Length Mean": pkt_len_mean,
            "Active Mean": active_mean,
            "Idle Mean": idle_mean,
            "Subflow Fwd Packets": subflow_fwd_pkts,
            "Subflow Bwd Bytes": subflow_bwd_bytes,
            "Init_Win_bytes_forward": init_win_fwd,
            "Bwd Packets/s": bwd_pkts_rate,
            "Fwd Header Length": fwd_header_len,
        }
    )

    for name in spec.numeric_feature_names:
        if name in data:
            continue
        data[name] = rng.normal(loc=0.0, scale=1.0, size=n_rows)

    df = pd.DataFrame(data)

    nan_mask = rng.random(df.shape) < 0.01
    df = df.mask(nan_mask)

    df[protocol_col] = protocol
    df[class_col] = y_class
    df[category_col] = y_category

    df.to_csv(path, index=False)
    return df
