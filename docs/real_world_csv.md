# Real-World CSV Log Normalization

Dashboard upload now accepts UNSW-NB15 raw rows and common production-style flow CSVs.

Supported schema families:

- Firewall or generic flow CSV: `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `bytes`, `packets`, `duration`
- NetFlow/IPFIX or cloud flow logs: `srcaddr`, `dstaddr`, `srcport`, `dstport`, `protocol`, `bytes`, `packets`
- Zeek `conn.log` CSV exports: `id.orig_h`, `id.resp_h`, `id.orig_p`, `id.resp_p`, `proto`, `service`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`
- Suricata EVE flattened CSV: `src_ip`, `dest_ip`, `proto`, `app_proto`, `flow.bytes_toserver`, `flow.bytes_toclient`, `flow.pkts_toserver`, `flow.pkts_toclient`

The normalizer maps these fields to the IDS flow schema and derives market-friendly features when they are missing:

- directional bytes and packets: `sbytes`, `dbytes`, `spkts`, `dpkts`
- duration from start/end timestamps when needed
- traffic rates: `sload`, `dload`
- packet size means: `smeansz`, `dmeansz`
- inter-packet timing estimates: `sintpkt`, `dintpkt`
- per-file context counts: `ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm`, `ct_src_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`

For best inference quality, include at least:

```csv
src_ip,dst_ip,src_port,dst_port,protocol,service,state,duration,src_bytes,dst_bytes,src_pkts,dst_pkts
10.0.0.5,192.0.2.10,51544,443,tcp,https,ESTABLISHED,1.2,860,4210,8,10
```

If only total `bytes` and `packets` are available, the normalizer splits them evenly across source and destination as a fallback. This keeps inference running, but confidence should be treated as lower than logs with directional counters.
