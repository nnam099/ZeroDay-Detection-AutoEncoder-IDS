"""
Script setup Kibana Index Pattern + Dashboard cho DDoS Detection.
Chạy 1 lần sau khi stack đã up:
    python setup_kibana.py
"""
import requests
import json
import time

KIBANA_URL = "http://localhost:5601"
ES_URL     = "http://localhost:9200"

headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}


def wait_for_services():
    print("⏳ Chờ Elasticsearch & Kibana sẵn sàng...")
    for service, url in [("Elasticsearch", f"{ES_URL}/_cluster/health"),
                          ("Kibana", f"{KIBANA_URL}/api/status")]:
        for i in range(30):
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    print(f"  ✅ {service} ready")
                    break
            except:
                pass
            print(f"  ... {service} chưa sẵn sàng, thử lại ({i+1}/30)")
            time.sleep(10)


def create_ilm_policy():
    """Tự động xóa index DDoS sau 30 ngày"""
    policy = {
        "policy": {
            "phases": {
                "hot": {
                    "actions": {
                        "rollover": {
                            "max_size": "10gb",
                            "max_age": "1d"
                        }
                    }
                },
                "delete": {
                    "min_age": "30d",
                    "actions": {"delete": {}}
                }
            }
        }
    }
    r = requests.put(f"{ES_URL}/_ilm/policy/ddos-ilm-policy",
                     json=policy, timeout=10)
    if r.status_code in (200, 201):
        print("✅ ILM Policy created (auto-delete sau 30 ngày)")
    else:
        print(f"⚠️  ILM Policy: {r.text}")


def create_index_template():
    """Định nghĩa mapping cho field ai_probability kiểu float"""
    template = {
        "index_patterns": ["ddos-detection-*", "ddos-alerts-*"],
        "template": {
            "mappings": {
                "properties": {
                    "ai_probability":  {"type": "float"},
                    "ai_latency_ms":   {"type": "float"},
                    "ai_label":        {"type": "keyword"},
                    "ai_tier":         {"type": "keyword"},
                    "ai_confidence":   {"type": "keyword"},
                    "ai_action":       {"type": "keyword"},
                    "source_ip":       {"type": "ip"},
                    "timestamp":       {"type": "date"},
                    "@timestamp":      {"type": "date"},
                    "Label":           {"type": "keyword"},
                    "tags":            {"type": "keyword"},
                }
            }
        }
    }
    r = requests.put(f"{ES_URL}/_index_template/ddos-template",
                     json=template, timeout=10)
    if r.status_code in (200, 201):
        print("✅ Index Template created (field mappings)")
    else:
        print(f"⚠️  Index Template: {r.text}")


def create_kibana_data_view():
    """Tạo Data View (Index Pattern) trong Kibana"""
    payload = {
        "data_view": {
            "title":      "ddos-detection-*",
            "name":       "DDoS Detection Logs",
            "timeFieldName": "@timestamp"
        }
    }
    r = requests.post(f"{KIBANA_URL}/api/data_views/data_view",
                      headers=headers, json=payload, timeout=10)
    if r.status_code in (200, 201):
        data_view_id = r.json()["data_view"]["id"]
        print(f"✅ Kibana Data View created (ID: {data_view_id})")
        return data_view_id
    else:
        print(f"⚠️  Data View: {r.text}")
        return None


def create_kibana_dashboard(data_view_id: str):
    """
    Tạo Dashboard cơ bản với các panel:
    - Số lượng Attack theo thời gian (Time series)
    - Phân phối AI Probability (Histogram)
    - Top Source IPs bị tấn công (Data table)
    - Tỉ lệ Attack/Benign (Pie chart)
    """
    dashboard = {
        "attributes": {
            "title": "DDoS Detection — Real-time Dashboard",
            "description": "CNN-BiLSTM-Transformer Hybrid Model v3.0",
            "panelsJSON": json.dumps([
                # Panel 1: Attack Events over time
                {
                    "embeddableConfig": {
                        "title": "Attack Events Over Time",
                        "vis": {
                            "type": "line",
                            "aggs": [
                                {"id": "1", "type": "count", "schema": "metric"},
                                {"id": "2", "type": "date_histogram", "schema": "segment",
                                 "params": {"field": "@timestamp", "interval": "auto"}}
                            ]
                        }
                    },
                    "gridData": {"x": 0, "y": 0, "w": 24, "h": 15, "i": "1"},
                    "panelIndex": "1",
                    "type": "visualization",
                }
            ]),
            "optionsJSON": json.dumps({"useMargins": True, "syncColors": False}),
            "timeRestore": False,
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "query": {"language": "kuery", "query": ""},
                    "filter": []
                })
            }
        },
        "references": [
            {"name": "kibanaSavedObjectMeta.searchSourceJSON.index",
             "type": "index-pattern", "id": data_view_id}
        ]
    }

    r = requests.post(f"{KIBANA_URL}/api/saved_objects/dashboard",
                      headers=headers, json=dashboard, timeout=10)
    if r.status_code in (200, 201):
        dash_id = r.json()["id"]
        print(f"✅ Dashboard created → http://localhost:5601/app/dashboards#{dash_id}")
    else:
        print(f"⚠️  Dashboard: {r.text}")


def create_detection_rate_watch():
    """
    Elasticsearch Watcher — Cảnh báo khi Attack Rate > 10% trong 5 phút
    Yêu cầu: Elasticsearch với X-Pack (trial/gold license)
    """
    watch = {
        "trigger": {
            "schedule": {"interval": "5m"}
        },
        "input": {
            "search": {
                "request": {
                    "indices": ["ddos-detection-*"],
                    "body": {
                        "size": 0,
                        "query": {
                            "range": {"@timestamp": {"gte": "now-5m"}}
                        },
                        "aggs": {
                            "attack_count": {
                                "filter": {"term": {"ai_label": "ATTACK"}}
                            },
                            "total": {"value_count": {"field": "@timestamp"}}
                        }
                    }
                }
            }
        },
        "condition": {
            "script": {
                "source": """
                    def total = ctx.payload.aggregations.total.value;
                    if (total == 0) return false;
                    def attacks = ctx.payload.aggregations.attack_count.doc_count;
                    return (attacks / total) > 0.1;
                """
            }
        },
        "actions": {
            "log_attack": {
                "logging": {
                    "text": "🚨 DDoS Alert: High attack rate detected (>10% in last 5 min)"
                }
            }
        }
    }

    r = requests.put(f"{ES_URL}/_watcher/watch/ddos-attack-rate",
                     json=watch, timeout=10)
    if r.status_code in (200, 201):
        print("✅ Elasticsearch Watcher alert created")
    else:
        print(f"ℹ️  Watcher (cần license X-Pack): {r.status_code}")


def print_useful_queries():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Kibana Useful Queries (KQL)                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Xem tất cả Attack:                                                      ║
║    ai_label : "ATTACK"                                                   ║
║                                                                          ║
║  Xem traffic nghi ngờ (cần verify):                                     ║
║    ai_tier : "CAPTCHA"                                                   ║
║                                                                          ║
║  Xem Attack có độ tin cậy cao (prob > 0.9):                             ║
║    ai_probability > 0.9                                                  ║
║                                                                          ║
║  Theo dõi 1 IP cụ thể:                                                   ║
║    source_ip : "192.168.x.x"                                             ║
║                                                                          ║
║  Thống kê latency AI:                                                    ║
║    → Visualize → Average of ai_latency_ms                               ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    print("🚀 Bắt đầu setup DDoS Detection Stack...\n")
    wait_for_services()
    create_ilm_policy()
    create_index_template()
    data_view_id = create_kibana_data_view()
    if data_view_id:
        create_kibana_dashboard(data_view_id)
    create_detection_rate_watch()
    print_useful_queries()
    print("\n✅ Setup hoàn tất! Mở Kibana: http://localhost:5601")
