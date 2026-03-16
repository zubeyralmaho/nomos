# Nomos Deployment Guide

This guide covers deploying Nomos in production environments.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Bare Metal Deployment](#bare-metal-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Security Considerations](#security-considerations)
- [Monitoring & Observability](#monitoring--observability)
- [High Availability](#high-availability)
- [Performance Tuning](#performance-tuning)

---

## Deployment Options

| Method | Best For | Complexity |
|--------|----------|------------|
| Docker | Quick setup, testing | Low |
| Bare Metal | Maximum performance | Medium |
| Kubernetes | Production scale | High |

---

## Docker Deployment

### Build the Docker Image

Create a `Dockerfile` in your project root:

```dockerfile
FROM rust:1.82-slim as builder

WORKDIR /app
COPY . .

RUN cargo build --release -p nomos-core

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/nomos-core /usr/local/bin/

EXPOSE 8080 8081

ENTRYPOINT ["nomos-core"]
```

### Build and Run

```bash
# Build the image
docker build -t nomos:latest .

# Run the container
docker run -d \
  --name nomos \
  -p 8080:8080 \
  -p 8081:8081 \
  -e NOMOS_UPSTREAM_HOST=api.example.com \
  -e NOMOS_UPSTREAM_PORT=443 \
  -e RUST_LOG=info \
  nomos:latest
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  nomos:
    build: .
    ports:
      - "8080:8080"   # Proxy port
      - "8081:8081"   # Control plane
    environment:
      - NOMOS_UPSTREAM_HOST=api.example.com
      - NOMOS_UPSTREAM_PORT=443
      - NOMOS_CONFIDENCE_THRESHOLD=0.75
      - RUST_LOG=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 512M
```

Run with:
```bash
docker-compose up -d
```

---

## Bare Metal Deployment

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 256 MB | 512 MB |
| Disk | 100 MB | 500 MB |
| OS | Linux x86_64 | Linux x86_64 |

### Build for Production

```bash
# Clone the repository
git clone https://github.com/zubeyralmaho/nomos.git
cd nomos

# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release -p nomos-core

# Verify the build
./target/release/nomos-core --version
```

### Create a Systemd Service

Create `/etc/systemd/system/nomos.service`:

```ini
[Unit]
Description=Nomos Schema-Healing Proxy
After=network.target

[Service]
Type=simple
User=nomos
Group=nomos
ExecStart=/usr/local/bin/nomos-core
Restart=always
RestartSec=5

# Environment configuration
Environment=NOMOS_UPSTREAM_HOST=api.example.com
Environment=NOMOS_UPSTREAM_PORT=443
Environment=NOMOS_LISTEN_PORT=8080
Environment=NOMOS_CONFIDENCE_THRESHOLD=0.75
Environment=RUST_LOG=info

# Resource limits
LimitNOFILE=65535
MemoryMax=512M

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Enable and Start

```bash
# Create service user
sudo useradd -r -s /bin/false nomos

# Copy binary
sudo cp target/release/nomos-core /usr/local/bin/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable nomos
sudo systemctl start nomos

# Check status
sudo systemctl status nomos
```

### With eBPF Acceleration

For maximum performance, enable eBPF (requires root):

```bash
# Build eBPF components
./build-ebpf.sh

# Modify service to run as root (required for eBPF)
# Add to [Service] section:
# User=root
# CapabilityBoundingSet=CAP_BPF CAP_NET_ADMIN

# Or run manually:
sudo ./target/release/nomos-core --enable-ebpf
```

---

## Kubernetes Deployment

### Deployment Manifest

Create `nomos-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nomos
  labels:
    app: nomos
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nomos
  template:
    metadata:
      labels:
        app: nomos
    spec:
      containers:
      - name: nomos
        image: nomos:latest
        ports:
        - containerPort: 8080
          name: proxy
        - containerPort: 8081
          name: control
        env:
        - name: NOMOS_UPSTREAM_HOST
          valueFrom:
            configMapKeyRef:
              name: nomos-config
              key: upstream_host
        - name: NOMOS_UPSTREAM_PORT
          valueFrom:
            configMapKeyRef:
              name: nomos-config
              key: upstream_port
        - name: NOMOS_CONFIDENCE_THRESHOLD
          value: "0.75"
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            cpu: "500m"
            memory: "256Mi"
          limits:
            cpu: "2000m"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: nomos
spec:
  selector:
    app: nomos
  ports:
  - name: proxy
    port: 80
    targetPort: 8080
  - name: control
    port: 8081
    targetPort: 8081
  type: ClusterIP
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: nomos-config
data:
  upstream_host: "api.example.com"
  upstream_port: "443"
```

### Deploy to Kubernetes

```bash
kubectl apply -f nomos-deployment.yaml

# Check deployment status
kubectl get pods -l app=nomos
kubectl logs -l app=nomos
```

### Horizontal Pod Autoscaler

Create `nomos-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nomos-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nomos
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Configuration

### Environment Variables

| Variable | Description | Default | Production Recommendation |
|----------|-------------|---------|---------------------------|
| `NOMOS_UPSTREAM_HOST` | Target API hostname | `localhost` | Your API endpoint |
| `NOMOS_UPSTREAM_PORT` | Target API port | `9090` | `443` for HTTPS |
| `NOMOS_LISTEN_PORT` | Proxy listen port | `8080` | `80` or `8080` |
| `NOMOS_CONFIDENCE_THRESHOLD` | Min match confidence (0.0-1.0) | `0.70` | `0.75`-`0.85` |
| `RUST_LOG` | Log level | `info` | `warn` or `info` |
| `TOKIO_WORKER_THREADS` | Async runtime threads | CPU cores | 4-8 |

### Configuration File (Optional)

Create `/etc/nomos/config.yaml`:

```yaml
proxy:
  listen_addr: "0.0.0.0:8080"
  upstream_host: "api.example.com"
  upstream_port: 443
  timeout_ms: 30000

healing:
  enabled: true
  confidence_threshold: 0.75
  max_candidates: 5
  
nlp:
  weights:
    levenshtein: 0.20
    jaro_winkler: 0.25
    synonym: 0.35
    tfidf: 0.20

control:
  listen_addr: "0.0.0.0:8081"
  
logging:
  level: "info"
  format: "json"
```

---

## Security Considerations

### Network Security

1. **Firewall Rules**
   ```bash
   # Allow proxy port
   ufw allow 8080/tcp
   
   # Restrict control plane to internal network
   ufw allow from 10.0.0.0/8 to any port 8081
   ```

2. **TLS Termination** (Use with a reverse proxy)
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /etc/ssl/certs/nomos.crt;
       ssl_certificate_key /etc/ssl/private/nomos.key;
       
       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Resource Limits

Always set resource limits to prevent runaway processes:

```bash
# Systemd
MemoryMax=512M
CPUQuota=200%
LimitNOFILE=65535

# Docker
--memory=512m --cpus=2

# Kubernetes (see manifest above)
```

---

## Monitoring & Observability

### Health Endpoint

```bash
curl http://localhost:8081/health
# {"status": "healthy", "uptime_seconds": 3600}
```

### Metrics Endpoint

```bash
curl http://localhost:8081/metrics
# {"latency_p50_ms": 0.12, "latency_p99_ms": 0.22, ...}
```

### Prometheus Integration

Expose metrics in Prometheus format at `/metrics/prometheus`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nomos'
    static_configs:
      - targets: ['nomos:8081']
    metrics_path: /metrics/prometheus
```

### Log Aggregation

Configure JSON logging for easy parsing:

```bash
RUST_LOG=info RUST_LOG_FORMAT=json ./nomos-core
```

Output:
```json
{"timestamp":"2026-01-15T12:00:00Z","level":"INFO","target":"nomos_core::proxy","message":"Request healed","fields":{"path":"/api/user","latency_us":89,"ops":5}}
```

---

## High Availability

### Load Balancing

Deploy multiple Nomos instances behind a load balancer:

```
           ┌─────────────┐
           │ Load        │
Client ───▶│ Balancer    │
           │ (nginx/HAProxy)│
           └──────┬──────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌────▼────┐ ┌────▼────┐ ┌────▼────┐
│ Nomos 1 │ │ Nomos 2 │ │ Nomos 3 │
└─────────┘ └─────────┘ └─────────┘
```

### HAProxy Configuration

```haproxy
frontend nomos_frontend
    bind *:80
    default_backend nomos_backend

backend nomos_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server nomos1 10.0.0.1:8080 check
    server nomos2 10.0.0.2:8080 check
    server nomos3 10.0.0.3:8080 check
```

---

## Performance Tuning

### CPU Optimization

```bash
# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Pin to specific CPU cores (reduces context switching)
taskset -c 0-3 ./nomos-core
```

### Memory Optimization

```bash
# Use jemalloc for better memory performance
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so ./nomos-core
```

### Kernel Tuning

Add to `/etc/sysctl.conf`:

```ini
# Increase socket buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 262144
net.core.wmem_default = 262144

# Increase connection backlog
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535

# TCP optimizations
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
```

Apply with:
```bash
sudo sysctl -p
```

### File Descriptor Limits

Add to `/etc/security/limits.conf`:

```
nomos soft nofile 65535
nomos hard nofile 65535
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High latency | Debug build | Rebuild with `--release` |
| Connection refused | Service not running | Check `systemctl status nomos` |
| Memory growth | Possible leak | Monitor with `memory_monitor.py` |
| 502 Bad Gateway | Upstream unreachable | Check `NOMOS_UPSTREAM_*` vars |

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
RUST_LOG=debug ./nomos-core
```

### Performance Analysis

```bash
# Run stress test
python3 tests/stress_test.py --rps 5000 --duration 60

# Monitor memory
python3 tests/memory_monitor.py --duration 300
```

---

## Next Steps

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[Architecture](ARCHITECTURE.md)** - How Nomos works internally
- **[API Reference](API.md)** - Control plane API documentation

---

*Need help? Open an issue on [GitHub](https://github.com/zubeyralmaho/nomos/issues).*
