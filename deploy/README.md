# MCP Router System Deployment

Complete deployment configuration for the MCP-based Claude Code + Ollama integration system.

## Quick Start

1. **Clone and prepare**:
   ```bash
   git clone <repository-url>
   cd intern/deploy
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start the system**:
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment**:
   ```bash
   # Check all services are running
   docker-compose ps
   
   # Check system health
   curl http://localhost:8090/health
   
   # Access Grafana dashboard
   open http://localhost:3000
   # Login: admin/mcp_admin_2024
   ```

## Architecture

The deployment includes:

- **MCP Router**: Main application service
- **PostgreSQL**: Database for routing history and metrics
- **Ollama**: Local LLM service
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard
- **Redis**: Caching layer
- **Nginx**: Reverse proxy (production profile)

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Database
DB_HOST=postgres
DB_PASSWORD=mcp_secure_password_2024

# Ollama
OLLAMA_BASE_URL=http://ollama:11434

# System
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=50

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=8090
```

### Production Configuration

For production deployment:

1. **Enable production profile**:
   ```bash
   docker-compose --profile production up -d
   ```

2. **Configure SSL certificates**:
   ```bash
   # Place SSL certificates in deploy/nginx/ssl/
   cp your-cert.pem deploy/nginx/ssl/cert.pem
   cp your-key.pem deploy/nginx/ssl/key.pem
   ```

3. **Secure database credentials**:
   ```bash
   # Generate strong passwords
   openssl rand -base64 32
   
   # Update in .env file
   DB_PASSWORD=your-secure-password
   ```

## Service Details

### MCP Router (Port 8090)

Main application service providing:
- `/health` - Health check endpoint
- `/metrics` - Prometheus metrics
- MCP stdio interface for Claude Code integration

**Health Check**:
```bash
curl http://localhost:8090/health
```

**Metrics**:
```bash
curl http://localhost:8090/metrics
```

### PostgreSQL (Port 5432)

Database configuration:
- Database: `mcp_routing`
- User: `mcp_user`
- Password: From `DB_PASSWORD` env var

**Connect to database**:
```bash
docker exec -it mcp-postgres psql -U mcp_user -d mcp_routing
```

**Backup database**:
```bash
docker exec mcp-postgres pg_dump -U mcp_user mcp_routing > backup.sql
```

### Ollama (Port 11434)

Local LLM service providing:
- Model management
- Text generation API
- Health monitoring

**Check available models**:
```bash
curl http://localhost:11434/api/tags
```

**Pull additional models**:
```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:7b"}'
```

### Grafana (Port 3000)

Monitoring dashboard:
- URL: http://localhost:3000
- Username: `admin`
- Password: `mcp_admin_2024`

Pre-configured dashboards:
- System Overview
- Routing Performance
- Model Metrics
- Error Analysis

### Prometheus (Port 9090)

Metrics collection:
- URL: http://localhost:9090
- Scrapes metrics every 15 seconds
- 30-day retention

**Query examples**:
```promql
# Request rate
rate(mcp_requests_total[5m])

# Response time percentiles
histogram_quantile(0.95, rate(mcp_response_time_seconds_bucket[5m]))

# Error rate
rate(mcp_requests_total{status="error"}[5m]) / rate(mcp_requests_total[5m])
```

### Redis (Port 6379)

Caching layer for:
- Category scan results
- Interview decisions
- Response transformations

**Connect to Redis**:
```bash
docker exec -it mcp-redis redis-cli
```

## Monitoring

### Health Checks

All services include health checks:
```bash
# Check all service health
docker-compose ps

# Individual service health
curl http://localhost:8090/health  # MCP Router
curl http://localhost:11434/api/tags  # Ollama
```

### Logs

View service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mcp-router

# With timestamps
docker-compose logs -f -t mcp-router
```

### Metrics

Key metrics to monitor:

1. **Request Rate**: `mcp_requests_total`
2. **Response Time**: `mcp_response_time_seconds`
3. **Error Rate**: `mcp_requests_total{status="error"}`
4. **Model Distribution**: `mcp_model_routing_total`
5. **System Health**: `mcp_ollama_health_status`

## Maintenance

### Database Maintenance

**Clean old records**:
```sql
-- Connect to database
SELECT * FROM cleanup_old_records(30); -- Keep 30 days
```

**View performance stats**:
```sql
SELECT * FROM routing_summary LIMIT 10;
SELECT * FROM category_performance ORDER BY interview_count DESC;
```

### Log Rotation

Logs are automatically rotated:
- Max file size: 100MB
- Keep 5 backup files
- Location: `/app/logs/` in container

### Backup Strategy

1. **Database backup**:
   ```bash
   ./scripts/backup_db.sh
   ```

2. **Configuration backup**:
   ```bash
   tar -czf config-backup.tar.gz deploy/config/ deploy/.env
   ```

### Updates

To update the system:

1. **Pull latest images**:
   ```bash
   docker-compose pull
   ```

2. **Update configuration**:
   ```bash
   # Review and update .env and config files
   ```

3. **Restart services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## Troubleshooting

### Common Issues

1. **Ollama models not loading**:
   ```bash
   # Check Ollama service
   docker-compose logs ollama
   
   # Manually pull model
   curl -X POST http://localhost:11434/api/pull \
     -H "Content-Type: application/json" \
     -d '{"name": "llama3.2"}'
   ```

2. **Database connection issues**:
   ```bash
   # Check database service
   docker-compose logs postgres
   
   # Test connection
   docker exec mcp-router python -c "
   import asyncio
   from src.database.connection import DatabaseManager
   asyncio.run(DatabaseManager().initialize())
   "
   ```

3. **High memory usage**:
   ```bash
   # Check resource usage
   docker stats
   
   # Adjust resource limits in docker-compose.yml
   ```

### Debug Mode

Enable debug mode:
```bash
# Set in .env
LOG_LEVEL=DEBUG
MCP_ENV=development

# Restart services
docker-compose restart mcp-router
```

### Performance Tuning

For high-load environments:

1. **Increase connection pools**:
   ```bash
   DB_POOL_MAX=50
   MAX_CONCURRENT_REQUESTS=100
   ```

2. **Scale services**:
   ```bash
   docker-compose up -d --scale mcp-router=3
   ```

3. **Enable caching**:
   ```bash
   CACHE_ENABLED=true
   REDIS_URL=redis://redis:6379/0
   ```

## Security

### Production Security

1. **Change default passwords**:
   - Database password in `.env`
   - Grafana admin password in `docker-compose.yml`

2. **Enable SSL**:
   - Configure certificates in `nginx/ssl/`
   - Enable HTTPS in nginx configuration

3. **Restrict access**:
   - Configure firewall rules
   - Use nginx IP restrictions
   - Enable rate limiting

4. **Monitor security**:
   - Review access logs
   - Monitor failed authentication attempts
   - Set up alerts for suspicious activity

### Network Security

Services communicate on isolated network:
- Network: `mcp-network` (172.20.0.0/16)
- Internal communication only
- External access via nginx proxy

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f`
2. Review health endpoints
3. Consult monitoring dashboards
4. Check GitHub issues: https://github.com/D0k-Soundwave/intern/issues