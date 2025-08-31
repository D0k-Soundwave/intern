#!/bin/bash
# 
# MCP Router System Entrypoint Script
# 
# Handles initialization, health checks, and graceful startup
# Per requirements.md: Complete implementation, no placeholders.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Configuration validation
validate_config() {
    log "Validating configuration..."
    
    # Required environment variables
    required_vars=(
        "DB_HOST"
        "DB_PORT" 
        "DB_NAME"
        "DB_USER"
        "DB_PASSWORD"
        "OLLAMA_BASE_URL"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    success "Configuration validation passed"
}

# Wait for database to be ready
wait_for_database() {
    log "Waiting for database connection..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if python -c "
import asyncio
import asyncpg
async def test_db():
    try:
        conn = await asyncpg.connect(
            host='$DB_HOST',
            port=$DB_PORT,
            database='$DB_NAME', 
            user='$DB_USER',
            password='$DB_PASSWORD'
        )
        await conn.close()
        return True
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False

result = asyncio.run(test_db())
exit(0 if result else 1)
        "; then
            success "Database connection established"
            return 0
        fi
        
        warn "Database connection attempt $attempt/$max_attempts failed, retrying in 5 seconds..."
        sleep 5
        ((attempt++))
    done
    
    error "Failed to connect to database after $max_attempts attempts"
    exit 1
}

# Wait for Ollama to be ready
wait_for_ollama() {
    log "Waiting for Ollama service..."
    
    local max_attempts=20
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
            success "Ollama service is ready"
            return 0
        fi
        
        warn "Ollama connection attempt $attempt/$max_attempts failed, retrying in 3 seconds..."
        sleep 3
        ((attempt++))
    done
    
    error "Failed to connect to Ollama after $max_attempts attempts"
    exit 1
}

# Initialize database schema
init_database() {
    log "Initializing database schema..."
    
    python -c "
import asyncio
from src.database.connection import DatabaseManager

async def init_db():
    db = DatabaseManager()
    try:
        await db.initialize()
        print('Database initialization completed')
    except Exception as e:
        print(f'Database initialization failed: {e}')
        raise
    finally:
        await db.close()

asyncio.run(init_db())
    " || {
        error "Database initialization failed"
        exit 1
    }
    
    success "Database schema initialized"
}

# Download required Ollama models
ensure_ollama_models() {
    log "Ensuring Ollama models are available..."
    
    local required_models=("llama3.2")
    
    for model in "${required_models[@]}"; do
        log "Checking model: $model"
        
        if curl -s -f "${OLLAMA_BASE_URL}/api/show" \
           -H "Content-Type: application/json" \
           -d "{\"name\":\"$model\"}" > /dev/null 2>&1; then
            log "Model $model is already available"
        else
            log "Pulling model: $model"
            curl -s -X POST "${OLLAMA_BASE_URL}/api/pull" \
                 -H "Content-Type: application/json" \
                 -d "{\"name\":\"$model\"}" || {
                warn "Failed to pull model $model, will attempt at runtime"
            }
        fi
    done
    
    success "Ollama model check completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    local dirs=(
        "/app/logs"
        "/app/data" 
        "/app/temp"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
}

# Set up logging configuration
setup_logging() {
    log "Setting up logging configuration..."
    
    # Create logs directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set log level from environment variable
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    
    # Configure Python logging
    cat > /app/logging.conf << EOF
[loggers]
keys=root

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=${LOG_LEVEL}
handlers=consoleHandler,fileHandler

[handler_consoleHandler]
class=StreamHandler
level=${LOG_LEVEL}
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=handlers.RotatingFileHandler
level=${LOG_LEVEL}
formatter=simpleFormatter
args=('/app/logs/mcp-router.log', 'a', 10485760, 5)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
EOF

    success "Logging configuration created"
}

# Health check endpoint setup
setup_health_check() {
    log "Setting up health check endpoint..."
    
    # Create simple health check script
    cat > /app/health_check.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import aiohttp
from aiohttp import web
import json
import sys
import os

# Add src to path
sys.path.insert(0, '/app')

async def health_handler(request):
    """Health check endpoint"""
    try:
        # Basic system health
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0.0",
            "components": {
                "database": "unknown",
                "ollama": "unknown",
                "system": "healthy"
            }
        }
        
        # Check database connectivity
        try:
            from src.database.connection import DatabaseManager
            db = DatabaseManager()
            await db.initialize()
            await db.close()
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = "unhealthy"
            health_status["status"] = "degraded"
        
        # Check Ollama connectivity
        try:
            ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags") as resp:
                    if resp.status == 200:
                        health_status["components"]["ollama"] = "healthy"
                    else:
                        health_status["components"]["ollama"] = "unhealthy"
                        health_status["status"] = "degraded"
        except Exception:
            health_status["components"]["ollama"] = "unhealthy"
            health_status["status"] = "degraded"
        
        return web.json_response(health_status)
        
    except Exception as e:
        return web.json_response({
            "status": "error",
            "error": str(e)
        }, status=500)

async def init_health_server():
    """Initialize health check server"""
    app = web.Application()
    app.router.add_get('/health', health_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 8090)
    await site.start()
    
    return runner

if __name__ == '__main__':
    # Simple health check for Docker
    import requests
    try:
        response = requests.get('http://localhost:8090/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') in ['healthy', 'degraded']:
                sys.exit(0)
        sys.exit(1)
    except:
        sys.exit(1)
EOF

    chmod +x /app/health_check.py
    success "Health check endpoint configured"
}

# Start health check server in background
start_health_server() {
    log "Starting health check server..."
    
    python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app')
from health_check import init_health_server

async def run_server():
    runner = await init_health_server()
    print('Health check server started on port 8090')
    # Keep server running
    while True:
        await asyncio.sleep(3600)

try:
    asyncio.run(run_server())
except KeyboardInterrupt:
    print('Health check server stopped')
    " &
    
    local health_pid=$!
    echo $health_pid > /app/health_server.pid
    
    # Wait a moment for server to start
    sleep 2
    
    success "Health check server started (PID: $health_pid)"
}

# Graceful shutdown handler
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Stop health check server
    if [[ -f /app/health_server.pid ]]; then
        local health_pid=$(cat /app/health_server.pid)
        if kill -0 $health_pid 2>/dev/null; then
            kill $health_pid
            log "Health check server stopped"
        fi
        rm -f /app/health_server.pid
    fi
    
    # Stop main application if running
    if [[ -n "$main_pid" ]]; then
        kill -TERM $main_pid 2>/dev/null || true
        wait $main_pid 2>/dev/null || true
        log "Main application stopped"
    fi
    
    success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Starting MCP Router System..."
    
    # Validate configuration
    validate_config
    
    # Create directories
    create_directories
    
    # Setup logging
    setup_logging
    
    # Setup health check
    setup_health_check
    
    # Start health check server
    start_health_server
    
    # Wait for dependencies
    wait_for_database
    wait_for_ollama
    
    # Initialize database
    init_database
    
    # Ensure Ollama models (optional, non-blocking)
    ensure_ollama_models || warn "Some Ollama models may not be available"
    
    success "Initialization completed successfully"
    
    # Execute the main command
    log "Starting main application: $*"
    exec "$@" &
    main_pid=$!
    
    # Wait for main process
    wait $main_pid
}

# Execute main function with all arguments
main "$@"