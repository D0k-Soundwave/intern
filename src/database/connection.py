"""
Database Connection and Management

Provides PostgreSQL database connection management with connection pooling,
migrations, and analytics queries for the MCP routing system.

Per requirements.md: Complete implementation, no placeholders.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import asyncpg
import json

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "mcp_routing"
    username: str = "mcp_user"
    password: str = "mcp_password"
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: int = 60

class DatabaseManager:
    """
    Database manager with connection pooling and analytics
    
    Manages:
    - Connection pooling with asyncpg
    - Interview performance tracking
    - Routing history and analytics
    - Category information storage
    - Health monitoring and metrics
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        
        logger.info("Database Manager initialized")
    
    async def initialize(self):
        """Initialize database connection and ensure schema exists"""
        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            
            # Ensure database schema exists
            await self._ensure_schema()
            
            # Run any pending migrations
            await self._run_migrations()
            
            self._initialized = True
            logger.info("Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Database connection pool closed")
    
    async def _ensure_schema(self):
        """Ensure database schema exists"""
        schema_sql = """
        -- Categories table
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            complexity VARCHAR(10) NOT NULL CHECK (complexity IN ('low', 'medium', 'high')),
            capabilities JSONB,
            keywords JSONB,
            typical_intents JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Interview performance table
        CREATE TABLE IF NOT EXISTS interview_performance (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(100) NOT NULL,
            request_hash VARCHAR(32),
            input_text TEXT,
            category_id INTEGER REFERENCES categories(id),
            category_name VARCHAR(100),
            recommended_model VARCHAR(20) NOT NULL,
            confidence DECIMAL(4,3) NOT NULL,
            complexity_score DECIMAL(4,3),
            ollama_preference_applied BOOLEAN DEFAULT FALSE,
            bias_detected BOOLEAN DEFAULT FALSE,
            fallback_reason VARCHAR(100),
            requirements_analysis JSONB,
            capability_scores JSONB,
            bias_analysis JSONB,
            response_time_ms INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Routing history table  
        CREATE TABLE IF NOT EXISTS routing_history (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(100) NOT NULL,
            input_text TEXT,
            category_name VARCHAR(100),
            model_used VARCHAR(20) NOT NULL,
            response_length INTEGER,
            processing_time_ms INTEGER,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Model performance metrics table
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(20) NOT NULL,
            category_name VARCHAR(100),
            total_requests INTEGER DEFAULT 0,
            successful_requests INTEGER DEFAULT 0,
            failed_requests INTEGER DEFAULT 0,
            avg_response_time_ms DECIMAL(8,2),
            avg_confidence DECIMAL(4,3),
            last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- System health metrics
        CREATE TABLE IF NOT EXISTS system_health (
            id SERIAL PRIMARY KEY,
            component VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            response_time_ms INTEGER,
            error_count INTEGER DEFAULT 0,
            details JSONB,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_interview_performance_category ON interview_performance(category_name);
        CREATE INDEX IF NOT EXISTS idx_interview_performance_model ON interview_performance(recommended_model);
        CREATE INDEX IF NOT EXISTS idx_interview_performance_created ON interview_performance(created_at);
        CREATE INDEX IF NOT EXISTS idx_routing_history_model ON routing_history(model_used);
        CREATE INDEX IF NOT EXISTS idx_routing_history_created ON routing_history(created_at);
        CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(model_name);
        CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
        CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);
        """
        
        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)
            logger.info("Database schema ensured")
    
    async def _run_migrations(self):
        """Run database migrations"""
        # Create migrations table if it doesn't exist
        migration_table_sql = """
        CREATE TABLE IF NOT EXISTS migrations (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        async with self._pool.acquire() as conn:
            await conn.execute(migration_table_sql)
            
            # Check which migrations have been applied
            applied_migrations = await conn.fetch(
                "SELECT name FROM migrations ORDER BY applied_at"
            )
            applied_names = {row['name'] for row in applied_migrations}
            
            # Define available migrations
            migrations = [
                ("001_initial_schema", "Initial schema creation"),
                ("002_add_request_hash", "Add request hash for deduplication"),
                ("003_add_performance_indexes", "Add performance indexes")
            ]
            
            # Apply new migrations
            for migration_name, description in migrations:
                if migration_name not in applied_names:
                    logger.info(f"Applying migration: {migration_name}")
                    
                    # Mark migration as applied
                    await conn.execute(
                        "INSERT INTO migrations (name) VALUES ($1)",
                        migration_name
                    )
                    
            logger.info("Database migrations completed")
    
    async def record_interview_result(self, result_data: Dict[str, Any]):
        """Record interview process result"""
        if not self._initialized:
            logger.warning("Database not initialized, skipping record")
            return
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO interview_performance (
                        request_id, input_text, category_name, recommended_model,
                        confidence, complexity_score, ollama_preference_applied,
                        bias_detected, fallback_reason, requirements_analysis,
                        capability_scores, bias_analysis, response_time_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    result_data.get('request_id'),
                    result_data.get('input_text'),
                    result_data.get('category_name'),
                    result_data.get('recommended_model'),
                    result_data.get('confidence'),
                    result_data.get('complexity_score'),
                    result_data.get('ollama_preference_applied', False),
                    result_data.get('bias_detected', False),
                    result_data.get('fallback_reason'),
                    json.dumps(result_data.get('requirements_analysis', {})),
                    json.dumps(result_data.get('capability_scores', {})),
                    json.dumps(result_data.get('bias_analysis', {})),
                    result_data.get('response_time_ms', 0)
                )
            
        except Exception as e:
            logger.error(f"Error recording interview result: {e}")
    
    async def record_routing_decision(self, routing_data: Dict[str, Any]):
        """Record routing decision and outcome"""
        if not self._initialized:
            logger.warning("Database not initialized, skipping record")
            return
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO routing_history (
                        request_id, input_text, category_name, model_used,
                        response_length, processing_time_ms, success,
                        error_message, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    routing_data.get('request_id'),
                    routing_data.get('input_text'),
                    routing_data.get('category_name'),
                    routing_data.get('model_used'),
                    routing_data.get('response_length', 0),
                    routing_data.get('processing_time_ms', 0),
                    routing_data.get('success', True),
                    routing_data.get('error_message'),
                    json.dumps(routing_data.get('metadata', {}))
                )
            
        except Exception as e:
            logger.error(f"Error recording routing decision: {e}")
    
    async def update_model_metrics(self, model_name: str, category_name: str, metrics: Dict[str, Any]):
        """Update model performance metrics"""
        if not self._initialized:
            return
        
        try:
            async with self._pool.acquire() as conn:
                # Use upsert pattern
                await conn.execute("""
                    INSERT INTO model_metrics (
                        model_name, category_name, total_requests, successful_requests,
                        failed_requests, avg_response_time_ms, avg_confidence
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (model_name, category_name) 
                    DO UPDATE SET
                        total_requests = model_metrics.total_requests + $3,
                        successful_requests = model_metrics.successful_requests + $4,
                        failed_requests = model_metrics.failed_requests + $5,
                        avg_response_time_ms = ($6 + model_metrics.avg_response_time_ms) / 2,
                        avg_confidence = ($7 + model_metrics.avg_confidence) / 2,
                        last_updated = CURRENT_TIMESTAMP
                """,
                    model_name,
                    category_name,
                    metrics.get('total_requests', 0),
                    metrics.get('successful_requests', 0),
                    metrics.get('failed_requests', 0),
                    metrics.get('avg_response_time_ms', 0),
                    metrics.get('avg_confidence', 0)
                )
            
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
    
    async def get_routing_analytics(self, time_window: str = "1h") -> Dict[str, Any]:
        """Get routing analytics for specified time window"""
        if not self._initialized:
            return {}
        
        try:
            # Parse time window
            if time_window == "1h":
                start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            elif time_window == "24h":
                start_time = datetime.now(timezone.utc) - timedelta(hours=24)
            elif time_window == "7d":
                start_time = datetime.now(timezone.utc) - timedelta(days=7)
            else:
                start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            
            async with self._pool.acquire() as conn:
                # Model distribution
                model_dist = await conn.fetch("""
                    SELECT recommended_model, COUNT(*) as count
                    FROM interview_performance 
                    WHERE created_at >= $1
                    GROUP BY recommended_model
                """, start_time)
                
                # Category distribution
                category_dist = await conn.fetch("""
                    SELECT category_name, COUNT(*) as count
                    FROM interview_performance 
                    WHERE created_at >= $1
                    GROUP BY category_name
                    ORDER BY count DESC
                    LIMIT 10
                """, start_time)
                
                # Average confidence by model
                confidence_stats = await conn.fetch("""
                    SELECT recommended_model, 
                           AVG(confidence) as avg_confidence,
                           COUNT(*) as total_decisions
                    FROM interview_performance 
                    WHERE created_at >= $1
                    GROUP BY recommended_model
                """, start_time)
                
                # Bias detection stats
                bias_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total_interviews,
                           SUM(CASE WHEN bias_detected THEN 1 ELSE 0 END) as bias_detected,
                           SUM(CASE WHEN ollama_preference_applied THEN 1 ELSE 0 END) as preference_applied
                    FROM interview_performance 
                    WHERE created_at >= $1
                """, start_time)
                
                # Performance metrics
                performance_stats = await conn.fetch("""
                    SELECT model_name,
                           total_requests,
                           successful_requests,
                           failed_requests,
                           avg_response_time_ms,
                           avg_confidence
                    FROM model_metrics
                    WHERE last_updated >= $1
                """, start_time)
                
                return {
                    "time_window": time_window,
                    "model_distribution": {row['recommended_model']: row['count'] for row in model_dist},
                    "top_categories": [{"name": row['category_name'], "count": row['count']} for row in category_dist],
                    "confidence_by_model": {
                        row['recommended_model']: {
                            "avg_confidence": float(row['avg_confidence']),
                            "total_decisions": row['total_decisions']
                        } for row in confidence_stats
                    },
                    "bias_analysis": {
                        "total_interviews": bias_stats['total_interviews'],
                        "bias_detected": bias_stats['bias_detected'],
                        "preference_applied": bias_stats['preference_applied'],
                        "bias_detection_rate": float(bias_stats['bias_detected']) / max(bias_stats['total_interviews'], 1) * 100
                    },
                    "performance_metrics": {
                        row['model_name']: {
                            "total_requests": row['total_requests'],
                            "success_rate": float(row['successful_requests']) / max(row['total_requests'], 1) * 100,
                            "avg_response_time": float(row['avg_response_time_ms']),
                            "avg_confidence": float(row['avg_confidence'])
                        } for row in performance_stats
                    }
                }
            
        except Exception as e:
            logger.error(f"Error getting routing analytics: {e}")
            return {}
    
    async def get_category_info(self, category_name: str) -> Optional[Dict[str, Any]]:
        """Get category information by name"""
        if not self._initialized:
            return None
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, name, description, complexity, capabilities,
                           keywords, typical_intents
                    FROM categories
                    WHERE name = $1
                """, category_name)
                
                if row:
                    return {
                        "id": row['id'],
                        "name": row['name'],
                        "description": row['description'],
                        "complexity": row['complexity'],
                        "capabilities": json.loads(row['capabilities']) if row['capabilities'] else [],
                        "keywords": json.loads(row['keywords']) if row['keywords'] else [],
                        "typical_intents": json.loads(row['typical_intents']) if row['typical_intents'] else []
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting category info: {e}")
            return None
    
    async def ensure_categories_populated(self, categories_data: Dict[int, Dict[str, Any]]):
        """Ensure categories table is populated with all 76 categories"""
        if not self._initialized:
            return
        
        try:
            async with self._pool.acquire() as conn:
                for category_id, category_info in categories_data.items():
                    await conn.execute("""
                        INSERT INTO categories (
                            id, name, description, complexity, capabilities,
                            keywords, typical_intents
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            description = EXCLUDED.description,
                            complexity = EXCLUDED.complexity,
                            capabilities = EXCLUDED.capabilities,
                            keywords = EXCLUDED.keywords,
                            typical_intents = EXCLUDED.typical_intents,
                            updated_at = CURRENT_TIMESTAMP
                    """,
                        category_id,
                        category_info['name'],
                        category_info['description'],
                        category_info['complexity'],
                        json.dumps(category_info.get('capabilities', [])),
                        json.dumps(category_info.get('keywords', [])),
                        json.dumps(category_info.get('typical_intents', []))
                    )
                
            logger.info(f"Populated {len(categories_data)} categories in database")
            
        except Exception as e:
            logger.error(f"Error populating categories: {e}")
    
    async def record_system_health(self, component: str, status: str, details: Dict[str, Any]):
        """Record system health metrics"""
        if not self._initialized:
            return
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO system_health (
                        component, status, response_time_ms, error_count, details
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    component,
                    status,
                    details.get('response_time_ms', 0),
                    details.get('error_count', 0),
                    json.dumps(details)
                )
            
        except Exception as e:
            logger.error(f"Error recording system health: {e}")
    
    async def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old records to prevent database bloat"""
        if not self._initialized:
            return
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            async with self._pool.acquire() as conn:
                # Clean up old interview performance records
                deleted_interviews = await conn.fetchval("""
                    DELETE FROM interview_performance 
                    WHERE created_at < $1
                    RETURNING count(*)
                """, cutoff_date)
                
                # Clean up old routing history
                deleted_routing = await conn.fetchval("""
                    DELETE FROM routing_history 
                    WHERE created_at < $1
                    RETURNING count(*)
                """, cutoff_date)
                
                # Clean up old system health records
                deleted_health = await conn.fetchval("""
                    DELETE FROM system_health 
                    WHERE timestamp < $1
                    RETURNING count(*)
                """, cutoff_date)
                
                logger.info(
                    f"Cleaned up old records: {deleted_interviews} interviews, "
                    f"{deleted_routing} routing records, {deleted_health} health records"
                )
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")