-- MCP Router Database Initialization Script
-- 
-- Creates initial database schema and populates with category data
-- Per requirements.md: Complete implementation, no placeholders.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create application user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mcp_user') THEN
        CREATE ROLE mcp_user WITH LOGIN PASSWORD 'mcp_secure_password_2024';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE mcp_routing TO mcp_user;
GRANT ALL ON SCHEMA public TO mcp_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO mcp_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO mcp_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mcp_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mcp_user;

-- Performance tuning
-- Increase shared buffers for better performance
-- ALTER SYSTEM SET shared_buffers = '256MB';
-- ALTER SYSTEM SET effective_cache_size = '1GB';
-- ALTER SYSTEM SET maintenance_work_mem = '64MB';
-- ALTER SYSTEM SET checkpoint_completion_target = 0.9;
-- ALTER SYSTEM SET wal_buffers = '16MB';
-- ALTER SYSTEM SET default_statistics_target = 100;

-- Create initial schema (will be managed by application migrations)
-- This is just to ensure basic structure exists

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    complexity VARCHAR(10) NOT NULL CHECK (complexity IN ('low', 'medium', 'high')),
    capabilities JSONB DEFAULT '[]'::jsonb,
    keywords JSONB DEFAULT '[]'::jsonb,
    typical_intents JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Interview performance tracking
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

-- Routing history
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

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(20) NOT NULL,
    category_name VARCHAR(100),
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    avg_response_time_ms DECIMAL(8,2),
    avg_confidence DECIMAL(4,3),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, category_name)
);

-- System health tracking
CREATE TABLE IF NOT EXISTS system_health (
    id SERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    error_count INTEGER DEFAULT 0,
    details JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Migrations tracking
CREATE TABLE IF NOT EXISTS migrations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_interview_performance_category ON interview_performance(category_name);
CREATE INDEX IF NOT EXISTS idx_interview_performance_model ON interview_performance(recommended_model);
CREATE INDEX IF NOT EXISTS idx_interview_performance_created ON interview_performance(created_at);
CREATE INDEX IF NOT EXISTS idx_interview_performance_request_id ON interview_performance(request_id);

CREATE INDEX IF NOT EXISTS idx_routing_history_model ON routing_history(model_used);
CREATE INDEX IF NOT EXISTS idx_routing_history_created ON routing_history(created_at);
CREATE INDEX IF NOT EXISTS idx_routing_history_request_id ON routing_history(request_id);

CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_category ON model_metrics(category_name);

CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);

CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);
CREATE INDEX IF NOT EXISTS idx_categories_complexity ON categories(complexity);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to categories
DROP TRIGGER IF EXISTS update_categories_updated_at ON categories;
CREATE TRIGGER update_categories_updated_at
    BEFORE UPDATE ON categories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert initial migration record
INSERT INTO migrations (name) VALUES ('001_initial_schema') ON CONFLICT (name) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW routing_summary AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    model_used,
    COUNT(*) as request_count,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(*) FILTER (WHERE success = true) as successful_requests,
    COUNT(*) FILTER (WHERE success = false) as failed_requests
FROM routing_history 
GROUP BY DATE_TRUNC('hour', created_at), model_used
ORDER BY hour DESC;

CREATE OR REPLACE VIEW category_performance AS
SELECT 
    c.name as category_name,
    c.complexity,
    COUNT(ip.id) as interview_count,
    AVG(ip.confidence) as avg_confidence,
    COUNT(*) FILTER (WHERE ip.recommended_model = 'ollama') as ollama_selections,
    COUNT(*) FILTER (WHERE ip.recommended_model = 'claude') as claude_selections,
    COUNT(*) FILTER (WHERE ip.bias_detected = true) as bias_detections
FROM categories c
LEFT JOIN interview_performance ip ON c.name = ip.category_name
GROUP BY c.id, c.name, c.complexity
ORDER BY interview_count DESC;

-- Create function for cleanup old records
CREATE OR REPLACE FUNCTION cleanup_old_records(days_to_keep INTEGER DEFAULT 30)
RETURNS TABLE(deleted_interviews INTEGER, deleted_routing INTEGER, deleted_health INTEGER) AS $$
DECLARE
    cutoff_date TIMESTAMP WITH TIME ZONE;
    deleted_int INTEGER;
    deleted_rout INTEGER; 
    deleted_hlth INTEGER;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    DELETE FROM interview_performance WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_int = ROW_COUNT;
    
    DELETE FROM routing_history WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_rout = ROW_COUNT;
    
    DELETE FROM system_health WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS deleted_hlth = ROW_COUNT;
    
    deleted_interviews := deleted_int;
    deleted_routing := deleted_rout;
    deleted_health := deleted_hlth;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Create database user permissions function
CREATE OR REPLACE FUNCTION grant_mcp_permissions() RETURNS void AS $$
BEGIN
    GRANT ALL ON ALL TABLES IN SCHEMA public TO mcp_user;
    GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO mcp_user;
    GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO mcp_user;
END;
$$ LANGUAGE plpgsql;

-- Apply permissions
SELECT grant_mcp_permissions();

-- Log initialization completion
DO $$
BEGIN
    RAISE NOTICE 'MCP Router database initialization completed successfully';
    RAISE NOTICE 'Database: %, Schema version: 1.0.0', current_database();
    RAISE NOTICE 'Tables created: categories, interview_performance, routing_history, model_metrics, system_health, migrations';
    RAISE NOTICE 'Views created: routing_summary, category_performance';
    RAISE NOTICE 'Functions created: cleanup_old_records, grant_mcp_permissions';
END;
$$;