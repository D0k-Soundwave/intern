"""
Pytest Configuration and Fixtures

Shared fixtures and configuration for the comprehensive test suite.
Provides database setup, mock services, and test data.

Per requirements.md: Complete implementation, no placeholders.
"""
import asyncio
import pytest
import json
import tempfile
import logging
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

# Configure test logging
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_database():
    """Mock database manager"""
    mock_db = AsyncMock()
    mock_db.initialize.return_value = None
    mock_db.close.return_value = None
    mock_db.record_interview_result.return_value = None
    mock_db.record_routing_decision.return_value = None
    mock_db.get_routing_analytics.return_value = {
        "model_distribution": {"claude": 60, "ollama": 40},
        "top_categories": [{"name": "code_generation_simple", "count": 25}],
        "bias_analysis": {"total_interviews": 100, "bias_detected": 15}
    }
    mock_db.get_category_info.return_value = {
        "id": 1,
        "name": "code_generation_simple",
        "description": "Basic code snippets",
        "complexity": "low",
        "capabilities": ["syntax", "basic_logic"],
        "keywords": ["code", "function", "simple"],
        "typical_intents": ["create_function", "write_snippet"]
    }
    return mock_db

@pytest.fixture
async def mock_ollama_client():
    """Mock Ollama client"""
    mock_client = AsyncMock()
    mock_client.ensure_alive.return_value = True
    mock_client.generate.return_value = {
        "response": "Mock Ollama response",
        "metadata": {
            "model": "llama3.2",
            "total_duration": 1000000,
            "eval_count": 50
        }
    }
    mock_client.list_models.return_value = [
        {"name": "llama3.2", "size": 2000000000}
    ]
    mock_client.get_status.return_value = {
        "status": "healthy",
        "request_count": 42,
        "error_count": 1,
        "error_rate": 0.024
    }
    mock_client.cleanup.return_value = None
    return mock_client

@pytest.fixture
async def mock_metrics_collector():
    """Mock metrics collector"""
    mock_metrics = AsyncMock()
    mock_metrics.track_request.return_value = None
    mock_metrics.track_routing.return_value = None
    mock_metrics.track_interview_performance.return_value = None
    mock_metrics.track_category_scan.return_value = None
    mock_metrics.update_health_status.return_value = None
    mock_metrics.get_system_metrics.return_value = {
        "total_requests": 100,
        "error_rate_percent": 2.5,
        "average_response_times": {"interview_simple": 150.5}
    }
    mock_metrics.get_prometheus_metrics.return_value = "# Mock Prometheus metrics"
    return mock_metrics

@pytest.fixture
def sample_categories():
    """Sample category data for testing"""
    return {
        1: {
            "name": "code_generation_simple",
            "description": "Basic code snippets, single functions",
            "complexity": "low",
            "capabilities": ["syntax", "basic_logic"],
            "keywords": ["write code", "function", "simple", "basic", "snippet"],
            "typical_intents": ["create_function", "write_snippet", "basic_implementation"]
        },
        2: {
            "name": "code_generation_complex",
            "description": "Full applications, complex algorithms",
            "complexity": "high",
            "capabilities": ["architecture", "optimization", "debugging"],
            "keywords": ["application", "complex", "system", "architecture", "advanced"],
            "typical_intents": ["build_application", "complex_algorithm", "system_design"]
        },
        16: {
            "name": "data_analysis_simple",
            "description": "Basic data queries and summaries",
            "complexity": "low",
            "capabilities": ["sql", "basic_stats"],
            "keywords": ["analyze data", "summary", "statistics", "simple query"],
            "typical_intents": ["analyze_data", "data_summary", "basic_statistics"]
        }
    }

@pytest.fixture
def sample_interview_request():
    """Sample interview request data"""
    return {
        "request_id": "test-request-123",
        "input_text": "Write a simple Python function to calculate fibonacci numbers",
        "context": {
            "user_id": "test-user",
            "session_id": "test-session"
        }
    }

@pytest.fixture
def sample_category_info():
    """Sample category information"""
    return {
        "name": "code_generation_simple",
        "description": "Basic code snippets, single functions",
        "complexity": "low",
        "capabilities": ["syntax", "basic_logic"],
        "keywords": ["write code", "function", "simple", "basic", "snippet"],
        "typical_intents": ["create_function", "write_snippet", "basic_implementation"]
    }

@pytest.fixture
def sample_ollama_response():
    """Sample Ollama response"""
    return {
        "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "metadata": {
            "model": "llama3.2",
            "created_at": "2024-01-01T12:00:00Z",
            "done": True,
            "total_duration": 2500000000,
            "load_duration": 100000000,
            "prompt_eval_count": 15,
            "eval_count": 45,
            "eval_duration": 2000000000
        }
    }

@pytest.fixture
def complex_input_text():
    """Complex input text for testing edge cases"""
    return """
    I need help building a distributed microservices architecture for an e-commerce platform.
    The system should handle high traffic, implement proper security measures, include
    real-time analytics, support multiple payment gateways, and have comprehensive
    monitoring and alerting. Please provide a detailed implementation plan with code
    examples for the core services, database design, API specifications, deployment
    configurations, and testing strategies.
    """

@pytest.fixture
def simple_input_text():
    """Simple input text for testing basic scenarios"""
    return "Write a hello world function in Python"

@pytest.fixture
def question_input_text():
    """Question-based input for testing Q&A scenarios"""
    return "What is the difference between a list and a tuple in Python?"

@pytest.fixture
async def test_database_config():
    """Test database configuration"""
    from src.database.connection import DatabaseConfig
    return DatabaseConfig(
        host="localhost",
        database="test_mcp_routing",
        username="test_user",
        password="test_password",
        min_connections=1,
        max_connections=5
    )

@pytest.fixture
def mock_health_check_result():
    """Mock health check result"""
    from src.adapters.ollama_client import HealthCheckResult, OllamaStatus
    return HealthCheckResult(
        status=OllamaStatus.HEALTHY,
        response_time=0.125,
        timestamp=datetime.now(timezone.utc),
        details={"models_count": 3, "endpoint": "http://localhost:11434/api/tags"}
    )

@pytest.fixture
def mock_scan_result():
    """Mock category scan result"""
    from src.routing.category_scanner import ScanResult, CategoryMatch
    from src.routing.categories import get_category_by_id
    
    category = get_category_by_id(1)
    match = CategoryMatch(
        category_id=1,
        category=category,
        confidence=0.85,
        matched_keywords=["function", "simple", "code"],
        matched_patterns=["def "],
        intent_matches=["create_function"]
    )
    
    return ScanResult(
        category=category,
        confidence=0.85,
        all_matches=[match],
        confidence_scores={1: 0.85},
        intents=["create"],
        keywords_found=["function", "python", "code"]
    )

@pytest.fixture
def mock_interview_result():
    """Mock interview result"""
    from src.interview.process_manager import InterviewResult
    return InterviewResult(
        recommended_model="ollama",
        confidence=0.78,
        reasoning="Simple code generation task suitable for Ollama with 3% preference applied",
        complexity_score=0.3,
        ollama_preference_applied=True,
        bias_detected=False,
        fallback_reason=None,
        metadata={
            "request_id": "test-123",
            "category": "code_generation_simple",
            "requirements": {"technical_complexity": 0.2},
            "timestamp": "2024-01-01T12:00:00Z"
        }
    )

# Mock external services
@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Automatically mock external services for all tests"""
    # Mock aiohttp for Ollama client
    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "models": [{"name": "llama3.2"}],
        "response": "Mocked response"
    }
    
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.closed = False
    
    # Mock aiohttp.ClientSession
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: mock_session)
    
    # Mock asyncpg for database
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    monkeypatch.setattr("asyncpg.create_pool", AsyncMock(return_value=mock_pool))

# Pytest configuration
def pytest_configure(config):
    """Pytest configuration"""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers"""
    for item in items:
        # Add unit test marker if no other marker specified
        if not any(marker.name in ["integration", "e2e"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)