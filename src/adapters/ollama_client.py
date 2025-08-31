"""
Ollama Lifecycle Manager and Client

Manages Ollama connections, health monitoring, and request lifecycle.
Provides robust integration with automatic recovery and connection pooling.

Per requirements.md: Complete implementation, no placeholders.
"""
import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class OllamaStatus(Enum):
    """Ollama service status"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

@dataclass
class HealthCheckResult:
    """Health check result"""
    status: OllamaStatus
    response_time: float
    timestamp: datetime
    error: Optional[str] = None
    details: Dict[str, Any] = None

@dataclass
class GenerationRequest:
    """Ollama generation request"""
    prompt: str
    model: str
    options: Dict[str, Any]
    stream: bool = False

@dataclass
class GenerationResponse:
    """Ollama generation response"""
    content: str
    model: str
    total_duration: Optional[int]
    load_duration: Optional[int]
    prompt_eval_count: Optional[int]
    eval_count: Optional[int]
    done: bool
    metadata: Dict[str, Any]

class OllamaClient:
    """
    Advanced Ollama client with lifecycle management
    
    Features:
    - Automatic health monitoring with 30-second checks
    - Connection pooling and reuse
    - Request retry with exponential backoff
    - Graceful degradation and fallback
    - 1-hour keep-alive for connection persistence
    - Performance metrics tracking
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        health_check_interval: int = 30,
        connection_timeout: int = 10,
        request_timeout: int = 300,
        max_retries: int = 3,
        keep_alive_duration: int = 3600
    ):
        self.base_url = base_url.rstrip('/')
        self.health_check_interval = health_check_interval
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.keep_alive_duration = keep_alive_duration
        
        # Connection management
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_activity = datetime.now(timezone.utc)
        self._connection_lock = asyncio.Lock()
        
        # Health monitoring
        self._current_status = OllamaStatus.UNKNOWN
        self._last_health_check = None
        self._health_history: List[HealthCheckResult] = []
        self._health_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        self._last_successful_request = None
        
        logger.info(f"Ollama Client initialized for {base_url}")
    
    async def ensure_alive(self) -> bool:
        """
        Ensure Ollama is alive and healthy
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            # Check if we need a fresh health check
            now = datetime.now(timezone.utc)
            
            if (self._last_health_check is None or 
                (now - self._last_health_check.timestamp).seconds > self.health_check_interval):
                
                await self._perform_health_check()
            
            # Start background health monitoring if not already running
            if self._health_task is None or self._health_task.done():
                self._health_task = asyncio.create_task(self._background_health_monitor())
            
            return self._current_status in [OllamaStatus.HEALTHY, OllamaStatus.DEGRADED]
            
        except Exception as e:
            logger.error(f"Error ensuring Ollama is alive: {e}")
            self._current_status = OllamaStatus.UNAVAILABLE
            return False
    
    async def generate(
        self, 
        prompt: str, 
        model: str = "llama3.2", 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response from Ollama
        
        Args:
            prompt: Input prompt
            model: Model name to use
            options: Generation options (temperature, max_tokens, etc.)
            
        Returns:
            Generation response with content and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure Ollama is available
            if not await self.ensure_alive():
                raise RuntimeError("Ollama is not available")
            
            # Prepare request
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options or {}
            }
            
            # Perform request with retry logic
            response_data = await self._make_request_with_retry(
                "POST", "/api/generate", request_data
            )
            
            # Process response
            response = self._process_generation_response(response_data)
            
            # Update metrics
            self._request_count += 1
            self._total_response_time += time.time() - start_time
            self._last_successful_request = datetime.now(timezone.utc)
            self._last_activity = datetime.now(timezone.utc)
            
            logger.info(f"Generated response: {len(response['response'])} chars in {time.time() - start_time:.2f}s")
            
            return response
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error generating response: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            if not await self.ensure_alive():
                return []
            
            response = await self._make_request_with_retry("GET", "/api/tags")
            return response.get('models', [])
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model: str) -> bool:
        """Pull/download a model"""
        try:
            if not await self.ensure_alive():
                return False
            
            request_data = {"name": model}
            await self._make_request_with_retry("POST", "/api/pull", request_data)
            
            logger.info(f"Successfully pulled model: {model}")
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current Ollama client status"""
        return {
            "status": self._current_status.value,
            "last_health_check": self._last_health_check.timestamp.isoformat() if self._last_health_check else None,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "average_response_time": self._total_response_time / max(self._request_count, 1),
            "last_successful_request": self._last_successful_request.isoformat() if self._last_successful_request else None,
            "last_activity": self._last_activity.isoformat(),
            "uptime_status": "healthy" if self._current_status == OllamaStatus.HEALTHY else "degraded"
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Cancel health monitoring
            if self._health_task and not self._health_task.done():
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass
            
            # Close HTTP session
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
            
            logger.info("Ollama client cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling"""
        async with self._connection_lock:
            now = datetime.now(timezone.utc)
            
            # Check if we need to recreate session (keep-alive expired or closed)
            if (self._session is None or 
                self._session.closed or 
                (now - self._last_activity).seconds > self.keep_alive_duration):
                
                # Close existing session if any
                if self._session and not self._session.closed:
                    await self._session.close()
                
                # Create new session
                connector = aiohttp.TCPConnector(
                    limit=10,  # Connection pool limit
                    ttl_dns_cache=300,  # DNS cache TTL
                    use_dns_cache=True,
                    keepalive_timeout=self.keep_alive_duration
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=self.request_timeout,
                    connect=self.connection_timeout
                )
                
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                logger.info("Created new HTTP session with connection pooling")
            
            self._last_activity = now
            return self._session
    
    async def _make_request_with_retry(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        
        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()
                url = f"{self.base_url}{endpoint}"
                
                if method == "GET":
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.json()
                else:
                    async with session.post(url, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except aiohttp.ClientError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error in request: {e}")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    async def _perform_health_check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()
        
        try:
            # Simple health check - list models
            session = await self._get_session()
            url = f"{self.base_url}/api/tags"
            
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                response_time = time.time() - start_time
                
                # Determine status based on response time
                if response_time < 1.0:
                    status = OllamaStatus.HEALTHY
                elif response_time < 5.0:
                    status = OllamaStatus.DEGRADED
                else:
                    status = OllamaStatus.DEGRADED
                
                result = HealthCheckResult(
                    status=status,
                    response_time=response_time,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "models_count": len(data.get('models', [])),
                        "endpoint": url
                    }
                )
                
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                status=OllamaStatus.UNAVAILABLE,
                response_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                error="Request timeout"
            )
            
        except Exception as e:
            result = HealthCheckResult(
                status=OllamaStatus.UNAVAILABLE,
                response_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )
        
        # Update status and history
        self._current_status = result.status
        self._last_health_check = result
        self._health_history.append(result)
        
        # Keep only last 100 health checks
        if len(self._health_history) > 100:
            self._health_history = self._health_history[-100:]
        
        logger.debug(f"Health check result: {result.status.value} ({result.response_time:.2f}s)")
        
        return result
    
    async def _background_health_monitor(self):
        """Background task for continuous health monitoring"""
        logger.info("Starting background health monitoring")
        
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)
                
                # Only perform health check if there's been recent activity
                now = datetime.now(timezone.utc)
                if (now - self._last_activity).seconds < self.keep_alive_duration:
                    await self._perform_health_check()
                else:
                    logger.debug("Skipping health check due to inactivity")
                    
        except asyncio.CancelledError:
            logger.info("Background health monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in background health monitoring: {e}")
    
    def _process_generation_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Ollama generation response"""
        
        # Extract content
        content = response_data.get('response', '')
        
        # Extract metadata
        metadata = {
            'model': response_data.get('model', 'unknown'),
            'created_at': response_data.get('created_at'),
            'done': response_data.get('done', True),
            'total_duration': response_data.get('total_duration'),
            'load_duration': response_data.get('load_duration'),
            'prompt_eval_count': response_data.get('prompt_eval_count'),
            'prompt_eval_duration': response_data.get('prompt_eval_duration'),
            'eval_count': response_data.get('eval_count'),
            'eval_duration': response_data.get('eval_duration'),
        }
        
        return {
            'response': content,
            'metadata': metadata,
            'raw_response': response_data
        }