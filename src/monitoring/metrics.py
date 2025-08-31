"""
Monitoring and Metrics Collection

Comprehensive monitoring system with Prometheus metrics,
performance tracking, and system health monitoring.

Per requirements.md: Complete implementation, no placeholders.
"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """Individual metric data point"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime

@dataclass
class HealthStatus:
    """System component health status"""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time: Optional[float]
    error_count: int
    details: Dict[str, Any]

class MetricsCollector:
    """
    Advanced metrics collection and monitoring system
    
    Features:
    - Prometheus metrics integration
    - Real-time performance tracking
    - System health monitoring
    - Alert generation based on thresholds
    - Historical data analysis
    - Custom metric aggregation
    """
    
    def __init__(self):
        # Prometheus registry
        self.registry = CollectorRegistry()
        
        # Define Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Internal metric storage
        self.metrics_buffer = deque(maxlen=10000)  # Rolling buffer
        self.health_status = {}
        self.performance_counters = defaultdict(int)
        self.timing_histograms = defaultdict(list)
        
        # Alert thresholds
        self.alert_thresholds = {
            'response_time_ms': 5000,  # 5 seconds
            'error_rate_percent': 10,   # 10%
            'memory_usage_percent': 85, # 85%
            'disk_usage_percent': 90    # 90%
        }
        
        # Background monitoring
        self._monitoring_task = None
        self._start_background_monitoring()
        
        logger.info("Metrics Collector initialized with Prometheus integration")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        
        # Request counters
        self.request_counter = Counter(
            'mcp_requests_total',
            'Total number of MCP requests',
            ['model', 'category', 'status'],
            registry=self.registry
        )
        
        # Response time histogram
        self.response_time_histogram = Histogram(
            'mcp_response_time_seconds',
            'Response time in seconds',
            ['model', 'category'],
            registry=self.registry
        )
        
        # Model routing distribution
        self.model_routing_counter = Counter(
            'mcp_model_routing_total',
            'Total model routing decisions',
            ['model', 'reason'],
            registry=self.registry
        )
        
        # Interview process metrics
        self.interview_duration_histogram = Histogram(
            'mcp_interview_duration_seconds',
            'Interview process duration in seconds',
            ['category'],
            registry=self.registry
        )
        
        # Bias detection counter
        self.bias_detection_counter = Counter(
            'mcp_bias_detections_total',
            'Total bias detections',
            ['bias_type', 'category'],
            registry=self.registry
        )
        
        # Ollama health gauge
        self.ollama_health_gauge = Gauge(
            'mcp_ollama_health_status',
            'Ollama health status (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
        
        # Category scanning metrics
        self.category_scan_histogram = Histogram(
            'mcp_category_scan_duration_seconds',
            'Category scanning duration in seconds',
            registry=self.registry
        )
        
        # System resource gauges
        self.system_memory_gauge = Gauge(
            'mcp_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_cpu_gauge = Gauge(
            'mcp_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
    
    async def track_request(self, request_data: Dict[str, Any]):
        """Track a request through the system"""
        try:
            model = request_data.get('model', 'unknown')
            category = request_data.get('category', 'unknown')
            status = request_data.get('status', 'success')
            response_time = request_data.get('response_time', 0)
            
            # Update Prometheus metrics
            self.request_counter.labels(
                model=model, 
                category=category, 
                status=status
            ).inc()
            
            self.response_time_histogram.labels(
                model=model, 
                category=category
            ).observe(response_time / 1000.0)  # Convert ms to seconds
            
            # Update internal counters
            self.performance_counters[f'requests_{model}'] += 1
            self.performance_counters[f'requests_{category}'] += 1
            
            # Store metric data
            metric = MetricData(
                name='request_tracked',
                value=1,
                labels={'model': model, 'category': category, 'status': status},
                timestamp=datetime.now(timezone.utc)
            )
            self.metrics_buffer.append(metric)
            
            logger.debug(f"Tracked request: {model}/{category} in {response_time}ms")
            
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
    
    async def track_routing(self, routing_data: Dict[str, Any]):
        """Track routing decision"""
        try:
            model = routing_data.get('model', 'unknown')
            reason = routing_data.get('reasoning', 'unknown')[:50]  # Truncate long reasons
            category = routing_data.get('category', 'unknown')
            confidence = routing_data.get('confidence', 0)
            
            # Update Prometheus metrics
            self.model_routing_counter.labels(
                model=model, 
                reason=reason
            ).inc()
            
            # Track confidence distribution
            metric = MetricData(
                name='routing_confidence',
                value=confidence,
                labels={'model': model, 'category': category},
                timestamp=datetime.now(timezone.utc)
            )
            self.metrics_buffer.append(metric)
            
            # Track 3% preference rule application
            if routing_data.get('ollama_preference_applied'):
                self.performance_counters['ollama_preference_applied'] += 1
            
            logger.debug(f"Tracked routing: {model} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error tracking routing: {e}")
    
    async def track_interview_performance(self, interview_data: Dict[str, Any]):
        """Track interview process performance"""
        try:
            category = interview_data.get('category', 'unknown')
            duration = interview_data.get('duration', 0)
            bias_detected = interview_data.get('bias_detected', False)
            bias_types = interview_data.get('bias_types', [])
            
            # Update Prometheus metrics
            self.interview_duration_histogram.labels(category=category).observe(duration / 1000.0)
            
            if bias_detected:
                for bias_type in bias_types:
                    self.bias_detection_counter.labels(
                        bias_type=bias_type,
                        category=category
                    ).inc()
            
            # Track internal metrics
            self.timing_histograms[f'interview_{category}'].append(duration)
            if len(self.timing_histograms[f'interview_{category}']) > 100:
                self.timing_histograms[f'interview_{category}'].pop(0)
            
            logger.debug(f"Tracked interview: {category} in {duration}ms")
            
        except Exception as e:
            logger.error(f"Error tracking interview: {e}")
    
    async def track_category_scan(self, scan_data: Dict[str, Any]):
        """Track category scanning performance"""
        try:
            duration = scan_data.get('duration', 0)
            categories_found = scan_data.get('categories_found', 0)
            confidence = scan_data.get('top_confidence', 0)
            
            # Update Prometheus metrics
            self.category_scan_histogram.observe(duration / 1000.0)
            
            # Track scan effectiveness
            metric = MetricData(
                name='category_scan_effectiveness',
                value=categories_found * confidence,
                labels={'categories_found': str(categories_found)},
                timestamp=datetime.now(timezone.utc)
            )
            self.metrics_buffer.append(metric)
            
            logger.debug(f"Tracked category scan: {categories_found} categories in {duration}ms")
            
        except Exception as e:
            logger.error(f"Error tracking category scan: {e}")
    
    async def update_health_status(self, component: str, status_data: Dict[str, Any]):
        """Update component health status"""
        try:
            status = status_data.get('status', 'unknown')
            response_time = status_data.get('response_time', None)
            error_count = status_data.get('error_count', 0)
            details = status_data.get('details', {})
            
            health_status = HealthStatus(
                component=component,
                status=status,
                last_check=datetime.now(timezone.utc),
                response_time=response_time,
                error_count=error_count,
                details=details
            )
            
            self.health_status[component] = health_status
            
            # Update Prometheus gauges for specific components
            if component == 'ollama':
                health_value = 1 if status == 'healthy' else 0
                self.ollama_health_gauge.set(health_value)
            
            # Check for alerts
            await self._check_health_alerts(component, health_status)
            
            logger.debug(f"Updated health status: {component} = {status}")
            
        except Exception as e:
            logger.error(f"Error updating health status: {e}")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            now = datetime.now(timezone.utc)
            
            # Calculate performance metrics
            total_requests = sum(
                count for key, count in self.performance_counters.items() 
                if key.startswith('requests_')
            )
            
            # Calculate average response times
            avg_response_times = {}
            for key, times in self.timing_histograms.items():
                if times:
                    avg_response_times[key] = sum(times) / len(times)
            
            # Get recent metrics (last hour)
            one_hour_ago = now - timedelta(hours=1)
            recent_metrics = [
                m for m in self.metrics_buffer 
                if m.timestamp >= one_hour_ago
            ]
            
            # Calculate error rates
            error_metrics = [m for m in recent_metrics if m.labels.get('status') == 'error']
            error_rate = len(error_metrics) / max(len(recent_metrics), 1) * 100
            
            # System health summary
            health_summary = {
                component: {
                    'status': status.status,
                    'last_check': status.last_check.isoformat(),
                    'response_time': status.response_time,
                    'error_count': status.error_count
                }
                for component, status in self.health_status.items()
            }
            
            return {
                'timestamp': now.isoformat(),
                'total_requests': total_requests,
                'error_rate_percent': error_rate,
                'average_response_times': avg_response_times,
                'performance_counters': dict(self.performance_counters),
                'health_status': health_summary,
                'recent_metrics_count': len(recent_metrics),
                'buffer_utilization': len(self.metrics_buffer) / self.metrics_buffer.maxlen * 100,
                'alerts_active': await self._get_active_alerts()
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return ""
    
    async def get_performance_report(self, time_window: str = "1h") -> Dict[str, Any]:
        """Generate performance report for specified time window"""
        try:
            if time_window == "1h":
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            elif time_window == "24h":
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            elif time_window == "7d":
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            else:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            
            # Filter metrics by time window
            window_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff]
            
            # Group by model
            model_metrics = defaultdict(list)
            for metric in window_metrics:
                if 'model' in metric.labels:
                    model_metrics[metric.labels['model']].append(metric)
            
            # Calculate statistics
            model_stats = {}
            for model, metrics in model_metrics.items():
                request_metrics = [m for m in metrics if m.name == 'request_tracked']
                confidence_metrics = [m for m in metrics if m.name == 'routing_confidence']
                
                if request_metrics:
                    success_rate = len([m for m in request_metrics if m.labels.get('status') == 'success']) / len(request_metrics) * 100
                else:
                    success_rate = 0
                
                if confidence_metrics:
                    avg_confidence = sum(m.value for m in confidence_metrics) / len(confidence_metrics)
                else:
                    avg_confidence = 0
                
                model_stats[model] = {
                    'requests': len(request_metrics),
                    'success_rate': success_rate,
                    'average_confidence': avg_confidence
                }
            
            return {
                'time_window': time_window,
                'total_metrics': len(window_metrics),
                'model_statistics': model_stats,
                'system_health': {
                    component: status.status 
                    for component, status in self.health_status.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    async def _check_health_alerts(self, component: str, health_status: HealthStatus):
        """Check for health-related alerts"""
        try:
            alerts = []
            
            # Check response time
            if health_status.response_time and health_status.response_time > self.alert_thresholds['response_time_ms']:
                alerts.append({
                    'type': 'high_response_time',
                    'component': component,
                    'value': health_status.response_time,
                    'threshold': self.alert_thresholds['response_time_ms']
                })
            
            # Check error count
            if health_status.error_count > 10:  # More than 10 errors
                alerts.append({
                    'type': 'high_error_count',
                    'component': component,
                    'value': health_status.error_count,
                    'threshold': 10
                })
            
            # Check component-specific thresholds
            if component == 'ollama' and health_status.status == 'unavailable':
                alerts.append({
                    'type': 'service_unavailable',
                    'component': component,
                    'status': health_status.status
                })
            
            # Log alerts
            for alert in alerts:
                logger.warning(f"Health alert: {alert}")
            
        except Exception as e:
            logger.error(f"Error checking health alerts: {e}")
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        active_alerts = []
        
        try:
            # Check recent error rates
            recent_metrics = [
                m for m in self.metrics_buffer 
                if m.timestamp >= datetime.now(timezone.utc) - timedelta(minutes=10)
            ]
            
            if recent_metrics:
                error_metrics = [m for m in recent_metrics if m.labels.get('status') == 'error']
                error_rate = len(error_metrics) / len(recent_metrics) * 100
                
                if error_rate > self.alert_thresholds['error_rate_percent']:
                    active_alerts.append({
                        'type': 'high_error_rate',
                        'value': error_rate,
                        'threshold': self.alert_thresholds['error_rate_percent']
                    })
            
            # Check component health
            for component, status in self.health_status.items():
                if status.status in ['degraded', 'unavailable']:
                    active_alerts.append({
                        'type': 'component_unhealthy',
                        'component': component,
                        'status': status.status,
                        'last_check': status.last_check.isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
        
        return active_alerts
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        try:
            self._monitoring_task = asyncio.create_task(self._background_monitor())
            logger.info("Started background monitoring")
        except Exception as e:
            logger.error(f"Error starting background monitoring: {e}")
    
    async def _background_monitor(self):
        """Background monitoring loop"""
        try:
            while True:
                await asyncio.sleep(60)  # Run every minute
                
                # Update system resource metrics
                await self._update_system_resources()
                
                # Clean old metrics
                await self._cleanup_old_metrics()
                
                # Generate periodic health summary
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    await self._log_health_summary()
                
        except asyncio.CancelledError:
            logger.info("Background monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in background monitoring: {e}")
    
    async def _update_system_resources(self):
        """Update system resource metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_gauge.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_memory_gauge.set(memory_percent)
            
            # Store as metrics
            for name, value in [('cpu_usage', cpu_percent), ('memory_usage', memory_percent)]:
                metric = MetricData(
                    name=f'system_{name}',
                    value=value,
                    labels={'resource': name},
                    timestamp=datetime.now(timezone.utc)
                )
                self.metrics_buffer.append(metric)
            
        except ImportError:
            # psutil not available, skip system monitoring
            pass
        except Exception as e:
            logger.error(f"Error updating system resources: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics from buffer"""
        try:
            # Remove metrics older than 24 hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            
            # This is handled automatically by deque maxlen, but we can also clean by time
            self.metrics_buffer = deque(
                (m for m in self.metrics_buffer if m.timestamp >= cutoff),
                maxlen=self.metrics_buffer.maxlen
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    async def _log_health_summary(self):
        """Log periodic health summary"""
        try:
            healthy_components = sum(1 for s in self.health_status.values() if s.status == 'healthy')
            total_components = len(self.health_status)
            
            logger.info(
                f"Health Summary: {healthy_components}/{total_components} components healthy, "
                f"Buffer: {len(self.metrics_buffer)} metrics, "
                f"Requests: {self.performance_counters.get('requests_total', 0)}"
            )
            
        except Exception as e:
            logger.error(f"Error logging health summary: {e}")
    
    async def shutdown(self):
        """Shutdown metrics collection"""
        try:
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Metrics collector shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during metrics collector shutdown: {e}")