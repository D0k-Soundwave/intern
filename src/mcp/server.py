"""
MCP Router Server Implementation

Provides MCP server that handles routing between Claude Code and Ollama
based on interview-driven capability assessment. No API keys required.

Per requirements.md: Complete implementation, no placeholders.
"""
import asyncio
import json
import uuid
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ..routing.category_scanner import CategoryScanner
from ..interview.process_manager import InterviewProcessManager
from ..response.transformer import MCPResponseHandler
from ..adapters.ollama_client import OllamaClient
from ..database.connection import DatabaseManager
from ..monitoring.metrics import MetricsCollector
from ..analytics.cost_savings import CostSavingsCalculator

logger = logging.getLogger(__name__)


class MCPRouterServer:
    """
    MCP Server providing routing capabilities to Claude Code
    
    Handles stdio/JSON-RPC communication and routes requests between
    Claude Code and Ollama based on interview results.
    """

    def __init__(self):
        self.server = Server("claude-router-mcp")
        self.category_scanner = CategoryScanner()
        self.interview_manager = InterviewProcessManager()
        self.response_handler = MCPResponseHandler()
        self.ollama_client = OllamaClient()
        self.db_manager = DatabaseManager()
        self.metrics = MetricsCollector()
        self.cost_calculator = CostSavingsCalculator(self.db_manager)
        self._register_tools()
        
        logger.info("MCP Router Server initialized")

    def _register_tools(self):
        """Register MCP tools for Claude Code to use"""

        @self.server.tool()
        async def analyze_and_route(input_text: str, context: Optional[Dict] = None) -> str:
            """
            Analyze input from Claude Code and determine routing
            
            Args:
                input_text: The user's request from Claude Code
                context: Additional context (user_id, preferences, etc.)
            
            Returns:
                JSON string with routing decision and response
            """
            try:
                request_id = str(uuid.uuid4())
                context = context or {}
                
                logger.info(f"Processing request {request_id}: {input_text[:100]}...")
                
                # Step 1: Scan input for all 76 categories
                category_result = await self.category_scanner.scan_claude_input(input_text)
                
                # Step 2: Conduct interview for offloading decision
                interview_result = await self.interview_manager.conduct_interview(
                    {'input_text': input_text, 'context': context, 'request_id': request_id},
                    category_result.category
                )
                
                # Step 3: Track routing decision
                await self.metrics.track_routing({
                    'request_id': request_id,
                    'category': category_result.category['name'],
                    'model': interview_result.recommended_model,
                    'complexity_score': interview_result.complexity_score,
                    'confidence': interview_result.confidence,
                    'reasoning': interview_result.reasoning
                })
                
                # Step 4: Route based on interview results
                if interview_result.recommended_model == 'ollama':
                    # Offload to Ollama with response transformation
                    logger.info(f"Routing request {request_id} to Ollama")
                    
                    ollama_response = await self._execute_ollama(
                        input_text, 
                        category_result.category,
                        context
                    )
                    
                    claude_formatted = await self.response_handler.transform_to_claude(
                        ollama_response,
                        {'input_text': input_text, 'request_id': request_id}
                    )
                    
                    return json.dumps({
                        'action': 'OLLAMA_PROCESSED',
                        'response': claude_formatted,
                        'metadata': {
                            'model': 'ollama',
                            'category': category_result.category['name'],
                            'confidence': interview_result.confidence,
                            'reasoning': interview_result.reasoning
                        }
                    })
                else:
                    # Return to Claude Code for direct processing
                    logger.info(f"Routing request {request_id} to Claude Code")
                    
                    return json.dumps({
                        'action': 'CLAUDE_REQUIRED',
                        'instruction': 'Process directly in Claude Code',
                        'metadata': {
                            'model': 'claude',
                            'category': category_result.category['name'],
                            'confidence': interview_result.confidence,
                            'reasoning': interview_result.reasoning,
                            'request_id': request_id
                        }
                    })
                    
            except Exception as e:
                logger.error(f"Error in analyze_and_route: {e}")
                return json.dumps({
                    'action': 'ERROR',
                    'error': str(e),
                    'fallback': 'CLAUDE_REQUIRED'
                })

        @self.server.tool()
        async def scan_categories(input_text: str) -> str:
            """
            Scan input and identify all applicable categories
            
            Args:
                input_text: The input to categorize
            
            Returns:
                JSON string with category analysis
            """
            try:
                result = await self.category_scanner.scan_claude_input(input_text)
                return json.dumps({
                    'primary_category': result.category,
                    'all_categories': result.all_matches,
                    'confidence_scores': result.confidence_scores,
                    'identified_intents': result.intents
                })
            except Exception as e:
                logger.error(f"Error in scan_categories: {e}")
                return json.dumps({'error': str(e)})

        @self.server.tool()
        async def check_offloading_capability(category: str) -> str:
            """
            Check if a specific category can be offloaded
            
            Args:
                category: Category name to check
            
            Returns:
                JSON string with offloading capability assessment
            """
            try:
                capability = await self._assess_category_offloading(category)
                return json.dumps(capability)
            except Exception as e:
                logger.error(f"Error in check_offloading_capability: {e}")
                return json.dumps({'error': str(e)})

        @self.server.tool()
        async def get_routing_stats() -> str:
            """
            Get current routing statistics
            
            Returns:
                JSON string with routing statistics and performance metrics
            """
            try:
                stats = await self.db_manager.get_routing_analytics('1h')
                return json.dumps(stats)
            except Exception as e:
                logger.error(f"Error in get_routing_stats: {e}")
                return json.dumps({'error': str(e)})

        @self.server.tool()
        async def calculate_cost_savings(time_window: str = "24h", report_format: str = "json") -> str:
            """
            Calculate cost savings from Ollama routing decisions
            
            Args:
                time_window: Time period to analyze (1h, 24h, 7d, 30d)
                report_format: Output format (json, markdown)
            
            Returns:
                JSON string with cost savings analysis and ROI metrics
            """
            try:
                if report_format == "full_report":
                    # Generate comprehensive cost report
                    report = await self.cost_calculator.generate_cost_report("json")
                    return report
                else:
                    # Get specific time window metrics
                    savings = await self.cost_calculator.calculate_current_savings(time_window)
                    roi_data = await self.cost_calculator.calculate_monthly_roi()
                    
                    result = {
                        'time_window': time_window,
                        'cost_metrics': {
                            'total_requests': savings.total_requests,
                            'claude_requests': savings.claude_requests,
                            'ollama_requests': savings.ollama_requests,
                            'claude_cost': float(savings.claude_cost),
                            'ollama_cost': float(savings.ollama_cost),
                            'total_savings': float(savings.total_savings),
                            'savings_percentage': savings.savings_percentage,
                            'avg_cost_per_request': float(savings.avg_cost_per_request)
                        },
                        'roi_analysis': roi_data,
                        'ollama_utilization': savings.ollama_requests / max(savings.total_requests, 1) * 100
                    }
                    
                    return json.dumps(result)
                    
            except Exception as e:
                logger.error(f"Error in calculate_cost_savings: {e}")
                return json.dumps({'error': str(e)})

    async def _execute_ollama(self, input_text: str, category: Dict, context: Dict) -> Dict:
        """Execute request on Ollama with proper lifecycle management"""
        try:
            # Ensure Ollama is alive
            await self.ollama_client.ensure_alive()
            
            # Execute request
            response = await self.ollama_client.generate(
                prompt=input_text,
                model=context.get('model', 'llama3.2'),
                options={
                    'temperature': context.get('temperature', 0.7),
                    'max_tokens': context.get('max_tokens', 2048)
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing Ollama request: {e}")
            raise

    async def _assess_category_offloading(self, category: str) -> Dict:
        """Assess if a category can be offloaded to Ollama"""
        try:
            # Get category info from database
            category_info = await self.db_manager.get_category_info(category)
            
            if not category_info:
                return {
                    'category': category,
                    'can_offload': False,
                    'reason': 'Unknown category'
                }
            
            # Use interview process to assess
            mock_request = {'input_text': f'Sample {category} request'}
            interview_result = await self.interview_manager.conduct_interview(
                mock_request, category_info
            )
            
            return {
                'category': category,
                'can_offload': interview_result.recommended_model == 'ollama',
                'complexity': category_info.get('complexity', 'unknown'),
                'confidence': interview_result.confidence,
                'reasoning': interview_result.reasoning
            }
            
        except Exception as e:
            logger.error(f"Error assessing category {category}: {e}")
            return {
                'category': category,
                'can_offload': False,
                'error': str(e)
            }

    async def run(self):
        """Run the MCP server with stdio transport"""
        try:
            logger.info("Starting MCP Router Server...")
            
            # Initialize database connection
            await self.db_manager.initialize()
            
            # Start server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(read_stream, write_stream)
                
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise
        finally:
            # Cleanup
            await self.db_manager.close()
            logger.info("MCP Router Server stopped")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down MCP Router Server...")
        await self.db_manager.close()
        await self.ollama_client.cleanup()


async def main():
    """Main entry point for MCP server"""
    server = MCPRouterServer()
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())