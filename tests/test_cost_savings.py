"""
Cost Savings Calculator Tests

Tests for the cost savings analysis and ROI calculation system.
Validates financial metrics and cost optimization recommendations.

Per requirements.md: Complete implementation, no placeholders.
"""
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from src.analytics.cost_savings import CostSavingsCalculator, CostMetrics, ModelCostConfig

class TestCostSavingsCalculator:
    """Test cases for CostSavingsCalculator"""
    
    @pytest.fixture
    async def calculator(self, mock_database):
        """Create CostSavingsCalculator with mocked database"""
        return CostSavingsCalculator(mock_database)
    
    @pytest.mark.asyncio
    async def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator is not None
        assert len(calculator.cost_configs) >= 2
        assert 'claude-3-sonnet' in calculator.cost_configs
        assert len(calculator.infrastructure_costs) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_current_savings_24h(self, calculator):
        """Test current savings calculation for 24-hour window"""
        result = await calculator.calculate_current_savings("24h")
        
        assert isinstance(result, CostMetrics)
        assert result.total_requests >= 0
        assert result.claude_requests >= 0
        assert result.ollama_requests >= 0
        assert result.total_savings >= 0
        assert 0 <= result.savings_percentage <= 100
    
    @pytest.mark.asyncio
    async def test_calculate_current_savings_weekly(self, calculator):
        """Test current savings calculation for weekly window"""
        result = await calculator.calculate_current_savings("7d")
        
        assert isinstance(result, CostMetrics)
        assert result.claude_cost >= 0
        assert result.ollama_cost >= 0
        assert result.avg_cost_per_request >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_monthly_roi(self, calculator):
        """Test monthly ROI calculation"""
        roi_data = await calculator.calculate_monthly_roi()
        
        assert 'monthly_savings_gross' in roi_data
        assert 'monthly_infrastructure_cost' in roi_data
        assert 'monthly_savings_net' in roi_data
        assert 'roi_percentage' in roi_data
        assert 'payback_period_months' in roi_data
        assert 'ollama_utilization_rate' in roi_data
        
        # ROI should be a number
        assert isinstance(roi_data['roi_percentage'], (int, float))
        assert roi_data['payback_period_months'] >= 0
    
    @pytest.mark.asyncio
    async def test_category_cost_breakdown(self, calculator):
        """Test category-specific cost breakdown"""
        breakdown = await calculator.get_category_cost_breakdown("7d")
        
        assert isinstance(breakdown, dict)
        
        for category, data in breakdown.items():
            assert 'total_requests' in data
            assert 'claude_requests' in data
            assert 'ollama_requests' in data
            assert 'current_cost' in data
            assert 'potential_savings' in data
            assert data['total_requests'] == data['claude_requests'] + data['ollama_requests']
            assert data['current_cost'] >= 0
    
    @pytest.mark.asyncio
    async def test_project_annual_savings(self, calculator):
        """Test annual savings projections"""
        projections = await calculator.project_annual_savings()
        
        assert 'current_weekly_trend' in projections
        assert 'annual_projections' in projections
        assert 'infrastructure_cost_annual' in projections
        
        # Check projection scenarios
        scenarios = projections['annual_projections']
        assert 'conservative' in scenarios
        assert 'moderate' in scenarios
        assert 'aggressive' in scenarios
        
        # Aggressive should be higher than conservative
        conservative = scenarios['conservative']['annual_net_savings']
        aggressive = scenarios['aggressive']['annual_net_savings']
        if conservative > 0 and aggressive > 0:
            assert aggressive >= conservative
    
    @pytest.mark.asyncio
    async def test_claude_cost_calculation(self, calculator):
        """Test Claude API cost calculation"""
        # Test with 100 requests
        cost = calculator._calculate_claude_costs(100)
        
        assert isinstance(cost, Decimal)
        assert cost >= 0
        
        # More requests should cost more
        cost_200 = calculator._calculate_claude_costs(200)
        assert cost_200 > cost
    
    @pytest.mark.asyncio
    async def test_ollama_cost_calculation(self, calculator):
        """Test Ollama infrastructure cost calculation"""
        # Test with 100 requests
        cost = calculator._calculate_ollama_costs(100)
        
        assert isinstance(cost, Decimal)
        assert cost >= 0
        
        # Ollama should be significantly cheaper than Claude
        claude_cost = calculator._calculate_claude_costs(100)
        assert cost < claude_cost
    
    @pytest.mark.asyncio
    async def test_cost_comparison_accuracy(self, calculator):
        """Test accuracy of cost comparisons"""
        requests = [10, 50, 100, 500, 1000]
        
        for request_count in requests:
            claude_cost = calculator._calculate_claude_costs(request_count)
            ollama_cost = calculator._calculate_ollama_costs(request_count)
            
            # Ollama should always be cheaper for same request count
            assert ollama_cost <= claude_cost
            
            # Costs should scale linearly with requests
            if request_count > 10:
                claude_cost_small = calculator._calculate_claude_costs(10)
                expected_ratio = request_count / 10
                actual_ratio = claude_cost / claude_cost_small
                # Allow for small rounding differences
                assert abs(actual_ratio - expected_ratio) < 0.01
    
    @pytest.mark.asyncio
    async def test_generate_cost_report_json(self, calculator):
        """Test JSON cost report generation"""
        report = await calculator.generate_cost_report("json")
        
        assert isinstance(report, str)
        
        # Should be valid JSON
        import json
        data = json.loads(report)
        
        assert 'generated_at' in data
        assert 'summary' in data
        assert 'detailed_metrics' in data
        assert 'category_analysis' in data
        assert 'recommendations' in data
        
        # Check summary structure
        summary = data['summary']
        assert 'daily_savings' in summary
        assert 'weekly_savings' in summary
        assert 'monthly_roi_percentage' in summary
        assert 'ollama_utilization_rate' in summary
    
    @pytest.mark.asyncio
    async def test_generate_cost_report_markdown(self, calculator):
        """Test Markdown cost report generation"""
        report = await calculator.generate_cost_report("markdown")
        
        assert isinstance(report, str)
        assert "# MCP Router Cost Savings Report" in report
        assert "## Executive Summary" in report
        assert "## Recommendations" in report
        assert "## Category Breakdown" in report
        assert "$" in report  # Should contain cost figures
    
    @pytest.mark.asyncio
    async def test_recommendations_generation(self, calculator):
        """Test cost optimization recommendations"""
        # Mock category data with different scenarios
        category_data = {
            'high_claude_suitable_for_ollama': {
                'total_requests': 100,
                'claude_requests': 80,
                'ollama_requests': 20,
                'ollama_suitability': 0.8  # High suitability but low usage
            },
            'already_optimized': {
                'total_requests': 100,
                'claude_requests': 20,
                'ollama_requests': 80,
                'ollama_suitability': 0.9
            }
        }
        
        roi_data = {'roi_percentage': 150}  # Good ROI
        
        recommendations = calculator._generate_recommendations(category_data, roi_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend increasing Ollama usage for suitable categories
        rec_text = ' '.join(recommendations)
        assert 'high_claude_suitable_for_ollama' in rec_text or 'Increase Ollama routing' in rec_text
    
    @pytest.mark.asyncio
    async def test_high_volume_recommendations(self, calculator):
        """Test recommendations for high-volume scenarios"""
        category_data = {
            'high_volume_category': {
                'total_requests': 2000,  # High volume
                'claude_requests': 1000,
                'ollama_requests': 1000,
                'ollama_suitability': 0.7
            }
        }
        
        roi_data = {'roi_percentage': 200}
        
        recommendations = calculator._generate_recommendations(category_data, roi_data)
        
        # Should recommend dedicated instance for high volume
        rec_text = ' '.join(recommendations)
        assert 'dedicated' in rec_text.lower() or 'high volume' in rec_text.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, calculator):
        """Test error handling in various scenarios"""
        # Test with invalid time window
        result = await calculator.calculate_current_savings("invalid")
        
        # Should not crash and return valid metrics
        assert isinstance(result, CostMetrics)
        assert result.total_requests >= 0
        
        # Test report generation with errors
        # Mock database to raise an exception
        if hasattr(calculator, 'db_manager') and calculator.db_manager:
            original_method = calculator._get_routing_data
            
            async def failing_method(*args, **kwargs):
                raise Exception("Database error")
            
            calculator._get_routing_data = failing_method
            
            # Should handle error gracefully
            report = await calculator.generate_cost_report("json")
            assert "Error generating report" in report
            
            # Restore original method
            calculator._get_routing_data = original_method
    
    @pytest.mark.asyncio
    async def test_cost_config_customization(self, calculator):
        """Test customization of cost configurations"""
        # Original config
        original_config = calculator.cost_configs['claude-3-sonnet']
        original_cost = calculator._calculate_claude_costs(100)
        
        # Update config with higher cost
        new_config = ModelCostConfig(
            claude_cost_per_token=original_config.claude_cost_per_token * 2,
            ollama_cost_per_token=original_config.ollama_cost_per_token,
            avg_tokens_per_request=original_config.avg_tokens_per_request
        )
        
        calculator.cost_configs['claude-3-sonnet'] = new_config
        
        # Cost should be higher
        new_cost = calculator._calculate_claude_costs(100)
        assert new_cost > original_cost
        
        # Restore original config
        calculator.cost_configs['claude-3-sonnet'] = original_config
    
    @pytest.mark.asyncio
    async def test_infrastructure_cost_impact(self, calculator):
        """Test impact of infrastructure costs on ROI"""
        # Get ROI with current infrastructure costs
        roi_data = await calculator.calculate_monthly_roi()
        original_roi = roi_data.get('roi_percentage', 0)
        
        # Double infrastructure costs
        original_costs = calculator.infrastructure_costs.copy()
        for key in calculator.infrastructure_costs:
            calculator.infrastructure_costs[key] *= 2
        
        # ROI should be lower
        new_roi_data = await calculator.calculate_monthly_roi()
        new_roi = new_roi_data.get('roi_percentage', 0)
        
        if original_roi > 0:
            assert new_roi < original_roi
        
        # Restore original costs
        calculator.infrastructure_costs = original_costs
    
    @pytest.mark.asyncio
    async def test_time_window_variations(self, calculator):
        """Test different time window calculations"""
        time_windows = ["1h", "24h", "7d", "30d"]
        
        results = {}
        for window in time_windows:
            results[window] = await calculator.calculate_current_savings(window)
        
        # All should return valid metrics
        for window, result in results.items():
            assert isinstance(result, CostMetrics)
            assert result.total_requests >= 0
            assert result.total_savings >= 0
    
    @pytest.mark.asyncio
    async def test_decimal_precision(self, calculator):
        """Test decimal precision in cost calculations"""
        # Test with small amounts
        cost = calculator._calculate_claude_costs(1)
        
        # Should maintain proper precision
        assert isinstance(cost, Decimal)
        assert cost.as_tuple().exponent >= -4  # At least 4 decimal places
        
        # Test rounding consistency
        cost1 = calculator._calculate_claude_costs(1)
        cost2 = calculator._calculate_claude_costs(2)
        
        # Should be exactly double (within decimal precision)
        assert abs(cost2 - (cost1 * 2)) < Decimal('0.0001')