# Cost Savings Analysis System

## Overview

The Cost Savings Calculator provides comprehensive financial analysis for the MCP Router System, calculating cost savings from routing requests to Ollama instead of Claude API. This system helps organizations understand the ROI of their Ollama deployment and optimize routing decisions for maximum cost efficiency.

## Features

### 1. Real-Time Cost Tracking
- Tracks actual costs for Claude API usage vs Ollama infrastructure costs
- Provides detailed breakdowns by time window (hourly, daily, weekly, monthly)
- Calculates precise savings based on actual routing decisions

### 2. ROI Analysis
- Monthly and annual ROI calculations
- Payback period analysis for Ollama deployment
- Infrastructure cost vs savings analysis
- Break-even point calculations

### 3. Category-Specific Analysis
- Cost breakdown by the 76 MCP categories
- Identifies categories with highest savings potential
- Recommends optimization opportunities
- Tracks Ollama suitability by category type

### 4. Predictive Projections
- Annual savings projections with multiple scenarios
- Growth factor analysis (conservative, moderate, aggressive)
- Trend analysis based on historical data
- Capacity planning recommendations

## MCP Tool Integration

### `calculate_cost_savings` Tool

Available through the MCP interface for Claude Code integration:

```python
# Basic usage
await mcp_server.calculate_cost_savings(
    time_window="24h",
    report_format="json"
)

# Full comprehensive report
await mcp_server.calculate_cost_savings(
    time_window="7d",
    report_format="full_report"
)
```

**Parameters:**
- `time_window`: Analysis period (1h, 24h, 7d, 30d)
- `report_format`: Output format (json, full_report)

**Returns:**
- Cost metrics with detailed breakdown
- ROI analysis and projections
- Optimization recommendations

## Cost Calculation Model

### Claude API Costs
- **Claude-3-Sonnet**: $3.00 per 1M tokens
- **Claude-3-Haiku**: $0.25 per 1M tokens
- Average tokens per request: 150 (configurable)

### Ollama Infrastructure Costs
- **Server Hosting**: $50/month
- **Storage**: $10/month
- **Bandwidth**: $5/month
- **Maintenance**: $20/month
- **Total Infrastructure**: $85/month

### Cost Per Request Calculation
```python
# Claude cost per request
claude_cost = (tokens_per_request * claude_rate_per_token)

# Ollama cost per request
ollama_cost = (monthly_infrastructure / monthly_requests) + 
              (tokens_per_request * ollama_marginal_cost)

# Savings per request
savings = claude_cost - ollama_cost
```

## Usage Analytics

### Daily Metrics
- Total requests processed
- Claude vs Ollama distribution
- Cost breakdown and savings
- Utilization rates

### Weekly Trends
- Routing pattern analysis
- Cost trend identification
- Performance vs cost correlation
- Category-specific insights

### Monthly ROI
- Gross savings calculation
- Net savings after infrastructure costs
- ROI percentage
- Payback period analysis

### Annual Projections
- Three scenario modeling (conservative, moderate, aggressive)
- Growth factor application
- Infrastructure scaling considerations
- Long-term cost optimization

## Cost Optimization Recommendations

### Automatic Recommendations
The system generates intelligent recommendations based on usage patterns:

1. **Category Optimization**
   - Identifies categories with high Claude usage but good Ollama suitability
   - Suggests routing adjustments for maximum savings

2. **Volume-Based Recommendations**
   - High-volume workload optimization
   - Dedicated instance recommendations
   - Scaling suggestions

3. **ROI Improvement**
   - Infrastructure cost optimization
   - Performance vs cost trade-offs
   - Capacity utilization improvements

### Example Recommendations
```json
{
  "recommendations": [
    "Increase Ollama routing for code_generation_simple (current: 60%, suitable: 85%)",
    "High volume detected - consider dedicated Ollama instance for better performance",
    "Consider optimizing Ollama routing for categories with high Claude usage"
  ]
}
```

## Integration with Monitoring

### Prometheus Metrics
- `mcp_cost_savings_total`: Total cost savings
- `mcp_claude_cost_total`: Total Claude API costs
- `mcp_ollama_cost_total`: Total Ollama infrastructure costs
- `mcp_roi_percentage`: Current ROI percentage

### Grafana Dashboards
- Cost savings overview dashboard
- ROI trending analysis
- Category cost breakdown
- Optimization opportunity identification

## API Examples

### Basic Cost Analysis
```python
from src.analytics.cost_savings import CostSavingsCalculator

calculator = CostSavingsCalculator(db_manager)

# Get 24-hour savings
savings = await calculator.calculate_current_savings("24h")
print(f"24h savings: ${savings.total_savings}")
print(f"Savings rate: {savings.savings_percentage}%")
```

### Monthly ROI Analysis
```python
roi_data = await calculator.calculate_monthly_roi()
print(f"Monthly ROI: {roi_data['roi_percentage']}%")
print(f"Payback period: {roi_data['payback_period_months']} months")
```

### Category Breakdown
```python
breakdown = await calculator.get_category_cost_breakdown("7d")
for category, metrics in breakdown.items():
    print(f"{category}: ${metrics['current_cost']} (potential savings: ${metrics['potential_savings']})")
```

### Comprehensive Report
```python
# JSON report
json_report = await calculator.generate_cost_report("json")
data = json.loads(json_report)

# Markdown report
md_report = await calculator.generate_cost_report("markdown")
```

## Configuration

### Cost Model Customization
```python
# Update Claude pricing
calculator.cost_configs['claude-3-sonnet'] = ModelCostConfig(
    claude_cost_per_token=Decimal('0.004') / 1000,  # $4 per 1M tokens
    ollama_cost_per_token=Decimal('0.0001') / 1000,
    avg_tokens_per_request=200
)

# Update infrastructure costs
calculator.infrastructure_costs.update({
    'ollama_server_monthly': Decimal('75.00'),  # Higher performance server
    'storage_monthly': Decimal('15.00'),        # More storage
})
```

### Environment Variables
```bash
# Cost calculation settings
CLAUDE_COST_PER_TOKEN=0.003
OLLAMA_INFRASTRUCTURE_MONTHLY=85.00
AVERAGE_TOKENS_PER_REQUEST=150

# ROI calculation settings
SETUP_COST=500.00
PAYBACK_THRESHOLD_MONTHS=12
```

## Benefits

### Financial Benefits
- **Immediate Cost Visibility**: Real-time cost tracking and savings calculation
- **ROI Justification**: Clear ROI metrics for Ollama deployment decisions
- **Budget Planning**: Accurate projections for annual cost planning
- **Optimization Guidance**: Data-driven recommendations for cost reduction

### Operational Benefits
- **Resource Optimization**: Better understanding of infrastructure utilization
- **Performance vs Cost**: Analysis of performance trade-offs
- **Capacity Planning**: Informed decisions about scaling infrastructure
- **Trend Analysis**: Historical data for pattern identification

### Strategic Benefits
- **Investment Justification**: Clear business case for Ollama adoption
- **Competitive Analysis**: Cost comparison with alternative solutions
- **Long-term Planning**: Annual projections for strategic planning
- **Risk Assessment**: Understanding of cost variability and factors

## Integration with MCP Router

The Cost Savings Calculator is fully integrated into the MCP Router system:

1. **Automatic Tracking**: All routing decisions are automatically tracked for cost analysis
2. **Real-time Updates**: Cost metrics update in real-time as requests are processed  
3. **Historical Analysis**: Complete historical data for trend analysis
4. **Predictive Modeling**: Future cost projections based on current patterns

This comprehensive cost analysis system enables organizations to make informed decisions about their AI infrastructure investments and optimize their routing strategies for maximum cost efficiency while maintaining quality and performance standards.