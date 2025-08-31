# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude-Ollama MCP Integration System that routes requests between Claude Code and Ollama based on interview-driven capability assessment. The system operates WITHOUT Claude API keys, using MCP (Model Context Protocol) for communication.

## Critical Requirements

**ALWAYS READ `Docs/1. requirements.md` BEFORE ANY CODE CHANGES**

### Zero Technical Debt Policy
- **NEVER** write placeholder code, TODOs, FIXMEs, or stub implementations
- **NEVER** use mock data or simulated results in production code
- Every function must be fully implemented with complete error handling
- All files must be < 2000 lines for Claude/Ollama compatibility

## Architecture Components

### 1. MCP Router Server (`src/mcp/server.py`)
- Implements JSON-RPC over stdio communication
- Registers tools: `analyze_and_route`, `scan_categories`, `check_offloading_capability`
- NO direct stdout writes except JSON-RPC responses

### 2. Category System (76 Categories)
- Categories 1-15: Programming & Development
- Categories 16-25: Data & Analytics  
- Categories 26-35: Content & Language
- Categories 36-45: System & Architecture
- Categories 46-55: Business & Strategy
- Categories 56-65: Security & Compliance
- Categories 66-76: Miscellaneous
- Each category has: `name`, `complexity` (low/medium/high)

### 3. Interview-Based Routing
- **RequirementsAnalyzer**: Analyzes what request actually needs
- **ComplexityScorer**: Scores complexity (0-1 scale)
- **CapabilityMatcher**: Matches requirements to model capabilities
- **BiasDetector**: Detects routing bias (70% threshold triggers compensation)
- **OffloadingDecisionEngine**: Makes final decision with 3% local preference for Ollama

### 4. Response Transformation
Ollama responses MUST be transformed to exact Claude format:
- ID format: `msg_{24-char-hex}`
- Structure: `{"id", "type": "message", "role": "assistant", "content": [...], "model": "claude-3-opus-20240229"}`
- Include token usage calculation

## Development Commands

### Linting and Quality Checks
```bash
# Python linting (all must pass with zero errors)
black src/ tests/ --check
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203
mypy src/ --strict --ignore-missing-imports
pylint src/ --fail-under=8.0

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_routing_framework.py -v
pytest tests/test_interview_process.py -v
pytest tests/test_response_transformation.py -v

# Integration tests
pytest tests/integration/ -v --cov=src --cov-report=term-missing
```

### Ollama Management
```bash
# Start Ollama server
ollama serve

# Keep model warm (1 hour)
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3.2", "keep_alive": 3600}'

# Check Ollama health
curl http://localhost:11434/api/tags
```

### GitHub Actions Workflows
```bash
# Manually trigger code generation
# Go to: https://github.com/D0k-Soundwave/intern/actions
# Select "Implementation Code Generation"
# Choose component and language

# Available components:
# - mcp-server
# - category-scanner
# - interview-manager
# - routing-engine
# - response-transformer
# - ollama-lifecycle
```

## Key Implementation Rules

### Routing Decision Flow
1. ALL requests undergo interview process (not predetermined)
2. Complexity > 0.8 → Route to Claude Code
3. Equal capabilities → Ollama wins (3% preference rule)
4. Bias detection runs every 100 requests
5. Response transformation makes Ollama indistinguishable from Claude

### Database Schema
- `interview_performance`: Tracks model performance per category
- `routing_history`: For bias detection (need 100+ records)
- `categories`: All 76 categories with routing preferences

### Environment Variables
```bash
MCP_PROTOCOL_VERSION=2024-11-05
MCP_TRANSPORT=stdio
INTERVIEW_ENABLED=true
BIAS_DETECTION_ENABLED=true
LOCAL_PREFERENCE_PERCENTAGE=3
OLLAMA_HOST=http://localhost:11434
OLLAMA_KEEP_ALIVE=3600
OLLAMA_AUTO_START=true
# NO CLAUDE API KEYS NEEDED
```

## File Structure Conventions

### Documentation
- All docs in `/mnt/r/intern/Docs/`
- Main plan: `Docs/full_plan.md`
- Requirements: `Docs/1. requirements.md` (MUST READ)

### Source Code (when generated)
- `src/mcp/` - MCP server implementation
- `src/routing/` - Category scanner and router
- `src/interview/` - Interview process manager
- `src/response/` - Response transformation
- `src/adapters/` - Ollama lifecycle manager

### Tests
- `tests/unit/` - Unit tests for each component
- `tests/integration/` - End-to-end tests
- `tests/fixtures/` - Test data

## Common Pitfalls to Avoid

1. **Never skip the interview process** - Routing must be dynamic, not predetermined
2. **Always apply 3% rule** - Ollama gets 0.03 bonus in all decisions
3. **Transform all Ollama responses** - Must appear exactly like Claude format
4. **Keep Ollama alive** - Health check every 30s, keep-alive for 1 hour
5. **Monitor for bias** - Check routing_history table regularly

## Performance Requirements

- Response time < 1 second
- Error rate < 1%  
- Memory usage < 1GB
- Ollama health check every 30 seconds
- All functions must handle errors properly

## Critical Integration Points

1. **MCP Server ↔ Claude Code**: stdio/JSON-RPC only
2. **Category Scanner → Interview Manager**: All 76 categories must be scanned
3. **Interview Manager → Decision Engine**: Interview results determine routing
4. **Ollama Service → Response Transformer**: All responses must be transformed
5. **Decision Engine → Performance DB**: All decisions logged for bias detection