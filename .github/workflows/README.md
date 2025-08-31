# GitHub Actions Workflows

This directory contains automated workflows for the Claude-Ollama MCP Integration System.

## Workflows

### 1. MCP Integration Analysis (`mcp-analysis.yml`)
- **Trigger**: On push/PR to MCP-related files
- **Purpose**: Analyzes MCP protocol compliance and integration correctness
- **Outputs**: Compliance report

### 2. Documentation Review (`doc-review.yml`)
- **Trigger**: On any documentation changes
- **Purpose**: Reviews documentation for accuracy, completeness, and compliance
- **Checks**: 
  - Requirements.md compliance
  - Cross-references validation
  - Technical accuracy

### 3. Routing Framework Validation (`routing-validation.yml`)
- **Trigger**: On routing/interview system changes, weekly schedule
- **Purpose**: Validates the complete routing framework
- **Tests**:
  - All 76 categories coverage
  - Interview logic validation
  - Response transformation checks
  - 3% local preference rule

### 4. Implementation Code Generation (`code-generation.yml`)
- **Trigger**: Manual dispatch or documentation updates
- **Purpose**: Generates implementation code from documentation
- **Components**:
  - MCP Server
  - Category Scanner
  - Interview Manager
  - Response Transformer
  - Ollama Lifecycle Manager
- **Languages**: Python, TypeScript, Go

### 5. Main CI/CD Pipeline (`main-ci.yml`)
- **Trigger**: On push/PR to main branches
- **Purpose**: Complete system validation
- **Checks**:
  - Requirements.md compliance
  - Complete integration verification
  - Routing logic tests
  - Security scanning

## Usage

### Manual Code Generation

To generate implementation code:

1. Go to Actions tab in GitHub
2. Select "Implementation Code Generation"
3. Click "Run workflow"
4. Choose component and language
5. Review the generated PR

### Viewing Reports

All validation reports are uploaded as artifacts and can be downloaded from the workflow run page.

## Requirements

These workflows use the Claude GitHub App (`anthropics/claude-code-action@main`) which must be installed on the repository.

## Compliance

All workflows enforce the requirements from `Docs/1. requirements.md`:
- No placeholder code
- Complete implementations only
- File size limits (< 2000 lines)
- Proper error handling