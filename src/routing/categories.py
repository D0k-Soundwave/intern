"""
Complete 76-Category Classification System

Defines all categories for request classification as specified in full_plan.md.
Each category includes name, complexity, and typical capabilities.

Per requirements.md: Complete implementation, no placeholders.
"""
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class CategoryInfo:
    """Information about a specific category"""
    id: int
    name: str
    description: str
    complexity: str  # low, medium, high
    capabilities: List[str]
    keywords: List[str]
    typical_intents: List[str]

# Complete 76-category system as per full_plan.md
CATEGORIES: Dict[int, Dict[str, Any]] = {
    # Programming & Development (1-15)
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
    3: {
        "name": "code_refactoring",
        "description": "Code improvement and optimization",
        "complexity": "medium",
        "capabilities": ["pattern_recognition", "best_practices"],
        "keywords": ["refactor", "improve", "optimize", "clean", "restructure"],
        "typical_intents": ["improve_code", "optimize_performance", "clean_code"]
    },
    4: {
        "name": "bug_fixing",
        "description": "Debugging and error resolution",
        "complexity": "high",
        "capabilities": ["error_analysis", "debugging"],
        "keywords": ["bug", "error", "fix", "debug", "issue", "broken"],
        "typical_intents": ["fix_bug", "resolve_error", "debug_issue"]
    },
    5: {
        "name": "code_review",
        "description": "Code quality assessment",
        "complexity": "medium",
        "capabilities": ["best_practices", "security_analysis"],
        "keywords": ["review", "analyze", "quality", "assessment", "feedback"],
        "typical_intents": ["review_code", "assess_quality", "provide_feedback"]
    },
    6: {
        "name": "testing",
        "description": "Test creation and validation",
        "complexity": "medium",
        "capabilities": ["test_design", "validation"],
        "keywords": ["test", "unittest", "validation", "verify", "check"],
        "typical_intents": ["write_tests", "validate_code", "test_coverage"]
    },
    7: {
        "name": "documentation",
        "description": "Code documentation and comments",
        "complexity": "low",
        "capabilities": ["technical_writing", "code_analysis"],
        "keywords": ["document", "comment", "readme", "docstring", "explain"],
        "typical_intents": ["document_code", "write_readme", "add_comments"]
    },
    8: {
        "name": "api_design",
        "description": "REST/GraphQL API design",
        "complexity": "medium",
        "capabilities": ["api_patterns", "openapi"],
        "keywords": ["api", "rest", "graphql", "endpoint", "service"],
        "typical_intents": ["design_api", "create_endpoint", "api_specification"]
    },
    9: {
        "name": "database_design",
        "description": "Database schema and queries",
        "complexity": "high",
        "capabilities": ["database_modeling", "sql_optimization"],
        "keywords": ["database", "schema", "table", "sql", "query"],
        "typical_intents": ["design_schema", "optimize_query", "database_modeling"]
    },
    10: {
        "name": "frontend_development",
        "description": "User interface development",
        "complexity": "medium",
        "capabilities": ["ui_design", "javascript"],
        "keywords": ["frontend", "ui", "interface", "react", "vue", "angular"],
        "typical_intents": ["build_ui", "create_interface", "frontend_component"]
    },
    11: {
        "name": "backend_development",
        "description": "Server-side development",
        "complexity": "medium",
        "capabilities": ["server_logic", "api_development"],
        "keywords": ["backend", "server", "api", "service", "microservice"],
        "typical_intents": ["build_server", "create_service", "backend_logic"]
    },
    12: {
        "name": "devops",
        "description": "DevOps and deployment",
        "complexity": "medium",
        "capabilities": ["deployment", "ci_cd"],
        "keywords": ["devops", "deploy", "ci", "cd", "pipeline", "docker"],
        "typical_intents": ["setup_deployment", "create_pipeline", "devops_automation"]
    },
    13: {
        "name": "mobile_development",
        "description": "Mobile app development",
        "complexity": "medium",
        "capabilities": ["mobile_frameworks", "app_design"],
        "keywords": ["mobile", "app", "ios", "android", "react native", "flutter"],
        "typical_intents": ["build_app", "mobile_interface", "app_development"]
    },
    14: {
        "name": "embedded_systems",
        "description": "Embedded and IoT development",
        "complexity": "high",
        "capabilities": ["hardware_integration", "embedded_programming"],
        "keywords": ["embedded", "iot", "hardware", "sensor", "microcontroller"],
        "typical_intents": ["embedded_programming", "iot_development", "hardware_control"]
    },
    15: {
        "name": "game_development",
        "description": "Game programming and design",
        "complexity": "high",
        "capabilities": ["game_engines", "graphics_programming"],
        "keywords": ["game", "unity", "unreal", "graphics", "animation"],
        "typical_intents": ["create_game", "game_mechanics", "graphics_programming"]
    },
    
    # Data & Analytics (16-25)
    16: {
        "name": "data_analysis_simple",
        "description": "Basic data queries and summaries",
        "complexity": "low",
        "capabilities": ["sql", "basic_stats"],
        "keywords": ["analyze data", "summary", "statistics", "simple query"],
        "typical_intents": ["analyze_data", "data_summary", "basic_statistics"]
    },
    17: {
        "name": "data_visualization",
        "description": "Creating charts and graphs",
        "complexity": "medium",
        "capabilities": ["plotting", "data_formatting"],
        "keywords": ["chart", "graph", "plot", "visualization", "dashboard"],
        "typical_intents": ["create_chart", "visualize_data", "build_dashboard"]
    },
    18: {
        "name": "machine_learning",
        "description": "ML model development and training",
        "complexity": "high",
        "capabilities": ["ml_frameworks", "model_optimization"],
        "keywords": ["machine learning", "model", "train", "predict", "ml"],
        "typical_intents": ["build_model", "train_ml", "predict_outcomes"]
    },
    19: {
        "name": "deep_learning",
        "description": "Neural networks and deep learning",
        "complexity": "high",
        "capabilities": ["neural_networks", "deep_learning_frameworks"],
        "keywords": ["deep learning", "neural", "network", "tensorflow", "pytorch"],
        "typical_intents": ["neural_network", "deep_learning_model", "ai_training"]
    },
    20: {
        "name": "statistical_analysis",
        "description": "Advanced statistical methods",
        "complexity": "medium",
        "capabilities": ["statistics", "hypothesis_testing"],
        "keywords": ["statistics", "hypothesis", "correlation", "regression"],
        "typical_intents": ["statistical_test", "correlation_analysis", "regression_model"]
    },
    21: {
        "name": "data_cleaning",
        "description": "Data preprocessing and cleaning",
        "complexity": "low",
        "capabilities": ["data_preprocessing", "data_validation"],
        "keywords": ["clean data", "preprocess", "normalize", "transform"],
        "typical_intents": ["clean_dataset", "preprocess_data", "data_transformation"]
    },
    22: {
        "name": "etl_pipelines",
        "description": "Extract, Transform, Load processes",
        "complexity": "medium",
        "capabilities": ["data_pipeline", "data_integration"],
        "keywords": ["etl", "pipeline", "extract", "transform", "load"],
        "typical_intents": ["build_pipeline", "etl_process", "data_integration"]
    },
    23: {
        "name": "big_data",
        "description": "Big data processing",
        "complexity": "high",
        "capabilities": ["big_data_tools", "distributed_computing"],
        "keywords": ["big data", "spark", "hadoop", "distributed", "large dataset"],
        "typical_intents": ["process_big_data", "distributed_analysis", "large_scale_processing"]
    },
    24: {
        "name": "time_series",
        "description": "Time series analysis",
        "complexity": "medium",
        "capabilities": ["time_series_analysis", "forecasting"],
        "keywords": ["time series", "forecast", "trend", "seasonal", "temporal"],
        "typical_intents": ["time_series_analysis", "forecasting", "trend_analysis"]
    },
    25: {
        "name": "predictive_modeling",
        "description": "Predictive analytics and modeling",
        "complexity": "high",
        "capabilities": ["predictive_analytics", "model_validation"],
        "keywords": ["predict", "forecast", "model", "analytics", "predictive"],
        "typical_intents": ["predictive_model", "forecast_analysis", "predictive_analytics"]
    },
    
    # Content & Language (26-35)
    26: {
        "name": "technical_writing",
        "description": "Documentation, tutorials, guides",
        "complexity": "medium",
        "capabilities": ["technical_accuracy", "formatting"],
        "keywords": ["documentation", "tutorial", "guide", "technical writing"],
        "typical_intents": ["write_documentation", "create_tutorial", "technical_guide"]
    },
    27: {
        "name": "creative_writing",
        "description": "Stories, narratives, creative content",
        "complexity": "low",
        "capabilities": ["creativity", "narrative"],
        "keywords": ["story", "creative", "narrative", "fiction", "writing"],
        "typical_intents": ["write_story", "creative_content", "narrative_creation"]
    },
    28: {
        "name": "translation",
        "description": "Language translation tasks",
        "complexity": "low",
        "capabilities": ["multilingual"],
        "keywords": ["translate", "translation", "language", "convert"],
        "typical_intents": ["translate_text", "language_conversion", "multilingual"]
    },
    29: {
        "name": "summarization",
        "description": "Text summarization",
        "complexity": "low",
        "capabilities": ["extraction", "condensation"],
        "keywords": ["summarize", "summary", "brief", "condense", "abstract"],
        "typical_intents": ["summarize_text", "create_abstract", "condense_content"]
    },
    30: {
        "name": "proofreading",
        "description": "Grammar and style checking",
        "complexity": "low",
        "capabilities": ["grammar_checking", "style_analysis"],
        "keywords": ["proofread", "grammar", "spelling", "edit", "correct"],
        "typical_intents": ["check_grammar", "proofread_text", "edit_content"]
    },
    31: {
        "name": "content_generation",
        "description": "General content creation",
        "complexity": "medium",
        "capabilities": ["content_creation", "topic_research"],
        "keywords": ["content", "generate", "create", "write", "article"],
        "typical_intents": ["generate_content", "create_article", "content_creation"]
    },
    32: {
        "name": "email_drafting",
        "description": "Email and communication drafting",
        "complexity": "low",
        "capabilities": ["communication", "formatting"],
        "keywords": ["email", "message", "communication", "draft", "letter"],
        "typical_intents": ["draft_email", "write_message", "communication"]
    },
    33: {
        "name": "report_writing",
        "description": "Business and analytical reports",
        "complexity": "medium",
        "capabilities": ["business_writing", "data_presentation"],
        "keywords": ["report", "analysis", "business", "findings", "presentation"],
        "typical_intents": ["write_report", "business_analysis", "present_findings"]
    },
    34: {
        "name": "academic_writing",
        "description": "Academic papers and research",
        "complexity": "high",
        "capabilities": ["academic_standards", "research_methodology"],
        "keywords": ["academic", "research", "paper", "thesis", "scholarly"],
        "typical_intents": ["academic_paper", "research_writing", "scholarly_content"]
    },
    35: {
        "name": "copywriting",
        "description": "Marketing and advertising copy",
        "complexity": "medium",
        "capabilities": ["marketing", "persuasive_writing"],
        "keywords": ["copy", "marketing", "advertising", "promotional", "sales"],
        "typical_intents": ["marketing_copy", "advertising_content", "promotional_writing"]
    },
    
    # System & Architecture (36-45)
    36: {
        "name": "system_architecture",
        "description": "System design and architecture",
        "complexity": "high",
        "capabilities": ["distributed_systems", "scalability"],
        "keywords": ["architecture", "system design", "scalable", "distributed"],
        "typical_intents": ["design_system", "architecture_planning", "scalable_design"]
    },
    37: {
        "name": "deployment_config",
        "description": "CI/CD, Docker, Kubernetes configs",
        "complexity": "medium",
        "capabilities": ["yaml", "container_orchestration"],
        "keywords": ["deployment", "docker", "kubernetes", "config", "yaml"],
        "typical_intents": ["deployment_setup", "container_config", "k8s_deployment"]
    },
    38: {
        "name": "network_design",
        "description": "Network architecture and protocols",
        "complexity": "high",
        "capabilities": ["networking", "protocols"],
        "keywords": ["network", "protocol", "tcp", "ip", "routing"],
        "typical_intents": ["network_design", "protocol_implementation", "network_config"]
    },
    39: {
        "name": "security_architecture",
        "description": "Security system design",
        "complexity": "high",
        "capabilities": ["security_design", "threat_modeling"],
        "keywords": ["security", "secure", "authentication", "authorization", "encryption"],
        "typical_intents": ["security_design", "secure_system", "threat_analysis"]
    },
    40: {
        "name": "cloud_architecture",
        "description": "Cloud system design",
        "complexity": "high",
        "capabilities": ["cloud_platforms", "serverless"],
        "keywords": ["cloud", "aws", "azure", "gcp", "serverless"],
        "typical_intents": ["cloud_design", "serverless_architecture", "cloud_migration"]
    },
    41: {
        "name": "microservices",
        "description": "Microservices architecture",
        "complexity": "high",
        "capabilities": ["service_design", "api_gateway"],
        "keywords": ["microservice", "service", "api gateway", "service mesh"],
        "typical_intents": ["microservice_design", "service_architecture", "api_design"]
    },
    42: {
        "name": "containerization",
        "description": "Container technology",
        "complexity": "medium",
        "capabilities": ["docker", "container_orchestration"],
        "keywords": ["container", "docker", "pod", "image", "registry"],
        "typical_intents": ["containerize_app", "docker_setup", "container_deployment"]
    },
    43: {
        "name": "ci_cd",
        "description": "Continuous integration/deployment",
        "complexity": "medium",
        "capabilities": ["automation", "pipeline_design"],
        "keywords": ["ci", "cd", "pipeline", "automation", "build"],
        "typical_intents": ["setup_pipeline", "automation", "continuous_deployment"]
    },
    44: {
        "name": "monitoring_setup",
        "description": "System monitoring and observability",
        "complexity": "medium",
        "capabilities": ["monitoring", "alerting"],
        "keywords": ["monitoring", "metrics", "logging", "alerting", "observability"],
        "typical_intents": ["setup_monitoring", "create_alerts", "observability"]
    },
    45: {
        "name": "disaster_recovery",
        "description": "Backup and disaster recovery",
        "complexity": "high",
        "capabilities": ["backup_strategies", "recovery_planning"],
        "keywords": ["backup", "recovery", "disaster", "failover", "resilience"],
        "typical_intents": ["disaster_recovery", "backup_strategy", "failover_planning"]
    },
    
    # Business & Strategy (46-55)
    46: {
        "name": "business_analysis",
        "description": "Business process analysis",
        "complexity": "medium",
        "capabilities": ["business_processes", "analysis"],
        "keywords": ["business", "analysis", "process", "strategy", "operations"],
        "typical_intents": ["business_analysis", "process_optimization", "strategic_planning"]
    },
    47: {
        "name": "market_research",
        "description": "Market analysis and research",
        "complexity": "medium",
        "capabilities": ["market_analysis", "competitive_intelligence"],
        "keywords": ["market", "research", "competitor", "industry", "trends"],
        "typical_intents": ["market_analysis", "competitor_research", "industry_trends"]
    },
    48: {
        "name": "financial_analysis",
        "description": "Financial modeling and analysis",
        "complexity": "high",
        "capabilities": ["financial_models", "risk_analysis"],
        "keywords": ["financial", "finance", "budget", "roi", "investment"],
        "typical_intents": ["financial_model", "budget_analysis", "investment_analysis"]
    },
    49: {
        "name": "project_planning",
        "description": "Project management and planning",
        "complexity": "medium",
        "capabilities": ["project_management", "resource_planning"],
        "keywords": ["project", "plan", "timeline", "milestone", "resource"],
        "typical_intents": ["project_plan", "timeline_creation", "resource_allocation"]
    },
    50: {
        "name": "risk_assessment",
        "description": "Risk analysis and mitigation",
        "complexity": "high",
        "capabilities": ["risk_analysis", "mitigation_strategies"],
        "keywords": ["risk", "assessment", "mitigation", "threat", "vulnerability"],
        "typical_intents": ["risk_analysis", "threat_assessment", "mitigation_planning"]
    },
    51: {
        "name": "competitive_analysis",
        "description": "Competitor and market analysis",
        "complexity": "medium",
        "capabilities": ["competitive_intelligence", "market_positioning"],
        "keywords": ["competitor", "competitive", "market position", "analysis"],
        "typical_intents": ["competitor_analysis", "market_positioning", "competitive_strategy"]
    },
    52: {
        "name": "product_strategy",
        "description": "Product development strategy",
        "complexity": "high",
        "capabilities": ["product_management", "strategic_planning"],
        "keywords": ["product", "strategy", "roadmap", "development", "feature"],
        "typical_intents": ["product_strategy", "roadmap_planning", "feature_prioritization"]
    },
    53: {
        "name": "pricing_strategy",
        "description": "Pricing models and strategies",
        "complexity": "medium",
        "capabilities": ["pricing_models", "market_analysis"],
        "keywords": ["pricing", "price", "cost", "revenue", "model"],
        "typical_intents": ["pricing_strategy", "price_optimization", "revenue_model"]
    },
    54: {
        "name": "marketing_strategy",
        "description": "Marketing planning and strategy",
        "complexity": "medium",
        "capabilities": ["marketing_planning", "campaign_design"],
        "keywords": ["marketing", "campaign", "promotion", "brand", "advertising"],
        "typical_intents": ["marketing_plan", "campaign_strategy", "brand_positioning"]
    },
    55: {
        "name": "sales_strategy",
        "description": "Sales process and strategy",
        "complexity": "medium",
        "capabilities": ["sales_processes", "customer_acquisition"],
        "keywords": ["sales", "selling", "customer", "acquisition", "conversion"],
        "typical_intents": ["sales_strategy", "customer_acquisition", "conversion_optimization"]
    },
    
    # Security & Compliance (56-65)
    56: {
        "name": "security_analysis",
        "description": "Security audits and vulnerability assessment",
        "complexity": "high",
        "capabilities": ["security_patterns", "vulnerability_detection"],
        "keywords": ["security", "vulnerability", "audit", "penetration", "assessment"],
        "typical_intents": ["security_audit", "vulnerability_assessment", "security_analysis"]
    },
    57: {
        "name": "encryption_implementation",
        "description": "Cryptography and secure coding",
        "complexity": "high",
        "capabilities": ["cryptography", "secure_protocols"],
        "keywords": ["encryption", "cryptography", "secure", "certificate", "key"],
        "typical_intents": ["implement_encryption", "secure_communication", "cryptographic_solution"]
    },
    58: {
        "name": "vulnerability_assessment",
        "description": "Security vulnerability testing",
        "complexity": "high",
        "capabilities": ["security_testing", "vulnerability_scanning"],
        "keywords": ["vulnerability", "scan", "security test", "penetration", "exploit"],
        "typical_intents": ["vulnerability_scan", "security_testing", "penetration_test"]
    },
    59: {
        "name": "penetration_testing",
        "description": "Ethical hacking and penetration testing",
        "complexity": "high",
        "capabilities": ["ethical_hacking", "security_testing"],
        "keywords": ["penetration", "pen test", "ethical hacking", "security test"],
        "typical_intents": ["penetration_test", "security_assessment", "ethical_hacking"]
    },
    60: {
        "name": "compliance_audit",
        "description": "Regulatory compliance checking",
        "complexity": "high",
        "capabilities": ["compliance_frameworks", "audit_processes"],
        "keywords": ["compliance", "audit", "regulation", "gdpr", "hipaa"],
        "typical_intents": ["compliance_check", "regulatory_audit", "compliance_assessment"]
    },
    61: {
        "name": "access_control",
        "description": "Authentication and authorization",
        "complexity": "medium",
        "capabilities": ["identity_management", "access_policies"],
        "keywords": ["access", "authentication", "authorization", "identity", "permission"],
        "typical_intents": ["access_control", "identity_management", "permission_system"]
    },
    62: {
        "name": "security_policies",
        "description": "Security policy development",
        "complexity": "medium",
        "capabilities": ["policy_development", "security_governance"],
        "keywords": ["policy", "security policy", "governance", "procedure", "standard"],
        "typical_intents": ["security_policy", "governance_framework", "security_procedures"]
    },
    63: {
        "name": "incident_response",
        "description": "Security incident handling",
        "complexity": "high",
        "capabilities": ["incident_management", "forensics"],
        "keywords": ["incident", "response", "security incident", "breach", "forensics"],
        "typical_intents": ["incident_response", "security_breach", "incident_management"]
    },
    64: {
        "name": "forensics",
        "description": "Digital forensics and investigation",
        "complexity": "high",
        "capabilities": ["digital_forensics", "evidence_analysis"],
        "keywords": ["forensics", "investigation", "evidence", "analysis", "digital"],
        "typical_intents": ["digital_forensics", "forensic_analysis", "evidence_investigation"]
    },
    65: {
        "name": "threat_modeling",
        "description": "Security threat analysis",
        "complexity": "high",
        "capabilities": ["threat_analysis", "risk_modeling"],
        "keywords": ["threat", "model", "attack", "risk", "security model"],
        "typical_intents": ["threat_model", "security_modeling", "attack_analysis"]
    },
    
    # Miscellaneous (66-76)
    66: {
        "name": "question_answering",
        "description": "Simple Q&A tasks",
        "complexity": "low",
        "capabilities": ["comprehension", "retrieval"],
        "keywords": ["question", "answer", "what", "how", "why", "explain"],
        "typical_intents": ["answer_question", "provide_information", "explain_concept"]
    },
    67: {
        "name": "explanation",
        "description": "Concept explanation and tutorials",
        "complexity": "medium",
        "capabilities": ["teaching", "simplification"],
        "keywords": ["explain", "how", "why", "concept", "tutorial", "teach"],
        "typical_intents": ["explain_concept", "provide_tutorial", "educational_content"]
    },
    68: {
        "name": "tutoring",
        "description": "Educational guidance and tutoring",
        "complexity": "medium",
        "capabilities": ["education", "personalized_learning"],
        "keywords": ["tutor", "learn", "teach", "education", "study", "help"],
        "typical_intents": ["provide_tutoring", "educational_assistance", "learning_support"]
    },
    69: {
        "name": "math_computation",
        "description": "Mathematical calculations and proofs",
        "complexity": "medium",
        "capabilities": ["symbolic_math", "theorem_proving"],
        "keywords": ["math", "calculate", "equation", "formula", "solve", "proof"],
        "typical_intents": ["solve_math", "mathematical_proof", "calculation"]
    },
    70: {
        "name": "research_analysis",
        "description": "Research paper analysis and synthesis",
        "complexity": "high",
        "capabilities": ["academic_writing", "citation"],
        "keywords": ["research", "paper", "analysis", "study", "academic", "literature"],
        "typical_intents": ["research_analysis", "literature_review", "academic_research"]
    },
    71: {
        "name": "legal_document",
        "description": "Legal document analysis",
        "complexity": "high",
        "capabilities": ["legal_knowledge", "compliance"],
        "keywords": ["legal", "law", "contract", "agreement", "regulation", "clause"],
        "typical_intents": ["legal_analysis", "contract_review", "legal_research"]
    },
    72: {
        "name": "medical_information",
        "description": "Medical and health-related queries",
        "complexity": "high",
        "capabilities": ["medical_knowledge", "safety"],
        "keywords": ["medical", "health", "symptom", "diagnosis", "treatment", "medicine"],
        "typical_intents": ["medical_information", "health_guidance", "medical_research"]
    },
    73: {
        "name": "scientific_analysis",
        "description": "Scientific research and analysis",
        "complexity": "high",
        "capabilities": ["scientific_method", "data_analysis"],
        "keywords": ["science", "scientific", "experiment", "hypothesis", "analysis"],
        "typical_intents": ["scientific_analysis", "research_methodology", "experimental_design"]
    },
    74: {
        "name": "philosophical_discussion",
        "description": "Philosophical topics and discussions",
        "complexity": "medium",
        "capabilities": ["philosophical_reasoning", "ethical_analysis"],
        "keywords": ["philosophy", "ethics", "moral", "philosophical", "meaning"],
        "typical_intents": ["philosophical_discussion", "ethical_analysis", "conceptual_exploration"]
    },
    75: {
        "name": "real_time_interaction",
        "description": "Chat, dialogue, interactive tasks",
        "complexity": "low",
        "capabilities": ["conversation", "context_management"],
        "keywords": ["chat", "talk", "conversation", "discuss", "interactive"],
        "typical_intents": ["conversational", "interactive_dialogue", "real_time_chat"]
    },
    76: {
        "name": "unknown_general",
        "description": "Uncategorized or general tasks",
        "complexity": "medium",
        "capabilities": ["general"],
        "keywords": ["general", "misc", "other", "unknown", "various"],
        "typical_intents": ["general_task", "miscellaneous", "uncategorized"]
    }
}

def get_category_by_id(category_id: int) -> Dict[str, Any]:
    """Get category information by ID"""
    return CATEGORIES.get(category_id, CATEGORIES[76])  # Default to unknown_general

def get_category_by_name(name: str) -> Dict[str, Any]:
    """Get category information by name"""
    for category in CATEGORIES.values():
        if category["name"] == name:
            return category
    return CATEGORIES[76]  # Default to unknown_general

def get_all_categories() -> Dict[int, Dict[str, Any]]:
    """Get all categories"""
    return CATEGORIES.copy()

def get_categories_by_complexity(complexity: str) -> List[Dict[str, Any]]:
    """Get all categories with specified complexity level"""
    return [cat for cat in CATEGORIES.values() if cat["complexity"] == complexity]