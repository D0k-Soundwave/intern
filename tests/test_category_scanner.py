"""
Category Scanner Tests

Comprehensive tests for the 76-category classification system
including pattern matching, keyword detection, and confidence scoring.

Per requirements.md: Complete implementation, no placeholders.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.routing.category_scanner import CategoryScanner, CategoryMatch, ScanResult
from src.routing.categories import get_category_by_id, get_all_categories

class TestCategoryScanner:
    """Test cases for CategoryScanner"""
    
    @pytest.fixture
    async def scanner(self):
        """Create CategoryScanner instance"""
        return CategoryScanner()
    
    @pytest.mark.asyncio
    async def test_initialization(self, scanner):
        """Test scanner initialization"""
        assert scanner is not None
        assert len(scanner.categories) == 76
        assert scanner.compiled_patterns is not None
        assert 'code' in scanner.compiled_patterns
        assert 'data' in scanner.compiled_patterns
        assert 'system' in scanner.compiled_patterns
        assert 'security' in scanner.compiled_patterns
        assert 'business' in scanner.compiled_patterns
    
    @pytest.mark.asyncio
    async def test_scan_simple_code_request(self, scanner, simple_input_text):
        """Test scanning simple code generation request"""
        result = await scanner.scan_claude_input(simple_input_text)
        
        assert isinstance(result, ScanResult)
        assert result.category['name'] in ['code_generation_simple', 'code_generation_complex']
        assert result.confidence > 0.5
        assert len(result.all_matches) > 0
        assert 'create' in result.intents or 'question' in result.intents
        
    @pytest.mark.asyncio
    async def test_scan_complex_architecture_request(self, scanner, complex_input_text):
        """Test scanning complex system architecture request"""
        result = await scanner.scan_claude_input(complex_input_text)
        
        assert isinstance(result, ScanResult)
        # Should match system architecture or microservices
        assert result.category['complexity'] in ['medium', 'high']
        assert result.confidence > 0.6
        assert len(result.all_matches) >= 3  # Should match multiple categories
        
    @pytest.mark.asyncio
    async def test_scan_question_input(self, scanner, question_input_text):
        """Test scanning question-based input"""
        result = await scanner.scan_claude_input(question_input_text)
        
        assert isinstance(result, ScanResult)
        assert 'question' in result.intents
        assert result.category['name'] in ['question_answering', 'explanation', 'tutoring']
        
    @pytest.mark.asyncio
    async def test_category_confidence_scoring(self, scanner):
        """Test category confidence scoring accuracy"""
        test_cases = [
            ("Write a Python function", "code_generation_simple", 0.7),
            ("Design microservices architecture", "system_architecture", 0.6),
            ("Analyze sales data with SQL", "data_analysis_simple", 0.6),
            ("Security vulnerability assessment", "security_analysis", 0.7),
            ("What is machine learning?", "question_answering", 0.5)
        ]
        
        for input_text, expected_category, min_confidence in test_cases:
            result = await scanner.scan_claude_input(input_text)
            
            # Check if expected category is in top matches
            category_names = [match.category['name'] for match in result.all_matches[:3]]
            assert any(expected_category in name for name in category_names), \
                f"Expected {expected_category} in top matches for '{input_text}'"
            
            # Check confidence threshold
            assert result.confidence >= min_confidence, \
                f"Confidence {result.confidence} below threshold {min_confidence}"
    
    @pytest.mark.asyncio
    async def test_keyword_matching(self, scanner):
        """Test keyword matching accuracy"""
        # Test specific keyword matches
        keyword_tests = [
            ("docker kubernetes deployment", "deployment_config"),
            ("machine learning neural network", "machine_learning"),
            ("sql database query", "database_design"),
            ("security audit penetration", "security_analysis"),
            ("financial analysis budget", "financial_analysis")
        ]
        
        for input_text, expected_category in keyword_tests:
            result = await scanner.scan_claude_input(input_text)
            
            # Check that expected category appears in matches
            found_categories = [match.category['name'] for match in result.all_matches]
            assert any(expected_category in cat for cat in found_categories), \
                f"Expected {expected_category} for keywords '{input_text}'"
    
    @pytest.mark.asyncio
    async def test_pattern_recognition(self, scanner):
        """Test pattern recognition for different content types"""
        pattern_tests = [
            ("def fibonacci(n): return n if n <= 1 else fib(n-1) + fib(n-2)", "code"),
            ("SELECT * FROM users WHERE age > 25", "code"),
            ("import pandas as pd; df.describe()", "data"),
            ("docker run -p 8080:80 nginx", "system"),
            ("encrypt data with AES-256", "security")
        ]
        
        for input_text, expected_pattern_type in pattern_tests:
            result = await scanner.scan_claude_input(input_text)
            
            # Verify pattern was detected by checking category alignment
            if expected_pattern_type == "code":
                assert any("code_generation" in match.category['name'] for match in result.all_matches)
            elif expected_pattern_type == "data":
                assert any("data" in match.category['name'] for match in result.all_matches)
            elif expected_pattern_type == "system":
                assert any(cat in match.category['name'] for match in result.all_matches 
                          for cat in ["deployment", "containerization", "system"])
    
    @pytest.mark.asyncio
    async def test_intent_extraction(self, scanner):
        """Test intent extraction from input text"""
        intent_tests = [
            ("Create a new function", ["create"]),
            ("Fix this bug in my code", ["fix"]),
            ("Analyze the performance data", ["analyze"]),
            ("How do I implement authentication?", ["question"]),
            ("Optimize this SQL query", ["optimize"]),
            ("Design a secure API", ["design"])
        ]
        
        for input_text, expected_intents in intent_tests:
            result = await scanner.scan_claude_input(input_text)
            
            # Check that at least one expected intent was found
            assert any(intent in result.intents for intent in expected_intents), \
                f"Expected intents {expected_intents} not found in {result.intents} for '{input_text}'"
    
    @pytest.mark.asyncio
    async def test_complexity_mapping(self, scanner):
        """Test complexity level mapping"""
        complexity_tests = [
            ("Hello world program", "low"),
            ("REST API with authentication", "medium"),
            ("Distributed blockchain system", "high"),
            ("Simple data visualization", "low"),
            ("Real-time ML inference pipeline", "high")
        ]
        
        for input_text, expected_complexity in complexity_tests:
            result = await scanner.scan_claude_input(input_text)
            
            # Primary category should have appropriate complexity
            assert result.category['complexity'] == expected_complexity, \
                f"Expected complexity {expected_complexity} for '{input_text}', got {result.category['complexity']}"
    
    @pytest.mark.asyncio
    async def test_multiple_category_matching(self, scanner):
        """Test that complex requests match multiple relevant categories"""
        complex_request = """
        Build a web application with user authentication, real-time chat,
        data analytics dashboard, and deploy it using Docker and Kubernetes
        """
        
        result = await scanner.scan_claude_input(complex_request)
        
        # Should match multiple categories
        assert len(result.all_matches) >= 4
        
        # Should include various category types
        category_names = [match.category['name'] for match in result.all_matches]
        
        # Check for expected category types
        has_web_dev = any("frontend" in name or "backend" in name for name in category_names)
        has_auth = any("access_control" in name or "security" in name for name in category_names)
        has_data = any("data" in name for name in category_names)
        has_deploy = any("deployment" in name or "containerization" in name for name in category_names)
        
        assert has_web_dev or has_auth or has_data or has_deploy, \
            "Complex request should match multiple relevant category types"
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, scanner):
        """Test edge cases and error handling"""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "x" * 1000,  # Very long string
            "🚀🎯💻🔥⚡",  # Emoji only
            "Hello world! What's up? How are you today? Great weather, isn't it?",  # No technical content
        ]
        
        for edge_case in edge_cases:
            result = await scanner.scan_claude_input(edge_case)
            
            # Should not crash and should return valid result
            assert isinstance(result, ScanResult)
            assert result.category is not None
            assert 0 <= result.confidence <= 1
            
            # Empty or very short inputs should default to unknown_general
            if len(edge_case.strip()) <= 2:
                assert result.category['name'] == 'unknown_general'
    
    @pytest.mark.asyncio
    async def test_category_suggestions(self, scanner):
        """Test category auto-completion suggestions"""
        suggestions = await scanner.get_category_suggestions("cod")
        
        assert len(suggestions) > 0
        assert all("cod" in suggestion['matched_keyword'].lower() for suggestion in suggestions)
        
        # Test with longer partial input
        suggestions = await scanner.get_category_suggestions("machine")
        code_suggestions = [s for s in suggestions if "machine" in s['matched_keyword'].lower()]
        assert len(code_suggestions) > 0
        
        # Test with very short input
        suggestions = await scanner.get_category_suggestions("a")
        assert len(suggestions) <= 10  # Should be limited
        
        # Test with non-matching input
        suggestions = await scanner.get_category_suggestions("xyzabc")
        assert len(suggestions) == 0
    
    @pytest.mark.asyncio
    async def test_category_stats(self, scanner):
        """Test category statistics generation"""
        stats = scanner.get_category_stats()
        
        assert 'total_categories' in stats
        assert stats['total_categories'] == 76
        
        assert 'complexity_distribution' in stats
        complexity_dist = stats['complexity_distribution']
        assert 'low' in complexity_dist
        assert 'medium' in complexity_dist
        assert 'high' in complexity_dist
        assert sum(complexity_dist.values()) == 76
        
        assert 'top_capabilities' in stats
        assert len(stats['top_capabilities']) <= 10
        
        assert 'keywords_per_category' in stats
        assert len(stats['keywords_per_category']) == 76
    
    @pytest.mark.asyncio
    async def test_normalize_input(self, scanner):
        """Test input normalization"""
        test_cases = [
            ("  HELLO   WORLD  ", "hello world"),
            ("Mixed-Case_Text.With!Symbols", "mixed case text with symbols"),
            ("Multiple\n\nNewlines", "multiple newlines"),
            ("Tabs\t\tAnd    Spaces", "tabs and spaces")
        ]
        
        for input_text, expected in test_cases:
            normalized = scanner._normalize_input(input_text)
            assert normalized == expected
    
    @pytest.mark.asyncio
    async def test_performance(self, scanner):
        """Test scanning performance"""
        import time
        
        test_inputs = [
            "Write a Python function",
            "Design microservices architecture with Docker and Kubernetes",
            "Analyze financial data using machine learning algorithms",
            "Create secure authentication system with OAuth2",
            "Build real-time chat application with WebSocket"
        ]
        
        # Measure scanning time
        start_time = time.time()
        
        for input_text in test_inputs:
            result = await scanner.scan_claude_input(input_text)
            assert result is not None
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_inputs)
        
        # Should scan each input in reasonable time (< 100ms average)
        assert avg_time < 0.1, f"Average scan time {avg_time:.3f}s exceeds threshold"
    
    @pytest.mark.asyncio
    async def test_confidence_calibration(self, scanner):
        """Test confidence score calibration"""
        # High confidence cases
        high_confidence_tests = [
            "Write a Python function to sort a list",
            "SELECT users FROM database WHERE active = true",
            "Deploy application using Docker containers"
        ]
        
        for input_text in high_confidence_tests:
            result = await scanner.scan_claude_input(input_text)
            assert result.confidence >= 0.6, \
                f"High-confidence input '{input_text}' got low confidence: {result.confidence}"
        
        # Low confidence cases
        low_confidence_tests = [
            "Something something",
            "Please help me",
            "I don't know what to do"
        ]
        
        for input_text in low_confidence_tests:
            result = await scanner.scan_claude_input(input_text)
            # These should get lower confidence or default to unknown_general
            if result.category['name'] != 'unknown_general':
                assert result.confidence <= 0.7, \
                    f"Low-confidence input '{input_text}' got high confidence: {result.confidence}"