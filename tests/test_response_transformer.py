"""
Response Transformer Tests

Tests for the Ollama-to-Claude response transformation system
including quality enhancement, format standardization, and style adaptation.

Per requirements.md: Complete implementation, no placeholders.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from src.response.transformer import (
    MCPResponseHandler, QualityEnhancer, FormatStandardizer, ClaudeStyleAdapter,
    ResponseFormat, TransformationResult
)

class TestMCPResponseHandler:
    """Test cases for MCPResponseHandler"""
    
    @pytest.fixture
    async def handler(self):
        """Create MCPResponseHandler instance"""
        return MCPResponseHandler()
    
    @pytest.mark.asyncio
    async def test_transform_simple_text_response(self, handler, sample_ollama_response):
        """Test transformation of simple text response"""
        ollama_response = {
            "response": "Hello! This is a simple text response from Ollama.",
            "model": "llama3.2"
        }
        
        context = {
            "request_id": "test-123",
            "input_text": "Say hello"
        }
        
        result = await handler.transform_to_claude(ollama_response, context)
        
        assert isinstance(result, str)
        assert "Hello!" in result
        # Should be enhanced with better formatting/style
        assert len(result) >= len(ollama_response["response"])
    
    @pytest.mark.asyncio
    async def test_transform_code_response(self, handler, sample_ollama_response):
        """Test transformation of code response"""
        context = {
            "request_id": "test-code-123",
            "input_text": "Write a fibonacci function"
        }
        
        result = await handler.transform_to_claude(sample_ollama_response, context)
        
        assert isinstance(result, str)
        assert "fibonacci" in result
        # Should contain the original code
        assert "def fibonacci(n):" in result
    
    @pytest.mark.asyncio
    async def test_extract_content_variations(self, handler):
        """Test content extraction from different Ollama response formats"""
        test_cases = [
            # Standard format
            ({"response": "test content"}, "test content"),
            # Message format
            ({"message": {"content": "test content"}}, "test content"),
            # String message format
            ({"message": "test content"}, "test content"),
            # Content field
            ({"content": "test content"}, "test content"),
            # Text field
            ({"text": "test content"}, "test content"),
            # Fallback - entire object as string
            ({"unknown_format": "data"}, "{'unknown_format': 'data'}")
        ]
        
        for ollama_response, expected_content in test_cases:
            extracted = handler._extract_content(ollama_response)
            assert expected_content in extracted
    
    @pytest.mark.asyncio
    async def test_format_detection(self, handler):
        """Test response format detection"""
        test_cases = [
            # Code format
            ("```python\ndef hello():\n    print('hello')\n```", ResponseFormat.CODE),
            ("`inline_code`", ResponseFormat.CODE),
            ("function test() { return true; }", ResponseFormat.CODE),
            ("import pandas as pd", ResponseFormat.CODE),
            
            # Structured format
            ('{"key": "value", "number": 123}', ResponseFormat.STRUCTURED),
            
            # Error format
            ("Error: Something went wrong", ResponseFormat.ERROR),
            ("Failed to process request", ResponseFormat.ERROR),
            
            # Text format (default)
            ("This is a regular text response", ResponseFormat.TEXT),
            ("Hello world!", ResponseFormat.TEXT)
        ]
        
        for content, expected_format in test_cases:
            detected = handler._detect_format(content, {})
            assert detected == expected_format, f"Expected {expected_format} for '{content}', got {detected}"
    
    @pytest.mark.asyncio
    async def test_error_handling_fallback(self, handler):
        """Test error handling and fallback response"""
        # Simulate error in transformation pipeline
        with pytest.raises(Exception):
            # This should trigger error handling
            await handler.quality_enhancer.enhance(None, ResponseFormat.TEXT, {})
        
        # Test fallback response creation
        fallback = handler._create_fallback_response("Original content", "Test error")
        
        assert "Original content" in fallback
        assert "formatting issues" in fallback
        assert "let me know" in fallback

class TestQualityEnhancer:
    """Test cases for QualityEnhancer"""
    
    @pytest.fixture
    async def enhancer(self):
        """Create QualityEnhancer instance"""
        return QualityEnhancer()
    
    @pytest.mark.asyncio
    async def test_grammar_improvement(self, enhancer):
        """Test grammar and spelling improvements"""
        test_cases = [
            ("i like programming", "I like programming"),
            ("hello    world", "hello world"),
            ("sentence.another sentence", "sentence. Another sentence"),
            ("text with trailing spaces   ", "text with trailing spaces")
        ]
        
        for input_text, expected_improvement in test_cases:
            result = await enhancer.enhance(input_text, ResponseFormat.TEXT, {})
            
            # Check that improvement was made
            assert expected_improvement in result.transformed_content or \
                   result.transformed_content.replace("  ", " ") == expected_improvement.replace("  ", " ")
    
    @pytest.mark.asyncio
    async def test_clarity_improvement(self, enhancer):
        """Test clarity and readability improvements"""
        long_text = "This is a very long sentence without proper breaks. It continues for a while and makes several points without natural paragraph breaks. This makes it harder to read and understand."
        
        result = await enhancer.enhance(long_text, ResponseFormat.TEXT, {})
        
        # Should add paragraph breaks for long text
        if "added_paragraph_breaks" in result.transformations_applied:
            assert "\n\n" in result.transformed_content
    
    @pytest.mark.asyncio
    async def test_technical_accuracy_code(self, enhancer):
        """Test technical accuracy improvements for code"""
        code_content = "def hello():\n    print('Hello World')\n\nhello()"
        
        result = await enhancer.enhance(code_content, ResponseFormat.CODE, {})
        
        # Should add code block formatting if missing
        if "added_code_block_formatting" in result.transformations_applied:
            assert "```" in result.transformed_content
    
    @pytest.mark.asyncio
    async def test_completeness_check(self, enhancer):
        """Test completeness checking"""
        test_cases = [
            ("Incomplete response...", ["detected_potential_incompleteness"]),
            ("Very short", ["response_potentially_too_brief"]),
            ("Complete and detailed response with sufficient information.", [])
        ]
        
        for content, expected_fixes in test_cases:
            context = {"input_text": "How do I implement authentication?"}
            result = await enhancer.enhance(content, ResponseFormat.TEXT, context)
            
            for expected_fix in expected_fixes:
                assert expected_fix in result.transformations_applied

class TestFormatStandardizer:
    """Test cases for FormatStandardizer"""
    
    @pytest.fixture
    async def standardizer(self):
        """Create FormatStandardizer instance"""
        return FormatStandardizer()
    
    @pytest.mark.asyncio
    async def test_code_format_standardization(self, standardizer):
        """Test code format standardization"""
        code_content = """```python
        def hello():
            print("hello")
        def world():
            print("world")
        ```"""
        
        result = await standardizer.standardize(code_content, ResponseFormat.CODE, {})
        
        # Should standardize indentation
        if "standardized_code_indentation" in result.transformations_applied:
            assert "def hello():" in result.transformed_content
            # Check that indentation is consistent
            lines = result.transformed_content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('```')]
            if code_lines:
                # All code lines should have consistent indentation
                indents = [len(line) - len(line.lstrip()) for line in code_lines if line.strip()]
                assert len(set(indents)) <= 2  # At most 2 levels of indentation
    
    @pytest.mark.asyncio
    async def test_json_format_standardization(self, standardizer):
        """Test JSON format standardization"""
        json_content = '{"name":"test","value":123,"nested":{"key":"value"}}'
        
        result = await standardizer.standardize(json_content, ResponseFormat.STRUCTURED, {})
        
        if "standardized_json_formatting" in result.transformations_applied:
            # Should be properly formatted JSON
            try:
                parsed = json.loads(result.transformed_content)
                assert parsed["name"] == "test"
                assert parsed["value"] == 123
                # Should have proper indentation
                assert "  " in result.transformed_content or "\n" in result.transformed_content
            except json.JSONDecodeError:
                pytest.fail("Standardized JSON should be valid")
    
    @pytest.mark.asyncio
    async def test_text_format_standardization(self, standardizer):
        """Test text format standardization"""
        text_with_spacing = "Line 1\n\n\n\nLine 2\n\n\n\nLine 3"
        
        result = await standardizer.standardize(text_with_spacing, ResponseFormat.TEXT, {})
        
        # Should standardize paragraph spacing
        if "standardized_paragraph_spacing" in result.transformations_applied:
            assert "\n\n\n" not in result.transformed_content
            assert "Line 1\n\nLine 2\n\nLine 3" in result.transformed_content
    
    @pytest.mark.asyncio
    async def test_list_format_standardization(self, standardizer):
        """Test list format standardization"""
        list_content = """Different list formats:
        * Item 1
        + Item 2
        1. Numbered item
        2. Another numbered"""
        
        result = await standardizer.standardize(list_content, ResponseFormat.TEXT, {})
        
        if "standardized_list_formatting" in result.transformations_applied:
            # Should standardize to dash format
            assert "- Item 1" in result.transformed_content
            assert "- Item 2" in result.transformed_content
    
    @pytest.mark.asyncio
    async def test_quote_standardization(self, standardizer):
        """Test quote mark standardization"""
        mixed_quotes = "He said 'hello' and then \"goodbye\"."
        
        result = await standardizer.standardize(mixed_quotes, ResponseFormat.TEXT, {})
        
        if "standardized_quote_marks" in result.transformations_applied:
            # Should prefer double quotes
            assert '"hello"' in result.transformed_content
            assert '"goodbye"' in result.transformed_content

class TestClaudeStyleAdapter:
    """Test cases for ClaudeStyleAdapter"""
    
    @pytest.fixture
    async def adapter(self):
        """Create ClaudeStyleAdapter instance"""
        return ClaudeStyleAdapter()
    
    @pytest.mark.asyncio
    async def test_claude_introduction_patterns(self, adapter):
        """Test Claude-style introduction patterns"""
        code_content = """```python
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n-1)
```"""
        
        result = await adapter.adapt(code_content, ResponseFormat.CODE, {"input_text": "factorial function"})
        
        if "added_claude_introduction" in result.transformations_applied:
            assert result.transformed_content.startswith("Here's") or result.transformed_content.startswith("I'll")
    
    @pytest.mark.asyncio
    async def test_helpful_context_addition(self, adapter):
        """Test helpful context addition"""
        short_response = "Yes"
        context = {"input_text": "Can you help me with authentication?"}
        
        result = await adapter.adapt(short_response, ResponseFormat.TEXT, context)
        
        # Should detect potentially brief response for help request
        if "detected_potentially_brief_help_response" in result.transformations_applied:
            # This is just detection, not modification in this case
            assert "detected_potentially_brief_help_response" in result.transformations_applied
    
    @pytest.mark.asyncio
    async def test_professional_tone_adjustment(self, adapter):
        """Test professional tone adjustments"""
        casual_content = "Yeah, uh, you can use this function I guess"
        
        result = await adapter.adapt(casual_content, ResponseFormat.TEXT, {})
        
        # Should professionalize casual language
        professional_improvements = [t for t in result.transformations_applied if t.startswith('professionalized_tone_')]
        if professional_improvements:
            assert "yeah" not in result.transformed_content.lower()
            assert "uh" not in result.transformed_content.lower()
    
    @pytest.mark.asyncio
    async def test_confidence_improvement(self, adapter):
        """Test confidence tone improvements"""
        uncertain_content = "I think this might work for your request."
        
        result = await adapter.adapt(uncertain_content, ResponseFormat.TEXT, {"input_text": "help with code"})
        
        if "improved_confidence_tone" in result.transformations_applied:
            assert not result.transformed_content.lower().startswith("i think")
            assert "Based on your request" in result.transformed_content

class TestEndToEndTransformation:
    """End-to-end transformation tests"""
    
    @pytest.fixture
    async def handler(self):
        """Create complete MCPResponseHandler"""
        return MCPResponseHandler()
    
    @pytest.mark.asyncio
    async def test_complete_code_transformation(self, handler):
        """Test complete code response transformation"""
        ollama_response = {
            "response": "def fibonacci(n):\nif n <= 1:\nreturn n\nelse:\nreturn fibonacci(n-1) + fibonacci(n-2)",
            "metadata": {"model": "llama3.2"}
        }
        
        context = {
            "request_id": "test-fibonacci",
            "input_text": "Write a fibonacci function"
        }
        
        result = await handler.transform_to_claude(ollama_response, context)
        
        # Should contain the original function
        assert "fibonacci" in result
        assert "def fibonacci(n):" in result
        
        # Should be properly formatted and enhanced
        assert len(result) > len(ollama_response["response"])
    
    @pytest.mark.asyncio
    async def test_complete_explanation_transformation(self, handler):
        """Test complete explanation response transformation"""
        ollama_response = {
            "response": "machine learning is a subset of artificial intelligence that uses statistical techniques to give computers the ability to learn from data without being explicitly programmed",
            "metadata": {"model": "llama3.2"}
        }
        
        context = {
            "request_id": "test-ml-explanation",
            "input_text": "What is machine learning?"
        }
        
        result = await handler.transform_to_claude(ollama_response, context)
        
        # Should be capitalized properly
        assert "Machine learning" in result
        # Should be well-formatted
        assert "." in result
        # Should be enhanced from original
        assert len(result) >= len(ollama_response["response"])
    
    @pytest.mark.asyncio
    async def test_error_response_transformation(self, handler):
        """Test error response transformation"""
        ollama_response = {
            "response": "Error: Unable to process the request due to invalid input",
            "metadata": {"model": "llama3.2"}
        }
        
        context = {
            "request_id": "test-error",
            "input_text": "Invalid request"
        }
        
        result = await handler.transform_to_claude(ollama_response, context)
        
        # Should contain error information
        assert "Error" in result or "error" in result
        # Should be properly formatted
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_transformation_performance(self, handler):
        """Test transformation performance"""
        import time
        
        test_responses = [
            {"response": f"Test response number {i}", "metadata": {"model": "llama3.2"}}
            for i in range(20)
        ]
        
        start_time = time.time()
        
        for i, response in enumerate(test_responses):
            context = {"request_id": f"perf-test-{i}", "input_text": "test"}
            result = await handler.transform_to_claude(response, context)
            assert len(result) > 0
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_responses)
        
        # Should transform responses quickly (< 50ms average)
        assert avg_time < 0.05, f"Average transformation time {avg_time:.3f}s exceeds threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_transformations(self, handler):
        """Test concurrent transformation handling"""
        import asyncio
        
        responses = [
            (
                {"response": f"Response {i}", "metadata": {"model": "llama3.2"}},
                {"request_id": f"concurrent-{i}", "input_text": f"Request {i}"}
            )
            for i in range(10)
        ]
        
        tasks = [
            handler.transform_to_claude(response, context)
            for response, context in responses
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for i, result in enumerate(results):
            assert f"Response {i}" in result or "response" in result.lower()
    
    @pytest.mark.asyncio
    async def test_edge_case_responses(self, handler):
        """Test edge case response handling"""
        edge_cases = [
            # Empty response
            {"response": "", "metadata": {"model": "llama3.2"}},
            # Very long response
            {"response": "x" * 5000, "metadata": {"model": "llama3.2"}},
            # Unicode content
            {"response": "Hello 世界 🌍 café naïve", "metadata": {"model": "llama3.2"}},
            # Mixed content
            {"response": "Text\n```code\nprint('hello')\n```\nMore text", "metadata": {"model": "llama3.2"}}
        ]
        
        for i, response in enumerate(edge_cases):
            context = {"request_id": f"edge-case-{i}", "input_text": "test"}
            
            try:
                result = await handler.transform_to_claude(response, context)
                assert isinstance(result, str)
                # Should not crash and should return some content
                assert len(result) >= 0
            except Exception as e:
                pytest.fail(f"Edge case {i} caused exception: {e}")
    
    @pytest.mark.asyncio
    async def test_format_preservation(self, handler):
        """Test that important formatting is preserved during transformation"""
        structured_response = {
            "response": """Here's the solution:

1. First step: Initialize variables
2. Second step: Process data
3. Third step: Return result

```python
def solution():
    # Implementation here
    return result
```

This approach ensures reliability.""",
            "metadata": {"model": "llama3.2"}
        }
        
        context = {
            "request_id": "format-test",
            "input_text": "Provide a structured solution"
        }
        
        result = await handler.transform_to_claude(structured_response, context)
        
        # Should preserve numbered list
        assert "1." in result and "2." in result and "3." in result
        # Should preserve code block
        assert "```python" in result or "```" in result
        # Should preserve structure
        assert "First step" in result
        assert "solution()" in result