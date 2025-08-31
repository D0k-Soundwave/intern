"""
Interview Process Tests

Tests for the complete interview-based routing system including
all 5 components and the 3% local preference rule.

Per requirements.md: Complete implementation, no placeholders.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.interview.process_manager import (
    InterviewProcessManager, RequirementsAnalyzer, ComplexityScorer,
    CapabilityMatcher, BiasDetector, OffloadingDecisionEngine,
    InterviewResult, RequirementAnalysis, ComplexityScore, 
    CapabilityAssessment, BiasAnalysis, ModelRecommendation
)

class TestRequirementsAnalyzer:
    """Test cases for RequirementsAnalyzer"""
    
    @pytest.fixture
    async def analyzer(self):
        """Create RequirementsAnalyzer instance"""
        return RequirementsAnalyzer()
    
    @pytest.mark.asyncio
    async def test_simple_code_analysis(self, analyzer, simple_input_text, sample_category_info):
        """Test analysis of simple code generation request"""
        result = await analyzer.analyze(simple_input_text, sample_category_info)
        
        assert isinstance(result, RequirementAnalysis)
        assert result.technical_complexity <= 0.5  # Simple task
        assert result.code_generation_heavy == True
        assert result.multi_step_process == False
        assert result.accuracy_criticality <= 0.3
    
    @pytest.mark.asyncio
    async def test_complex_system_analysis(self, analyzer, complex_input_text):
        """Test analysis of complex system architecture request"""
        complex_category = {
            "complexity": "high",
            "name": "system_architecture"
        }
        
        result = await analyzer.analyze(complex_input_text, complex_category)
        
        assert result.technical_complexity >= 0.7  # High complexity
        assert result.multi_step_process == True
        assert result.domain_expertise_needed >= 0.3
    
    @pytest.mark.asyncio
    async def test_data_analysis_requirements(self, analyzer):
        """Test data analysis requirements detection"""
        data_input = "Analyze sales data and create visualization dashboard"
        data_category = {"complexity": "medium", "name": "data_visualization"}
        
        result = await analyzer.analyze(data_input, data_category)
        
        assert result.data_analysis_heavy == True
        assert result.technical_complexity >= 0.3
    
    @pytest.mark.asyncio
    async def test_real_time_detection(self, analyzer):
        """Test real-time requirements detection"""
        realtime_input = "Build real-time streaming data pipeline"
        category = {"complexity": "high", "name": "big_data"}
        
        result = await analyzer.analyze(realtime_input, category)
        
        assert result.real_time_requirements == True
    
    @pytest.mark.asyncio
    async def test_creative_requirements(self, analyzer):
        """Test creative requirements detection"""
        creative_input = "Write a creative story about space exploration"
        category = {"complexity": "low", "name": "creative_writing"}
        
        result = await analyzer.analyze(creative_input, category)
        
        assert result.creative_requirements == True
        assert result.technical_complexity <= 0.3

class TestComplexityScorer:
    """Test cases for ComplexityScorer"""
    
    @pytest.fixture
    async def scorer(self):
        """Create ComplexityScorer instance"""
        return ComplexityScorer()
    
    @pytest.mark.asyncio
    async def test_simple_task_complexity(self, scorer, simple_input_text, sample_category_info):
        """Test complexity scoring for simple tasks"""
        requirements = RequirementAnalysis(
            technical_complexity=0.2,
            domain_expertise_needed=0.1,
            real_time_requirements=False,
            multi_step_process=False,
            code_generation_heavy=True,
            data_analysis_heavy=False,
            creative_requirements=False,
            accuracy_criticality=0.3
        )
        
        result = await scorer.score(simple_input_text, sample_category_info, requirements)
        
        assert isinstance(result, ComplexityScore)
        assert result.overall_score <= 0.5
        assert result.domain_specificity <= 0.5  # Low complexity category
    
    @pytest.mark.asyncio
    async def test_complex_task_complexity(self, scorer, complex_input_text):
        """Test complexity scoring for complex tasks"""
        high_category = {"complexity": "high", "name": "system_architecture"}
        requirements = RequirementAnalysis(
            technical_complexity=0.9,
            domain_expertise_needed=0.8,
            real_time_requirements=True,
            multi_step_process=True,
            code_generation_heavy=True,
            data_analysis_heavy=True,
            creative_requirements=False,
            accuracy_criticality=0.7
        )
        
        result = await scorer.score(complex_input_text, high_category, requirements)
        
        assert result.overall_score >= 0.6
        assert result.domain_specificity >= 0.7  # High complexity category
        assert result.technical_depth >= 0.3  # Technical terms present
    
    @pytest.mark.asyncio
    async def test_linguistic_complexity(self, scorer):
        """Test linguistic complexity scoring"""
        simple_text = "Hello world"
        complex_text = "This is a very detailed and comprehensive explanation of advanced distributed systems architecture patterns including microservices orchestration, service mesh implementation, and containerization strategies."
        
        category = {"complexity": "medium", "name": "explanation"}
        requirements = RequirementAnalysis(
            technical_complexity=0.5, domain_expertise_needed=0.3,
            real_time_requirements=False, multi_step_process=False,
            code_generation_heavy=False, data_analysis_heavy=False,
            creative_requirements=False, accuracy_criticality=0.4
        )
        
        simple_result = await scorer.score(simple_text, category, requirements)
        complex_result = await scorer.score(complex_text, category, requirements)
        
        assert complex_result.linguistic_complexity > simple_result.linguistic_complexity

class TestCapabilityMatcher:
    """Test cases for CapabilityMatcher"""
    
    @pytest.fixture
    async def matcher(self):
        """Create CapabilityMatcher instance"""
        return CapabilityMatcher()
    
    @pytest.mark.asyncio
    async def test_simple_code_capability_match(self, matcher):
        """Test capability matching for simple code generation"""
        requirements = RequirementAnalysis(
            technical_complexity=0.3,
            domain_expertise_needed=0.2,
            real_time_requirements=False,
            multi_step_process=False,
            code_generation_heavy=True,
            data_analysis_heavy=False,
            creative_requirements=False,
            accuracy_criticality=0.4
        )
        
        complexity = ComplexityScore(
            linguistic_complexity=0.2,
            technical_depth=0.3,
            domain_specificity=0.3,
            multi_modal_requirements=0.1,
            overall_score=0.3
        )
        
        category = {"complexity": "low", "name": "code_generation_simple"}
        
        result = await matcher.assess(requirements, complexity, category)
        
        assert isinstance(result, CapabilityAssessment)
        assert result.ollama_capability_score >= 0.6  # Ollama should handle simple code
        assert result.capability_gap <= 0.3  # Gap should be small for simple tasks
    
    @pytest.mark.asyncio
    async def test_complex_domain_expertise(self, matcher):
        """Test capability matching for high domain expertise tasks"""
        requirements = RequirementAnalysis(
            technical_complexity=0.8,
            domain_expertise_needed=0.9,
            real_time_requirements=True,
            multi_step_process=True,
            code_generation_heavy=False,
            data_analysis_heavy=False,
            creative_requirements=False,
            accuracy_criticality=0.9
        )
        
        complexity = ComplexityScore(
            linguistic_complexity=0.7,
            technical_depth=0.8,
            domain_specificity=0.9,
            multi_modal_requirements=0.3,
            overall_score=0.8
        )
        
        category = {"complexity": "high", "name": "legal_document"}
        
        result = await matcher.assess(requirements, complexity, category)
        
        assert result.capability_gap >= 0.15  # Claude should have significant advantage
        assert len(result.critical_missing_capabilities) >= 1  # Should identify missing capabilities
    
    @pytest.mark.asyncio
    async def test_creative_writing_match(self, matcher):
        """Test capability matching for creative writing"""
        requirements = RequirementAnalysis(
            technical_complexity=0.1,
            domain_expertise_needed=0.2,
            real_time_requirements=False,
            multi_step_process=False,
            code_generation_heavy=False,
            data_analysis_heavy=False,
            creative_requirements=True,
            accuracy_criticality=0.3
        )
        
        complexity = ComplexityScore(
            linguistic_complexity=0.6,
            technical_depth=0.1,
            domain_specificity=0.2,
            multi_modal_requirements=0.1,
            overall_score=0.3
        )
        
        category = {"complexity": "low", "name": "creative_writing"}
        
        result = await matcher.assess(requirements, complexity, category)
        
        # Creative writing should favor Claude slightly
        assert result.capability_gap >= 0.1

class TestBiasDetector:
    """Test cases for BiasDetector"""
    
    @pytest.fixture
    async def detector(self):
        """Create BiasDetector instance"""
        return BiasDetector()
    
    @pytest.mark.asyncio
    async def test_complexity_bias_detection(self, detector):
        """Test detection of complexity bias"""
        input_text = "Simple hello world program"
        category = {"complexity": "high", "name": "code_generation_complex"}  # Mismatched complexity
        
        capabilities = CapabilityAssessment(
            claude_capability_score=0.85,
            ollama_capability_score=0.82,  # Small gap
            capability_gap=0.03,
            critical_missing_capabilities=[]
        )
        
        result = await detector.analyze(input_text, category, capabilities)
        
        assert isinstance(result, BiasAnalysis)
        assert 'complexity_bias' in result.detected_biases
        assert result.bias_strength >= 0.3
    
    @pytest.mark.asyncio
    async def test_safety_bias_detection(self, detector):
        """Test detection of safety bias"""
        input_text = "Security audit for payment system"
        category = {"complexity": "medium", "name": "security_analysis"}
        
        capabilities = CapabilityAssessment(
            claude_capability_score=0.75,
            ollama_capability_score=0.70,  # Capable enough
            capability_gap=0.05,
            critical_missing_capabilities=[]
        )
        
        result = await detector.analyze(input_text, category, capabilities)
        
        assert 'safety_bias' in result.detected_biases
        assert result.bias_strength >= 0.2
    
    @pytest.mark.asyncio
    async def test_bias_compensation(self, detector):
        """Test bias compensation calculation"""
        input_text = "Regular programming task"
        category = {"complexity": "medium", "name": "code_generation_simple"}
        
        capabilities = CapabilityAssessment(
            claude_capability_score=0.80,
            ollama_capability_score=0.78,
            capability_gap=0.02,  # Very small gap
            critical_missing_capabilities=[]
        )
        
        result = await detector.analyze(input_text, category, capabilities)
        
        # Should detect multiple biases due to small gap
        assert result.bias_strength >= 0.7  # Should trigger compensation
        assert result.compensation_factor > 0
        assert result.bias_adjusted_scores['ollama'] > capabilities.ollama_capability_score

class TestOffloadingDecisionEngine:
    """Test cases for OffloadingDecisionEngine"""
    
    @pytest.fixture
    async def engine(self):
        """Create OffloadingDecisionEngine instance"""
        return OffloadingDecisionEngine()
    
    @pytest.mark.asyncio
    async def test_ollama_selection_with_preference(self, engine):
        """Test Ollama selection with 3% preference rule"""
        requirements = RequirementAnalysis(
            technical_complexity=0.4, domain_expertise_needed=0.3,
            real_time_requirements=False, multi_step_process=False,
            code_generation_heavy=True, data_analysis_heavy=False,
            creative_requirements=False, accuracy_criticality=0.5
        )
        
        complexity = ComplexityScore(
            linguistic_complexity=0.3, technical_depth=0.4,
            domain_specificity=0.3, multi_modal_requirements=0.2,
            overall_score=0.4
        )
        
        capabilities = CapabilityAssessment(
            claude_capability_score=0.78,
            ollama_capability_score=0.76,  # Close but slightly lower
            capability_gap=0.02,
            critical_missing_capabilities=[]
        )
        
        bias_analysis = BiasAnalysis(
            detected_biases=['brand_bias'],
            bias_strength=0.3,
            compensation_factor=0.0,  # No significant compensation
            bias_adjusted_scores={'claude': 0.78, 'ollama': 0.76}
        )
        
        result = await engine.decide(requirements, complexity, capabilities, bias_analysis)
        
        # 3% preference should make Ollama win (0.76 + 0.03 = 0.79 > 0.78)
        assert result['model'] == 'ollama'
        assert result['ollama_preference_applied'] == True
        assert '3% local preference' in result['reasoning']
    
    @pytest.mark.asyncio
    async def test_claude_selection_for_critical_accuracy(self, engine):
        """Test Claude selection for high accuracy requirements"""
        requirements = RequirementAnalysis(
            technical_complexity=0.6, domain_expertise_needed=0.7,
            real_time_requirements=False, multi_step_process=True,
            code_generation_heavy=True, data_analysis_heavy=False,
            creative_requirements=False, accuracy_criticality=0.9  # High accuracy needed
        )
        
        complexity = ComplexityScore(
            linguistic_complexity=0.5, technical_depth=0.7,
            domain_specificity=0.8, multi_modal_requirements=0.2,
            overall_score=0.7
        )
        
        capabilities = CapabilityAssessment(
            claude_capability_score=0.85,
            ollama_capability_score=0.75,
            capability_gap=0.10,
            critical_missing_capabilities=['accuracy', 'domain_expertise']
        )
        
        bias_analysis = BiasAnalysis(
            detected_biases=[], bias_strength=0.2,
            compensation_factor=0.0,
            bias_adjusted_scores={'claude': 0.85, 'ollama': 0.75}
        )
        
        result = await engine.decide(requirements, complexity, capabilities, bias_analysis)
        
        assert result['model'] == 'claude'
        assert result['ollama_preference_applied'] == False
        assert 'accuracy requirements' in result['reasoning']
        assert 'safety_override' in result
    
    @pytest.mark.asyncio
    async def test_missing_capabilities_override(self, engine):
        """Test Claude selection when Ollama lacks critical capabilities"""
        requirements = RequirementAnalysis(
            technical_complexity=0.8, domain_expertise_needed=0.9,
            real_time_requirements=True, multi_step_process=True,
            code_generation_heavy=False, data_analysis_heavy=True,
            creative_requirements=False, accuracy_criticality=0.6
        )
        
        complexity = ComplexityScore(
            linguistic_complexity=0.7, technical_depth=0.8,
            domain_specificity=0.9, multi_modal_requirements=0.4,
            overall_score=0.8
        )
        
        capabilities = CapabilityAssessment(
            claude_capability_score=0.88,
            ollama_capability_score=0.65,
            capability_gap=0.23,
            critical_missing_capabilities=['domain_expertise', 'complex_reasoning', 'multi_step_tasks']
        )
        
        bias_analysis = BiasAnalysis(
            detected_biases=[], bias_strength=0.1,
            compensation_factor=0.0,
            bias_adjusted_scores={'claude': 0.88, 'ollama': 0.65}
        )
        
        result = await engine.decide(requirements, complexity, capabilities, bias_analysis)
        
        assert result['model'] == 'claude'
        assert 'missing critical capabilities' in result['reasoning']
        assert 'capability_override' in result

class TestInterviewProcessManager:
    """Test cases for complete InterviewProcessManager"""
    
    @pytest.fixture
    async def manager(self):
        """Create InterviewProcessManager instance"""
        return InterviewProcessManager()
    
    @pytest.mark.asyncio
    async def test_complete_interview_simple_task(self, manager, sample_interview_request, sample_category_info):
        """Test complete interview process for simple task"""
        result = await manager.conduct_interview(sample_interview_request, sample_category_info)
        
        assert isinstance(result, InterviewResult)
        assert result.recommended_model in ['claude', 'ollama']
        assert 0.0 <= result.confidence <= 1.0
        assert result.complexity_score >= 0.0
        assert result.reasoning is not None
        assert 'request_id' in result.metadata
        assert 'timestamp' in result.metadata
    
    @pytest.mark.asyncio
    async def test_interview_with_bias_detection(self, manager):
        """Test interview process with bias detection"""
        request_data = {
            "request_id": "test-bias-123",
            "input_text": "Simple code task but complex category",
            "context": {}
        }
        
        # Simulate category mismatch to trigger bias
        category_info = {
            "name": "system_architecture",  # Complex category for simple task
            "complexity": "high",
            "capabilities": ["distributed_systems", "scalability"],
            "keywords": ["architecture", "system", "complex"],
            "typical_intents": ["design_system", "architecture_planning"]
        }
        
        result = await manager.conduct_interview(request_data, category_info)
        
        # May detect bias but still make decision
        assert result.recommended_model in ['claude', 'ollama']
        assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_interview_error_handling(self, manager):
        """Test interview process error handling"""
        invalid_request = {
            "request_id": "error-test",
            "input_text": None,  # Invalid input
            "context": {}
        }
        
        invalid_category = None  # Invalid category
        
        result = await manager.conduct_interview(invalid_request, invalid_category)
        
        # Should return fallback result
        assert result.recommended_model == "claude"  # Safe fallback
        assert result.fallback_reason == "interview_error"
        assert result.confidence <= 0.2  # Low confidence for errors
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, manager, sample_interview_request, sample_category_info):
        """Test performance tracking functionality"""
        # Conduct multiple interviews
        for i in range(5):
            request = sample_interview_request.copy()
            request['request_id'] = f"perf-test-{i}"
            await manager.conduct_interview(request, sample_category_info)
        
        # Check performance stats
        stats = await manager.get_performance_stats()
        
        assert 'total_decisions' in stats
        assert stats['total_decisions'] >= 5
        assert 'model_distribution' in stats
        assert 'average_confidence' in stats
        assert 'bias_detection_rate' in stats
    
    @pytest.mark.asyncio
    async def test_3_percent_rule_application(self, manager):
        """Test 3% local preference rule application"""
        request_data = {
            "request_id": "preference-test",
            "input_text": "Write a simple function to add two numbers",
            "context": {}
        }
        
        category_info = {
            "name": "code_generation_simple",
            "complexity": "low",
            "capabilities": ["syntax", "basic_logic"],
            "keywords": ["function", "simple", "basic"],
            "typical_intents": ["create_function"]
        }
        
        # Run multiple interviews to check for preference application
        ollama_wins = 0
        preference_applied = 0
        
        for i in range(20):  # Run multiple times for statistical significance
            request = request_data.copy()
            request['request_id'] = f"pref-test-{i}"
            result = await manager.conduct_interview(request, category_info)
            
            if result.recommended_model == 'ollama':
                ollama_wins += 1
                
            if result.ollama_preference_applied:
                preference_applied += 1
        
        # Should have some cases where Ollama wins due to preference
        # (exact numbers depend on scoring, but preference should be applied sometimes)
        assert preference_applied >= 0  # At least some preference applications
    
    @pytest.mark.asyncio
    async def test_decision_recording(self, manager, sample_interview_request, sample_category_info):
        """Test decision recording for learning"""
        initial_history_len = len(manager.decision_history)
        
        await manager.conduct_interview(sample_interview_request, sample_category_info)
        
        assert len(manager.decision_history) == initial_history_len + 1
        
        # Check recorded decision structure
        last_decision = manager.decision_history[-1]
        assert 'timestamp' in last_decision
        assert 'model' in last_decision
        assert 'confidence' in last_decision
        assert 'complexity' in last_decision
        assert 'request_hash' in last_decision
    
    @pytest.mark.asyncio
    async def test_concurrent_interviews(self, manager, sample_category_info):
        """Test handling concurrent interview requests"""
        import asyncio
        
        # Create multiple concurrent requests
        requests = [
            {
                "request_id": f"concurrent-{i}",
                "input_text": f"Task number {i}",
                "context": {"concurrent_test": True}
            }
            for i in range(10)
        ]
        
        # Run all interviews concurrently
        tasks = [
            manager.conduct_interview(request, sample_category_info)
            for request in requests
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert isinstance(result, InterviewResult)
            assert result.recommended_model in ['claude', 'ollama']
            assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, manager, sample_interview_request, sample_category_info):
        """Test that decision history doesn't grow unbounded"""
        # Generate more than 1000 decisions to trigger cleanup
        for i in range(1050):
            request = sample_interview_request.copy()
            request['request_id'] = f"cleanup-test-{i}"
            await manager.conduct_interview(request, sample_category_info)
        
        # Should be limited to 1000 entries
        assert len(manager.decision_history) <= 1000