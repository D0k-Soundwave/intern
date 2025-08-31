"""
Interview Process Manager - Dynamic Routing Decision System

Conducts interviews to determine optimal routing between Claude Code and Ollama.
Uses 5-component system for intelligent, bias-free routing decisions.

Per requirements.md: Complete implementation, no placeholders.
"""
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

class ModelRecommendation(Enum):
    """Model recommendation options"""
    CLAUDE = "claude"
    OLLAMA = "ollama"

@dataclass 
class RequirementAnalysis:
    """Analysis of request requirements"""
    technical_complexity: float  # 0.0 to 1.0
    domain_expertise_needed: float
    real_time_requirements: bool
    multi_step_process: bool
    code_generation_heavy: bool
    data_analysis_heavy: bool
    creative_requirements: bool
    accuracy_criticality: float

@dataclass
class ComplexityScore:
    """Complexity scoring breakdown"""
    linguistic_complexity: float
    technical_depth: float
    domain_specificity: float
    multi_modal_requirements: float
    overall_score: float

@dataclass
class CapabilityAssessment:
    """Capability matching results"""
    claude_capability_score: float
    ollama_capability_score: float
    capability_gap: float
    critical_missing_capabilities: List[str]

@dataclass
class BiasAnalysis:
    """Bias detection and compensation"""
    detected_biases: List[str]
    bias_strength: float  # 0.0 to 1.0
    compensation_factor: float
    bias_adjusted_scores: Dict[str, float]

@dataclass
class InterviewResult:
    """Complete interview result"""
    recommended_model: str
    confidence: float
    reasoning: str
    complexity_score: float
    ollama_preference_applied: bool
    bias_detected: bool
    fallback_reason: Optional[str]
    metadata: Dict[str, Any]

class InterviewProcessManager:
    """
    Complete interview-based routing system
    
    Uses 5 core components:
    1. RequirementsAnalyzer - Analyzes request requirements
    2. ComplexityScorer - Scores complexity on multiple dimensions  
    3. CapabilityMatcher - Matches requirements to model capabilities
    4. BiasDetector - Detects and compensates for routing biases
    5. OffloadingDecisionEngine - Makes final routing decision with 3% rule
    """
    
    def __init__(self):
        self.requirements_analyzer = RequirementsAnalyzer()
        self.complexity_scorer = ComplexityScorer()
        self.capability_matcher = CapabilityMatcher()
        self.bias_detector = BiasDetector()
        self.decision_engine = OffloadingDecisionEngine()
        
        # Performance tracking for learning
        self.decision_history = []
        
        logger.info("Interview Process Manager initialized")
    
    async def conduct_interview(
        self, 
        request_data: Dict[str, Any], 
        category_info: Dict[str, Any]
    ) -> InterviewResult:
        """
        Conduct complete interview process for routing decision
        
        Args:
            request_data: Contains input_text, context, request_id
            category_info: Category classification results
            
        Returns:
            InterviewResult with routing decision and reasoning
        """
        try:
            request_id = request_data.get('request_id', 'unknown')
            input_text = request_data.get('input_text', '')
            
            logger.info(f"Starting interview for request {request_id}")
            
            # Step 1: Analyze requirements
            requirements = await self.requirements_analyzer.analyze(
                input_text, category_info
            )
            
            # Step 2: Score complexity
            complexity = await self.complexity_scorer.score(
                input_text, category_info, requirements
            )
            
            # Step 3: Match capabilities
            capabilities = await self.capability_matcher.assess(
                requirements, complexity, category_info
            )
            
            # Step 4: Detect and compensate for bias
            bias_analysis = await self.bias_detector.analyze(
                input_text, category_info, capabilities
            )
            
            # Step 5: Make final routing decision
            decision = await self.decision_engine.decide(
                requirements, complexity, capabilities, bias_analysis
            )
            
            # Create interview result
            result = InterviewResult(
                recommended_model=decision['model'],
                confidence=decision['confidence'],
                reasoning=decision['reasoning'],
                complexity_score=complexity.overall_score,
                ollama_preference_applied=decision['ollama_preference_applied'],
                bias_detected=bias_analysis.bias_strength > 0.7,
                fallback_reason=decision.get('fallback_reason'),
                metadata={
                    'request_id': request_id,
                    'category': category_info.get('name', 'unknown'),
                    'requirements': requirements.__dict__,
                    'complexity_breakdown': complexity.__dict__,
                    'capability_scores': {
                        'claude': capabilities.claude_capability_score,
                        'ollama': capabilities.ollama_capability_score
                    },
                    'bias_analysis': bias_analysis.__dict__,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Store for performance tracking
            self._record_decision(result)
            
            logger.info(
                f"Interview complete for {request_id}: "
                f"{result.recommended_model} ({result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in interview process: {e}")
            # Return safe fallback
            return InterviewResult(
                recommended_model="claude",
                confidence=0.1,
                reasoning=f"Interview failed, defaulting to Claude: {str(e)}",
                complexity_score=0.5,
                ollama_preference_applied=False,
                bias_detected=False,
                fallback_reason="interview_error",
                metadata={"error": str(e)}
            )
    
    def _record_decision(self, result: InterviewResult):
        """Record decision for performance analysis"""
        decision_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model': result.recommended_model,
            'confidence': result.confidence,
            'complexity': result.complexity_score,
            'bias_detected': result.bias_detected,
            'request_hash': hashlib.md5(
                result.metadata.get('request_id', '').encode()
            ).hexdigest()[:8]
        }
        
        self.decision_history.append(decision_record)
        
        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get interview process performance statistics"""
        if not self.decision_history:
            return {"message": "No decisions recorded yet"}
        
        total_decisions = len(self.decision_history)
        claude_decisions = sum(1 for d in self.decision_history if d['model'] == 'claude')
        ollama_decisions = total_decisions - claude_decisions
        
        avg_confidence = sum(d['confidence'] for d in self.decision_history) / total_decisions
        avg_complexity = sum(d['complexity'] for d in self.decision_history) / total_decisions
        
        bias_detections = sum(1 for d in self.decision_history if d['bias_detected'])
        
        return {
            'total_decisions': total_decisions,
            'model_distribution': {
                'claude': claude_decisions,
                'ollama': ollama_decisions,
                'claude_percentage': (claude_decisions / total_decisions) * 100,
                'ollama_percentage': (ollama_decisions / total_decisions) * 100
            },
            'average_confidence': avg_confidence,
            'average_complexity': avg_complexity,
            'bias_detections': bias_detections,
            'bias_detection_rate': (bias_detections / total_decisions) * 100
        }


class RequirementsAnalyzer:
    """Analyzes request requirements to understand what's needed"""
    
    async def analyze(self, input_text: str, category_info: Dict[str, Any]) -> RequirementAnalysis:
        """Analyze requirements from input text and category"""
        
        # Technical complexity indicators
        technical_indicators = [
            'implement', 'develop', 'architecture', 'system', 'complex',
            'algorithm', 'optimize', 'performance', 'scalable', 'distributed'
        ]
        tech_score = sum(1 for indicator in technical_indicators if indicator in input_text.lower()) / len(technical_indicators)
        
        # Domain expertise indicators
        domain_indicators = [
            'machine learning', 'deep learning', 'blockchain', 'quantum',
            'security', 'cryptography', 'medical', 'legal', 'financial'
        ]
        domain_score = sum(1 for indicator in domain_indicators if indicator in input_text.lower()) / len(domain_indicators)
        
        # Real-time requirements
        realtime_keywords = ['real-time', 'live', 'streaming', 'instant', 'immediate']
        real_time_req = any(keyword in input_text.lower() for keyword in realtime_keywords)
        
        # Multi-step process detection
        multi_step_indicators = ['then', 'next', 'after', 'step', 'phase', 'workflow']
        multi_step = any(indicator in input_text.lower() for indicator in multi_step_indicators) or input_text.count('.') > 3
        
        # Code generation heavy
        code_keywords = ['code', 'function', 'class', 'method', 'implementation', 'program']
        code_heavy = sum(1 for keyword in code_keywords if keyword in input_text.lower()) > 2
        
        # Data analysis heavy
        data_keywords = ['data', 'analysis', 'statistics', 'chart', 'visualization', 'dataset']
        data_heavy = sum(1 for keyword in data_keywords if keyword in input_text.lower()) > 2
        
        # Creative requirements
        creative_keywords = ['creative', 'story', 'narrative', 'design', 'artistic', 'original']
        creative = any(keyword in input_text.lower() for keyword in creative_keywords)
        
        # Accuracy criticality
        accuracy_indicators = ['accurate', 'precise', 'exact', 'critical', 'production', 'enterprise']
        accuracy_score = sum(1 for indicator in accuracy_indicators if indicator in input_text.lower()) / len(accuracy_indicators)
        
        return RequirementAnalysis(
            technical_complexity=min(tech_score + (0.2 if category_info.get('complexity') == 'high' else 0), 1.0),
            domain_expertise_needed=domain_score,
            real_time_requirements=real_time_req,
            multi_step_process=multi_step,
            code_generation_heavy=code_heavy,
            data_analysis_heavy=data_heavy,
            creative_requirements=creative,
            accuracy_criticality=accuracy_score
        )


class ComplexityScorer:
    """Scores complexity across multiple dimensions"""
    
    async def score(
        self, 
        input_text: str, 
        category_info: Dict[str, Any], 
        requirements: RequirementAnalysis
    ) -> ComplexityScore:
        """Score complexity on multiple dimensions"""
        
        # Linguistic complexity
        word_count = len(input_text.split())
        sentence_count = input_text.count('.') + input_text.count('!') + input_text.count('?')
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        linguistic_score = min((avg_sentence_length - 5) / 20, 1.0) if avg_sentence_length > 5 else 0.0
        
        # Technical depth
        technical_terms = [
            'algorithm', 'optimization', 'architecture', 'framework', 'protocol',
            'infrastructure', 'scalability', 'performance', 'concurrency', 'distributed'
        ]
        tech_depth = sum(1 for term in technical_terms if term in input_text.lower()) / len(technical_terms)
        
        # Domain specificity
        complexity_mapping = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        domain_specificity = complexity_mapping.get(category_info.get('complexity', 'medium'), 0.5)
        
        # Multi-modal requirements
        multimodal_indicators = ['image', 'video', 'audio', 'file', 'document', 'chart', 'graph']
        multimodal_score = min(sum(1 for indicator in multimodal_indicators if indicator in input_text.lower()) / 3, 1.0)
        
        # Calculate overall score
        overall = (
            linguistic_score * 0.2 +
            tech_depth * 0.3 +
            domain_specificity * 0.3 +
            multimodal_score * 0.2
        )
        
        # Boost from requirements
        if requirements.technical_complexity > 0.7:
            overall += 0.1
        if requirements.multi_step_process:
            overall += 0.05
        
        overall = min(overall, 1.0)
        
        return ComplexityScore(
            linguistic_complexity=linguistic_score,
            technical_depth=tech_depth,
            domain_specificity=domain_specificity,
            multi_modal_requirements=multimodal_score,
            overall_score=overall
        )


class CapabilityMatcher:
    """Matches requirements to model capabilities"""
    
    def __init__(self):
        # Define model capabilities based on empirical knowledge
        self.claude_capabilities = {
            'code_generation': 0.95,
            'technical_analysis': 0.90,
            'creative_writing': 0.85,
            'complex_reasoning': 0.90,
            'domain_expertise': 0.85,
            'multi_step_tasks': 0.88,
            'accuracy': 0.92,
            'safety': 0.95,
            'context_understanding': 0.90
        }
        
        self.ollama_capabilities = {
            'code_generation': 0.75,
            'technical_analysis': 0.70,
            'creative_writing': 0.65,
            'complex_reasoning': 0.65,
            'domain_expertise': 0.60,
            'multi_step_tasks': 0.55,
            'accuracy': 0.70,
            'safety': 0.75,
            'context_understanding': 0.65
        }
    
    async def assess(
        self, 
        requirements: RequirementAnalysis, 
        complexity: ComplexityScore, 
        category_info: Dict[str, Any]
    ) -> CapabilityAssessment:
        """Assess how well each model matches the requirements"""
        
        # Calculate capability requirements
        requirement_weights = {
            'code_generation': 1.0 if requirements.code_generation_heavy else 0.3,
            'technical_analysis': requirements.technical_complexity,
            'creative_writing': 1.0 if requirements.creative_requirements else 0.2,
            'complex_reasoning': complexity.overall_score,
            'domain_expertise': requirements.domain_expertise_needed,
            'multi_step_tasks': 1.0 if requirements.multi_step_process else 0.3,
            'accuracy': requirements.accuracy_criticality,
            'safety': 0.8,  # Always important
            'context_understanding': min(complexity.linguistic_complexity + 0.3, 1.0)
        }
        
        # Calculate weighted capability scores
        claude_score = sum(
            self.claude_capabilities[cap] * weight 
            for cap, weight in requirement_weights.items()
        ) / sum(requirement_weights.values())
        
        ollama_score = sum(
            self.ollama_capabilities[cap] * weight 
            for cap, weight in requirement_weights.items()
        ) / sum(requirement_weights.values())
        
        # Identify critical missing capabilities for Ollama
        critical_missing = []
        for cap, weight in requirement_weights.items():
            if weight > 0.7 and self.ollama_capabilities[cap] < 0.7:
                if self.claude_capabilities[cap] - self.ollama_capabilities[cap] > 0.15:
                    critical_missing.append(cap)
        
        capability_gap = claude_score - ollama_score
        
        return CapabilityAssessment(
            claude_capability_score=claude_score,
            ollama_capability_score=ollama_score,
            capability_gap=capability_gap,
            critical_missing_capabilities=critical_missing
        )


class BiasDetector:
    """Detects and compensates for routing biases"""
    
    async def analyze(
        self, 
        input_text: str, 
        category_info: Dict[str, Any], 
        capabilities: CapabilityAssessment
    ) -> BiasAnalysis:
        """Detect biases and calculate compensation factors"""
        
        detected_biases = []
        bias_strength = 0.0
        
        # Complexity bias - tendency to over-route complex tasks to Claude
        if category_info.get('complexity') == 'high' and capabilities.capability_gap < 0.1:
            detected_biases.append('complexity_bias')
            bias_strength += 0.3
        
        # Safety bias - over-cautious routing to Claude for any security/safety concerns
        safety_keywords = ['security', 'safe', 'protect', 'confidential', 'private']
        if any(keyword in input_text.lower() for keyword in safety_keywords):
            if capabilities.ollama_capability_score > 0.6:
                detected_biases.append('safety_bias')
                bias_strength += 0.2
        
        # Brand bias - preference for Claude due to brand recognition
        if capabilities.capability_gap < 0.05:
            detected_biases.append('brand_bias')
            bias_strength += 0.2
        
        # Recency bias - over-weighting recent failures
        detected_biases.append('recency_bias')  # Always present to some degree
        bias_strength += 0.1
        
        # Calculate compensation factor
        compensation_factor = 0.0
        if bias_strength > 0.7:  # Significant bias detected
            compensation_factor = min(bias_strength * 0.1, 0.15)
        
        # Apply bias compensation to capability scores
        bias_adjusted_scores = {
            'claude': capabilities.claude_capability_score - compensation_factor,
            'ollama': capabilities.ollama_capability_score + compensation_factor
        }
        
        return BiasAnalysis(
            detected_biases=detected_biases,
            bias_strength=bias_strength,
            compensation_factor=compensation_factor,
            bias_adjusted_scores=bias_adjusted_scores
        )


class OffloadingDecisionEngine:
    """Makes final routing decision with 3% local preference rule"""
    
    async def decide(
        self,
        requirements: RequirementAnalysis,
        complexity: ComplexityScore,
        capabilities: CapabilityAssessment,
        bias_analysis: BiasAnalysis
    ) -> Dict[str, Any]:
        """Make final routing decision with all factors considered"""
        
        # Use bias-adjusted scores
        claude_score = bias_analysis.bias_adjusted_scores['claude']
        ollama_score = bias_analysis.bias_adjusted_scores['ollama']
        
        # Apply 3% local preference rule for Ollama
        ollama_preference = 0.03
        ollama_adjusted_score = ollama_score + ollama_preference
        ollama_preference_applied = False
        
        # Safety checks - always route to Claude for critical safety requirements
        if requirements.accuracy_criticality > 0.8:
            return {
                'model': 'claude',
                'confidence': 0.95,
                'reasoning': 'High accuracy requirements mandate Claude routing',
                'ollama_preference_applied': False,
                'safety_override': True
            }
        
        # Critical missing capabilities check
        if len(capabilities.critical_missing_capabilities) > 2:
            return {
                'model': 'claude',
                'confidence': 0.90,
                'reasoning': f'Ollama missing critical capabilities: {capabilities.critical_missing_capabilities}',
                'ollama_preference_applied': False,
                'capability_override': True
            }
        
        # Main decision logic
        if ollama_adjusted_score >= claude_score:
            # Ollama wins (including with 3% preference)
            if ollama_adjusted_score > ollama_score:
                ollama_preference_applied = True
            
            confidence = min(ollama_adjusted_score, 0.95)
            reasoning_parts = []
            
            if ollama_preference_applied:
                reasoning_parts.append("3% local preference applied")
            
            if bias_analysis.compensation_factor > 0:
                reasoning_parts.append(f"bias compensation applied ({bias_analysis.compensation_factor:.3f})")
            
            reasoning_parts.append(f"Ollama capability score: {ollama_score:.3f}")
            
            return {
                'model': 'ollama',
                'confidence': confidence,
                'reasoning': f"Ollama selected - {', '.join(reasoning_parts)}",
                'ollama_preference_applied': ollama_preference_applied
            }
        else:
            # Claude wins
            confidence = min(claude_score, 0.95)
            reasoning_parts = [f"Claude capability score: {claude_score:.3f}"]
            
            if bias_analysis.compensation_factor > 0:
                reasoning_parts.append(f"bias compensation applied ({bias_analysis.compensation_factor:.3f})")
            
            return {
                'model': 'claude',
                'confidence': confidence,
                'reasoning': f"Claude selected - {', '.join(reasoning_parts)}",
                'ollama_preference_applied': False
            }