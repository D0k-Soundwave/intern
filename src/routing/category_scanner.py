"""
Category Scanner for Claude Code Input Analysis

Scans and categorizes input from Claude Code using the complete 76-category system.
Uses keyword matching, pattern recognition, and intent analysis.

Per requirements.md: Complete implementation, no placeholders.
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .categories import CATEGORIES, get_category_by_id, get_all_categories

logger = logging.getLogger(__name__)

@dataclass
class CategoryMatch:
    """Represents a category match with confidence score"""
    category_id: int
    category: Dict[str, Any]
    confidence: float
    matched_keywords: List[str]
    matched_patterns: List[str]
    intent_matches: List[str]

@dataclass
class ScanResult:
    """Result of category scanning operation"""
    category: Dict[str, Any]  # Primary category
    confidence: float
    all_matches: List[CategoryMatch]
    confidence_scores: Dict[int, float]
    intents: List[str]
    keywords_found: List[str]

class CategoryScanner:
    """
    Advanced category scanner for Claude Code input
    
    Uses multiple techniques:
    - Keyword matching with fuzzy logic
    - Pattern recognition for technical terms
    - Intent analysis based on linguistic patterns
    - Confidence scoring with multiple factors
    """
    
    def __init__(self):
        self.categories = get_all_categories()
        self._compile_patterns()
        logger.info("Category Scanner initialized with 76 categories")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.code_patterns = [
            r'\b(function|class|def|import|from|public|private|void|int|string)\b',
            r'\b(async|await|promise|callback|lambda|arrow)\b',
            r'\b(html|css|javascript|python|java|cpp|rust|go|php)\b',
            r'[{}();]',
            r'\b(git|commit|push|pull|branch|merge|clone)\b'
        ]
        
        self.data_patterns = [
            r'\b(data|dataset|csv|json|xml|database|sql|query)\b',
            r'\b(analyze|analysis|statistics|chart|graph|plot)\b',
            r'\b(model|training|prediction|ml|ai|neural|network)\b',
            r'\b(pandas|numpy|matplotlib|sklearn|tensorflow|pytorch)\b'
        ]
        
        self.system_patterns = [
            r'\b(deploy|deployment|docker|kubernetes|k8s|ci|cd)\b',
            r'\b(server|client|api|rest|graphql|microservice)\b',
            r'\b(cloud|aws|azure|gcp|serverless|lambda)\b',
            r'\b(monitoring|metrics|logging|observability)\b'
        ]
        
        self.security_patterns = [
            r'\b(security|secure|encrypt|decrypt|auth|authorization)\b',
            r'\b(vulnerability|exploit|attack|threat|risk)\b',
            r'\b(compliance|audit|gdpr|hipaa|sox|pci)\b',
            r'\b(penetration|pentest|vulnerability|scan)\b'
        ]
        
        self.business_patterns = [
            r'\b(business|strategy|market|competitor|analysis)\b',
            r'\b(finance|financial|budget|roi|investment|revenue)\b',
            r'\b(project|plan|timeline|milestone|resource)\b',
            r'\b(marketing|sales|customer|acquisition|conversion)\b'
        ]
        
        # Compile all patterns
        self.compiled_patterns = {
            'code': [re.compile(p, re.IGNORECASE) for p in self.code_patterns],
            'data': [re.compile(p, re.IGNORECASE) for p in self.data_patterns],
            'system': [re.compile(p, re.IGNORECASE) for p in self.system_patterns],
            'security': [re.compile(p, re.IGNORECASE) for p in self.security_patterns],
            'business': [re.compile(p, re.IGNORECASE) for p in self.business_patterns]
        }
    
    async def scan_claude_input(self, input_text: str) -> ScanResult:
        """
        Main scanning function for Claude Code input
        
        Args:
            input_text: The user's request from Claude Code
            
        Returns:
            ScanResult with primary category and all matches
        """
        try:
            logger.info(f"Scanning input: {input_text[:100]}...")
            
            # Clean and normalize input
            normalized_text = self._normalize_input(input_text)
            
            # Extract intents and context
            intents = self._extract_intents(normalized_text)
            keywords_found = self._extract_keywords(normalized_text)
            
            # Scan all categories
            all_matches = []
            confidence_scores = {}
            
            for category_id, category_info in self.categories.items():
                match_result = self._match_category(
                    normalized_text, category_id, category_info, intents, keywords_found
                )
                
                if match_result.confidence > 0.1:  # Threshold for consideration
                    all_matches.append(match_result)
                    confidence_scores[category_id] = match_result.confidence
            
            # Sort by confidence and select primary
            all_matches.sort(key=lambda x: x.confidence, reverse=True)
            
            # Select primary category
            if all_matches:
                primary_category = all_matches[0].category
                primary_confidence = all_matches[0].confidence
            else:
                # Default to unknown_general if no matches
                primary_category = get_category_by_id(76)
                primary_confidence = 0.5
            
            result = ScanResult(
                category=primary_category,
                confidence=primary_confidence,
                all_matches=all_matches,
                confidence_scores=confidence_scores,
                intents=intents,
                keywords_found=keywords_found
            )
            
            logger.info(
                f"Scan complete: Primary={primary_category['name']} "
                f"({primary_confidence:.2f}), Total matches={len(all_matches)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning input: {e}")
            # Return safe default
            return ScanResult(
                category=get_category_by_id(76),
                confidence=0.1,
                all_matches=[],
                confidence_scores={76: 0.1},
                intents=[],
                keywords_found=[]
            )
    
    def _normalize_input(self, text: str) -> str:
        """Normalize input text for better matching"""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep programming symbols
        normalized = re.sub(r'[^\w\s\-_\.\(\)\[\]\{\};:,]', ' ', normalized)
        
        return normalized.strip()
    
    def _extract_intents(self, text: str) -> List[str]:
        """Extract user intents from text using linguistic patterns"""
        intents = []
        
        # Action verbs that indicate intent
        action_patterns = {
            'create': r'\b(create|make|build|generate|develop|write|implement)\b',
            'fix': r'\b(fix|repair|debug|solve|resolve|correct)\b',
            'analyze': r'\b(analyze|review|examine|study|investigate|check)\b',
            'optimize': r'\b(optimize|improve|enhance|refactor|upgrade)\b',
            'deploy': r'\b(deploy|publish|release|launch|install)\b',
            'test': r'\b(test|verify|validate|check|ensure)\b',
            'explain': r'\b(explain|describe|show|tell|help|teach)\b',
            'design': r'\b(design|architect|plan|structure|model)\b'
        }
        
        for intent, pattern in action_patterns.items():
            if re.search(pattern, text):
                intents.append(intent)
        
        # Question patterns
        if re.search(r'\b(what|how|why|when|where|which)\b', text):
            intents.append('question')
        
        if re.search(r'\?', text):
            intents.append('question')
        
        return list(set(intents))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        keywords = []
        
        # Technical keywords
        tech_keywords = [
            'api', 'database', 'server', 'client', 'frontend', 'backend',
            'docker', 'kubernetes', 'cloud', 'aws', 'azure', 'security',
            'auth', 'testing', 'deployment', 'monitoring', 'logging'
        ]
        
        for keyword in tech_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        # Programming languages
        languages = [
            'python', 'javascript', 'java', 'cpp', 'rust', 'go',
            'typescript', 'php', 'ruby', 'swift', 'kotlin'
        ]
        
        for lang in languages:
            if lang in text:
                keywords.append(lang)
        
        return keywords
    
    def _match_category(
        self, 
        text: str, 
        category_id: int, 
        category_info: Dict[str, Any],
        intents: List[str],
        keywords_found: List[str]
    ) -> CategoryMatch:
        """Match text against a specific category"""
        
        matched_keywords = []
        matched_patterns = []
        intent_matches = []
        
        # Keyword matching
        keyword_score = 0.0
        for keyword in category_info['keywords']:
            if keyword.lower() in text:
                matched_keywords.append(keyword)
                keyword_score += 1.0
        
        # Normalize keyword score
        if category_info['keywords']:
            keyword_score = keyword_score / len(category_info['keywords'])
        
        # Intent matching
        intent_score = 0.0
        for intent in category_info['typical_intents']:
            intent_keywords = intent.split('_')
            for intent_keyword in intent_keywords:
                if intent_keyword in text or any(intent_keyword in i for i in intents):
                    intent_matches.append(intent)
                    intent_score += 0.5
                    break
        
        # Normalize intent score
        if category_info['typical_intents']:
            intent_score = min(intent_score / len(category_info['typical_intents']), 1.0)
        
        # Pattern matching based on category type
        pattern_score = self._get_pattern_score(text, category_id)
        
        # Context matching
        context_score = self._get_context_score(text, category_info, keywords_found)
        
        # Calculate weighted confidence
        confidence = (
            keyword_score * 0.4 +      # Keywords are most important
            intent_score * 0.3 +       # Intent matching
            pattern_score * 0.2 +      # Pattern recognition
            context_score * 0.1        # Contextual clues
        )
        
        # Boost confidence for exact matches
        if any(keyword.lower() == text.strip() for keyword in category_info['keywords']):
            confidence += 0.2
        
        # Complexity adjustment - higher complexity categories get slight boost for complex queries
        if category_info['complexity'] == 'high' and len(text.split()) > 20:
            confidence += 0.05
        elif category_info['complexity'] == 'low' and len(text.split()) <= 5:
            confidence += 0.05
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        return CategoryMatch(
            category_id=category_id,
            category=category_info,
            confidence=confidence,
            matched_keywords=matched_keywords,
            matched_patterns=matched_patterns,
            intent_matches=intent_matches
        )
    
    def _get_pattern_score(self, text: str, category_id: int) -> float:
        """Get pattern matching score based on category type"""
        pattern_scores = {
            'code': 0.0,
            'data': 0.0,
            'system': 0.0,
            'security': 0.0,
            'business': 0.0
        }
        
        # Check each pattern type
        for pattern_type, patterns in self.compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
            pattern_scores[pattern_type] = min(matches / len(patterns), 1.0)
        
        # Map category ranges to pattern types
        if 1 <= category_id <= 15:  # Programming & Development
            return pattern_scores['code']
        elif 16 <= category_id <= 25:  # Data & Analytics
            return pattern_scores['data']
        elif 36 <= category_id <= 45:  # System & Architecture
            return pattern_scores['system']
        elif 56 <= category_id <= 65:  # Security & Compliance
            return pattern_scores['security']
        elif 46 <= category_id <= 55:  # Business & Strategy
            return pattern_scores['business']
        else:
            # For other categories, use average of all patterns
            return sum(pattern_scores.values()) / len(pattern_scores)
    
    def _get_context_score(
        self, 
        text: str, 
        category_info: Dict[str, Any], 
        keywords_found: List[str]
    ) -> float:
        """Get contextual score based on surrounding context"""
        score = 0.0
        
        # Check if category capabilities match found keywords
        for capability in category_info.get('capabilities', []):
            if any(keyword in capability.lower() for keyword in keywords_found):
                score += 0.1
        
        # Check text length appropriateness for category
        text_length = len(text.split())
        if category_info['complexity'] == 'high' and text_length > 10:
            score += 0.1
        elif category_info['complexity'] == 'low' and text_length <= 10:
            score += 0.1
        
        # Check for technical indicators
        if category_info['complexity'] == 'high':
            technical_indicators = ['implement', 'develop', 'create', 'build', 'design']
            if any(indicator in text for indicator in technical_indicators):
                score += 0.05
        
        return min(score, 1.0)
    
    async def get_category_suggestions(self, partial_input: str) -> List[Dict[str, Any]]:
        """Get category suggestions for partial input (for auto-completion)"""
        try:
            if len(partial_input) < 2:
                return []
            
            suggestions = []
            normalized_input = self._normalize_input(partial_input)
            
            for category_id, category_info in self.categories.items():
                # Check if any keyword starts with the partial input
                for keyword in category_info['keywords']:
                    if keyword.lower().startswith(normalized_input):
                        suggestions.append({
                            'category_id': category_id,
                            'category_name': category_info['name'],
                            'matched_keyword': keyword,
                            'description': category_info['description']
                        })
                        break
            
            return suggestions[:10]  # Limit to top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error getting category suggestions: {e}")
            return []
    
    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about the category system"""
        complexity_counts = defaultdict(int)
        capability_counts = defaultdict(int)
        
        for category_info in self.categories.values():
            complexity_counts[category_info['complexity']] += 1
            
            for capability in category_info.get('capabilities', []):
                capability_counts[capability] += 1
        
        return {
            'total_categories': len(self.categories),
            'complexity_distribution': dict(complexity_counts),
            'top_capabilities': dict(sorted(
                capability_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),
            'keywords_per_category': {
                cat_id: len(cat_info['keywords']) 
                for cat_id, cat_info in self.categories.items()
            }
        }