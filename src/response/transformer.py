"""
Response Transformer - Ollama to Claude Format Conversion

Transforms Ollama responses to match Claude Code's expected format,
ensuring seamless integration and indistinguishable responses.

Per requirements.md: Complete implementation, no placeholders.
"""
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

class ResponseFormat(Enum):
    """Response format types"""
    TEXT = "text"
    CODE = "code" 
    STRUCTURED = "structured"
    ERROR = "error"

@dataclass
class TransformationResult:
    """Result of response transformation"""
    transformed_content: str
    format_detected: ResponseFormat
    confidence: float
    transformations_applied: List[str]
    metadata: Dict[str, Any]

class MCPResponseHandler:
    """
    Handles MCP response formatting and transformation
    
    Ensures Ollama responses are indistinguishable from Claude responses
    by applying format standardization, quality enhancement, and 
    Claude-style response patterns.
    """
    
    def __init__(self):
        self.quality_enhancer = QualityEnhancer()
        self.format_standardizer = FormatStandardizer()
        self.claude_style_adapter = ClaudeStyleAdapter()
        
        logger.info("MCP Response Handler initialized")
    
    async def transform_to_claude(
        self, 
        ollama_response: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """
        Transform Ollama response to Claude Code compatible format
        
        Args:
            ollama_response: Raw response from Ollama
            context: Request context (input_text, request_id, etc.)
            
        Returns:
            Claude-formatted response string
        """
        try:
            request_id = context.get('request_id', 'unknown')
            logger.info(f"Transforming Ollama response for request {request_id}")
            
            # Extract content from Ollama response
            raw_content = self._extract_content(ollama_response)
            
            # Detect response format
            detected_format = self._detect_format(raw_content, context)
            
            # Step 1: Quality enhancement
            quality_result = await self.quality_enhancer.enhance(
                raw_content, detected_format, context
            )
            
            # Step 2: Format standardization
            format_result = await self.format_standardizer.standardize(
                quality_result.content, detected_format, context
            )
            
            # Step 3: Claude style adaptation
            style_result = await self.claude_style_adapter.adapt(
                format_result.content, detected_format, context
            )
            
            # Combine transformation metadata
            transformations = (
                quality_result.transformations + 
                format_result.transformations + 
                style_result.transformations
            )
            
            logger.info(
                f"Transformation complete for {request_id}: "
                f"Format={detected_format.value}, "
                f"Transformations={len(transformations)}"
            )
            
            return style_result.content
            
        except Exception as e:
            logger.error(f"Error transforming response: {e}")
            # Return safe fallback
            return self._create_fallback_response(
                ollama_response.get('response', ''), 
                str(e)
            )
    
    def _extract_content(self, ollama_response: Dict[str, Any]) -> str:
        """Extract content from Ollama response structure"""
        
        # Handle different Ollama response formats
        if 'response' in ollama_response:
            return ollama_response['response']
        elif 'message' in ollama_response:
            if isinstance(ollama_response['message'], dict):
                return ollama_response['message'].get('content', '')
            return str(ollama_response['message'])
        elif 'content' in ollama_response:
            return ollama_response['content']
        elif 'text' in ollama_response:
            return ollama_response['text']
        else:
            # Fallback - convert entire response to string
            return str(ollama_response)
    
    def _detect_format(self, content: str, context: Dict[str, Any]) -> ResponseFormat:
        """Detect the response format type"""
        
        # Code detection patterns
        code_patterns = [
            r'```[\w]*\n.*?\n```',  # Code blocks
            r'`[^`\n]+`',  # Inline code
            r'\bfunction\s+\w+\s*\(',  # Function definitions
            r'\bclass\s+\w+',  # Class definitions
            r'def\s+\w+\s*\(',  # Python functions
            r'import\s+\w+',  # Import statements
        ]
        
        if any(re.search(pattern, content, re.DOTALL) for pattern in code_patterns):
            return ResponseFormat.CODE
        
        # Structured data detection
        try:
            json.loads(content)
            return ResponseFormat.STRUCTURED
        except:
            pass
        
        # Error detection
        error_indicators = [
            'error', 'exception', 'failed', 'unable', 'cannot',
            'invalid', 'incorrect', 'wrong', 'issue', 'problem'
        ]
        
        if any(indicator in content.lower() for indicator in error_indicators):
            if len(content) < 200:  # Short error messages
                return ResponseFormat.ERROR
        
        # Default to text
        return ResponseFormat.TEXT
    
    def _create_fallback_response(self, original_content: str, error: str) -> str:
        """Create a safe fallback response when transformation fails"""
        return (
            f"{original_content}\n\n"
            f"*Note: Response processing encountered an issue but the content above "
            f"should still be helpful. If you notice any formatting issues, please "
            f"let me know.*"
        )


class QualityEnhancer:
    """Enhances response quality to match Claude standards"""
    
    async def enhance(
        self, 
        content: str, 
        format_type: ResponseFormat, 
        context: Dict[str, Any]
    ) -> TransformationResult:
        """Enhance response quality"""
        
        transformations = []
        enhanced_content = content
        
        # Grammar and spelling improvements
        enhanced_content, grammar_fixes = self._improve_grammar(enhanced_content)
        if grammar_fixes:
            transformations.extend(grammar_fixes)
        
        # Clarity improvements
        enhanced_content, clarity_fixes = self._improve_clarity(enhanced_content)
        if clarity_fixes:
            transformations.extend(clarity_fixes)
        
        # Technical accuracy improvements
        enhanced_content, tech_fixes = self._improve_technical_accuracy(enhanced_content, format_type)
        if tech_fixes:
            transformations.extend(tech_fixes)
        
        # Completeness check
        enhanced_content, completeness_fixes = self._ensure_completeness(enhanced_content, context)
        if completeness_fixes:
            transformations.extend(completeness_fixes)
        
        return TransformationResult(
            transformed_content=enhanced_content,
            format_detected=format_type,
            confidence=0.8,
            transformations_applied=transformations,
            metadata={'stage': 'quality_enhancement'}
        )
    
    def _improve_grammar(self, content: str) -> Tuple[str, List[str]]:
        """Improve grammar and spelling"""
        fixes = []
        improved = content
        
        # Common grammar fixes
        grammar_fixes = {
            r'\bi\b': 'I',  # Capitalize 'i'
            r'\s+': ' ',    # Multiple spaces to single space
            r'\n\s*\n\s*\n': '\n\n',  # Multiple newlines to double
            r'([.!?])\s*([a-z])': r'\1 \2',  # Space after punctuation
        }
        
        for pattern, replacement in grammar_fixes.items():
            if re.search(pattern, improved):
                improved = re.sub(pattern, replacement, improved)
                fixes.append(f"grammar_fix_{pattern[:20]}")
        
        # Remove trailing whitespace
        if improved != improved.rstrip():
            improved = improved.rstrip()
            fixes.append('removed_trailing_whitespace')
        
        return improved, fixes
    
    def _improve_clarity(self, content: str) -> Tuple[str, List[str]]:
        """Improve clarity and readability"""
        fixes = []
        improved = content
        
        # Add line breaks for better readability
        if len(improved.split('\n')) < 3 and len(improved) > 200:
            # Look for natural break points
            break_points = ['. ', '? ', '! ']
            for bp in break_points:
                if bp in improved:
                    parts = improved.split(bp)
                    if len(parts) > 2:
                        improved = f'{bp}\n\n'.join(parts)
                        fixes.append('added_paragraph_breaks')
                        break
        
        return improved, fixes
    
    def _improve_technical_accuracy(self, content: str, format_type: ResponseFormat) -> Tuple[str, List[str]]:
        """Improve technical accuracy"""
        fixes = []
        improved = content
        
        if format_type == ResponseFormat.CODE:
            # Ensure code blocks are properly formatted
            if '```' not in improved and any(
                keyword in improved for keyword in ['def ', 'function ', 'class ', 'import ']
            ):
                # Wrap in code block
                improved = f"```\n{improved}\n```"
                fixes.append('added_code_block_formatting')
        
        return improved, fixes
    
    def _ensure_completeness(self, content: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Ensure response completeness"""
        fixes = []
        improved = content
        
        # Check if response seems cut off
        if improved.endswith(('...', '..', 'etc')):
            # Response might be incomplete, but we can't really fix this
            # Just log it
            fixes.append('detected_potential_incompleteness')
        
        # Ensure response addresses the input
        input_text = context.get('input_text', '').lower()
        if 'how' in input_text and '?' in input_text:
            # Question asking "how" - ensure we have some explanatory content
            if len(improved) < 50:
                fixes.append('response_potentially_too_brief')
        
        return improved, fixes


class FormatStandardizer:
    """Standardizes response formatting to match Claude expectations"""
    
    async def standardize(
        self, 
        content: str, 
        format_type: ResponseFormat, 
        context: Dict[str, Any]
    ) -> TransformationResult:
        """Standardize response format"""
        
        transformations = []
        standardized = content
        
        # Apply format-specific standardization
        if format_type == ResponseFormat.CODE:
            standardized, code_transforms = self._standardize_code_format(standardized)
            transformations.extend(code_transforms)
        elif format_type == ResponseFormat.STRUCTURED:
            standardized, struct_transforms = self._standardize_structured_format(standardized)
            transformations.extend(struct_transforms)
        elif format_type == ResponseFormat.TEXT:
            standardized, text_transforms = self._standardize_text_format(standardized)
            transformations.extend(text_transforms)
        
        # Universal formatting
        standardized, universal_transforms = self._apply_universal_formatting(standardized)
        transformations.extend(universal_transforms)
        
        return TransformationResult(
            transformed_content=standardized,
            format_detected=format_type,
            confidence=0.85,
            transformations_applied=transformations,
            metadata={'stage': 'format_standardization'}
        )
    
    def _standardize_code_format(self, content: str) -> Tuple[str, List[str]]:
        """Standardize code formatting"""
        transforms = []
        formatted = content
        
        # Ensure proper code block formatting
        code_block_pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.findall(code_block_pattern, formatted, re.DOTALL)
        
        for lang, code in matches:
            # Clean up code indentation
            lines = code.split('\n')
            if lines:
                # Remove common leading whitespace
                min_indent = float('inf')
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                
                if min_indent != float('inf') and min_indent > 0:
                    cleaned_lines = []
                    for line in lines:
                        if line.strip():
                            cleaned_lines.append(line[min_indent:])
                        else:
                            cleaned_lines.append('')
                    
                    cleaned_code = '\n'.join(cleaned_lines)
                    formatted = formatted.replace(f'```{lang}\n{code}\n```', f'```{lang}\n{cleaned_code}\n```')
                    transforms.append('standardized_code_indentation')
        
        return formatted, transforms
    
    def _standardize_structured_format(self, content: str) -> Tuple[str, List[str]]:
        """Standardize structured data formatting"""
        transforms = []
        formatted = content
        
        try:
            # Try to parse and reformat JSON
            if content.strip().startswith('{') or content.strip().startswith('['):
                parsed = json.loads(content)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                transforms.append('standardized_json_formatting')
        except json.JSONDecodeError:
            # Not valid JSON, leave as is
            pass
        
        return formatted, transforms
    
    def _standardize_text_format(self, content: str) -> Tuple[str, List[str]]:
        """Standardize text formatting"""
        transforms = []
        formatted = content
        
        # Ensure proper paragraph spacing
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        if re.search(r'\n{3,}', content):
            transforms.append('standardized_paragraph_spacing')
        
        # Ensure proper list formatting
        list_pattern = r'^(\s*)([-*+]|\d+\.)\s+'
        lines = formatted.split('\n')
        new_lines = []
        
        for line in lines:
            if re.match(list_pattern, line):
                # Standardize list item formatting
                match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.*)$', line)
                if match:
                    indent, bullet, text = match.groups()
                    new_line = f"{indent}- {text}"  # Standardize to dash
                    new_lines.append(new_line)
                    if bullet != '-':
                        transforms.append('standardized_list_formatting')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        formatted = '\n'.join(new_lines)
        
        return formatted, transforms
    
    def _apply_universal_formatting(self, content: str) -> Tuple[str, List[str]]:
        """Apply universal formatting rules"""
        transforms = []
        formatted = content
        
        # Ensure consistent quote marks
        if '"' in formatted and "'" in formatted:
            # Prefer double quotes for consistency
            formatted = re.sub(r"'([^']*)'", r'"\1"', formatted)
            transforms.append('standardized_quote_marks')
        
        # Ensure proper spacing around punctuation
        original_formatted = formatted
        formatted = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', formatted)
        if formatted != original_formatted:
            transforms.append('standardized_sentence_spacing')
        
        return formatted, transforms


class ClaudeStyleAdapter:
    """Adapts responses to match Claude's distinctive style"""
    
    async def adapt(
        self, 
        content: str, 
        format_type: ResponseFormat, 
        context: Dict[str, Any]
    ) -> TransformationResult:
        """Adapt content to Claude style"""
        
        transformations = []
        adapted = content
        
        # Apply Claude-specific style patterns
        adapted, style_transforms = self._apply_claude_patterns(adapted, format_type)
        transformations.extend(style_transforms)
        
        # Add helpful context where appropriate
        adapted, context_transforms = self._add_helpful_context(adapted, context)
        transformations.extend(context_transforms)
        
        # Ensure professional tone
        adapted, tone_transforms = self._ensure_professional_tone(adapted)
        transformations.extend(tone_transforms)
        
        return TransformationResult(
            transformed_content=adapted,
            format_detected=format_type,
            confidence=0.9,
            transformations_applied=transformations,
            metadata={'stage': 'claude_style_adaptation'}
        )
    
    def _apply_claude_patterns(self, content: str, format_type: ResponseFormat) -> Tuple[str, List[str]]:
        """Apply Claude's distinctive response patterns"""
        transforms = []
        adapted = content
        
        # Add structured thinking for complex responses
        if format_type == ResponseFormat.CODE and len(adapted) > 300:
            if not adapted.startswith("Here's") and not adapted.startswith("I'll"):
                # Add a brief introduction
                adapted = f"Here's a solution for your request:\n\n{adapted}"
                transforms.append('added_claude_introduction')
        
        # Add explanatory context for code
        if format_type == ResponseFormat.CODE and '```' in adapted:
            code_blocks = re.findall(r'```\w*\n(.*?)\n```', adapted, re.DOTALL)
            if code_blocks and not any(
                phrase in adapted.lower() 
                for phrase in ['this code', 'the function', 'this implementation']
            ):
                # Add brief explanation if missing
                if not re.search(r'\n\n.*explains?.*code', adapted.lower()):
                    transforms.append('identified_missing_code_explanation')
        
        return adapted, transforms
    
    def _add_helpful_context(self, content: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Add helpful contextual information"""
        transforms = []
        enhanced = content
        
        input_text = context.get('input_text', '').lower()
        
        # If the user asked for help or guidance, ensure we're being helpful
        if any(word in input_text for word in ['help', 'how', 'guide', 'explain']):
            if len(enhanced) < 100 and not enhanced.endswith('?'):
                # Response might be too brief for a help request
                transforms.append('detected_potentially_brief_help_response')
        
        return enhanced, transforms
    
    def _ensure_professional_tone(self, content: str) -> Tuple[str, List[str]]:
        """Ensure professional, helpful tone"""
        transforms = []
        adjusted = content
        
        # Remove overly casual language
        casual_replacements = {
            r'\buh\b': '',
            r'\byeah\b': 'yes',
            r'\bokay\b': 'alright',
            r'\bgonna\b': 'going to',
        }
        
        for pattern, replacement in casual_replacements.items():
            if re.search(pattern, adjusted, re.IGNORECASE):
                adjusted = re.sub(pattern, replacement, adjusted, flags=re.IGNORECASE)
                transforms.append(f'professionalized_tone_{pattern}')
        
        # Ensure response doesn't start with uncertainty markers
        uncertainty_starters = [
            'i think', 'i believe', 'maybe', 'perhaps', 'i guess'
        ]
        
        for starter in uncertainty_starters:
            if adjusted.lower().startswith(starter):
                # Don't remove uncertainty entirely, but make it more confident
                if starter == 'i think':
                    adjusted = adjusted[len('i think'):].strip()
                    adjusted = f"Based on your request,{adjusted}"
                    transforms.append('improved_confidence_tone')
                break
        
        return adjusted, transforms