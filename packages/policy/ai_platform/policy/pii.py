"""
PII Detector - Privacy enforcement and data sanitization.

Detects and redacts Personally Identifiable Information (PII) using regex patterns:
- Credit card numbers
- Social Security Numbers (SSN)
- Email addresses
- Phone numbers

Example usage:
    detector = PIIDetector()
    if detector.contains_pii("My SSN is 123-45-6789"):
        sanitized = detector.sanitize("My SSN is 123-45-6789")
        # Result: "My SSN is [REDACTED_SSN]"
"""

from __future__ import annotations

import re
from typing import NamedTuple


class PIIPattern(NamedTuple):
    """Definition of a PII detection pattern."""
    name: str
    regex: str
    replacement: str


# Default PII patterns
DEFAULT_PATTERNS = [
    PIIPattern(
        name="credit_card",
        regex=r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        replacement="[REDACTED_CC]"
    ),
    PIIPattern(
        name="ssn",
        regex=r"\b\d{3}-\d{2}-\d{4}\b",
        replacement="[REDACTED_SSN]"
    ),
    PIIPattern(
        name="email",
        regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        replacement="[REDACTED_EMAIL]"
    ),
    PIIPattern(
        name="phone",
        regex=r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
        replacement="[REDACTED_PHONE]"
    ),
]


class PIIDetector:
    """
    PII detector and sanitizer.
    
    Uses regex patterns to detect and redact personally identifiable information.
    Supports custom patterns in addition to default patterns.
    
    Attributes:
        patterns: List of PII detection patterns
        compiled_patterns: Compiled regex patterns for performance
    """
    
    def __init__(self, custom_patterns: list[PIIPattern] | None = None):
        """
        Initialize the PII detector.
        
        Args:
            custom_patterns: Optional list of additional patterns to detect
        """
        self.patterns = DEFAULT_PATTERNS.copy()
        
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        # Pre-compile regex patterns for performance
        self.compiled_patterns = [
            (pattern.name, re.compile(pattern.regex), pattern.replacement)
            for pattern in self.patterns
        ]
    
    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.
        
        Args:
            text: Text to check
            
        Returns:
            True if PII is detected, False otherwise
        """
        if not text:
            return False
        
        for name, pattern, _ in self.compiled_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def sanitize(self, text: str) -> str:
        """
        Replace all PII in text with redaction markers.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text with PII replaced by [REDACTED_*] markers
        """
        if not text:
            return text
        
        sanitized = text
        
        for name, pattern, replacement in self.compiled_patterns:
            sanitized = pattern.sub(replacement, sanitized)
        
        return sanitized
    
    def detect(self, text: str) -> dict[str, list[str]]:
        """
        Detect and extract all PII matches by type.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping PII type to list of detected values
        """
        if not text:
            return {}
        
        results: dict[str, list[str]] = {}
        
        for name, pattern, _ in self.compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                results[name] = matches
        
        return results
    
    def sanitize_dict(self, data: dict) -> dict:
        """
        Recursively sanitize PII in a dictionary.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            New dictionary with PII sanitized in all string values
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize(item) if isinstance(item, str)
                    else self.sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def from_config(cls, config: dict) -> PIIDetector:
        """
        Create a PIIDetector from a policy configuration.
        
        Args:
            config: PII enforcement config dict from policy YAML
            
        Returns:
            Configured PIIDetector instance
        """
        custom_patterns = []
        
        # Extract custom patterns from config
        patterns_config = config.get("patterns", [])
        for pattern_def in patterns_config:
            if "name" in pattern_def and "regex" in pattern_def:
                custom_patterns.append(
                    PIIPattern(
                        name=pattern_def["name"],
                        regex=pattern_def["regex"],
                        replacement=f"[REDACTED_{pattern_def['name'].upper()}]"
                    )
                )
        
        return cls(custom_patterns=custom_patterns if custom_patterns else None)
