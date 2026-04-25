"""
Tests for PIIDetector.

Verifies:
- PII detection (credit cards, SSNs, emails, phones)
- PII sanitization/redaction
- Clean text handling
- Dictionary sanitization
"""

import pytest

from ai_platform.policy import PIIDetector, PIIPattern


@pytest.fixture
def detector():
    """Create a PIIDetector with default patterns."""
    return PIIDetector()


class TestPIIDetection:
    """Test PII detection capabilities."""
    
    def test_detect_credit_card(self, detector):
        """Test detecting credit card numbers."""
        text = "My credit card is 4532-1234-5678-9010"
        assert detector.contains_pii(text) is True
        
        # Also test without dashes
        text2 = "Card number: 4532123456789010"
        assert detector.contains_pii(text2) is True
    
    def test_detect_ssn(self, detector):
        """Test detecting Social Security Numbers."""
        text = "My SSN is 123-45-6789"
        assert detector.contains_pii(text) is True
    
    def test_detect_email(self, detector):
        """Test detecting email addresses."""
        text = "Contact me at john.doe@example.com"
        assert detector.contains_pii(text) is True
        
        text2 = "Email: user+test@domain.co.uk"
        assert detector.contains_pii(text2) is True
    
    def test_detect_phone(self, detector):
        """Test detecting phone numbers."""
        text = "Call me at (555) 123-4567"
        assert detector.contains_pii(text) is True
        
        # Various formats
        assert detector.contains_pii("Phone: 555-123-4567") is True
        assert detector.contains_pii("Mobile: 5551234567") is True
        assert detector.contains_pii("+1-555-123-4567") is True
    
    def test_clean_text_no_pii(self, detector):
        """Test that clean text is not flagged."""
        text = "This is a normal message without any PII"
        assert detector.contains_pii(text) is False
        
        text2 = "Order number 12345 shipped today"
        assert detector.contains_pii(text2) is False
    
    def test_multiple_pii_types(self, detector):
        """Test detecting multiple PII types in one text."""
        text = "Contact John at john@example.com or call 555-1234. SSN: 123-45-6789"
        assert detector.contains_pii(text) is True


class TestPIISanitization:
    """Test PII sanitization/redaction."""
    
    def test_sanitize_credit_card(self, detector):
        """Test sanitizing credit card numbers."""
        text = "My card is 4532-1234-5678-9010"
        sanitized = detector.sanitize(text)
        
        assert "4532-1234-5678-9010" not in sanitized
        assert "[REDACTED_CC]" in sanitized
    
    def test_sanitize_ssn(self, detector):
        """Test sanitizing SSN."""
        text = "SSN: 123-45-6789"
        sanitized = detector.sanitize(text)
        
        assert "123-45-6789" not in sanitized
        assert "[REDACTED_SSN]" in sanitized
    
    def test_sanitize_email(self, detector):
        """Test sanitizing email addresses."""
        text = "Email me at user@domain.com"
        sanitized = detector.sanitize(text)
        
        assert "user@domain.com" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
    
    def test_sanitize_phone(self, detector):
        """Test sanitizing phone numbers."""
        text = "Call (555) 123-4567"
        sanitized = detector.sanitize(text)
        
        assert "(555) 123-4567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized
    
    def test_sanitize_clean_text(self, detector):
        """Test that clean text is unchanged."""
        text = "This is a clean message"
        sanitized = detector.sanitize(text)
        
        assert sanitized == text
    
    def test_sanitize_multiple_pii(self, detector):
        """Test sanitizing text with multiple PII instances."""
        text = "Contact john@example.com or call 555-123-4567. Card: 4532123456789010"
        sanitized = detector.sanitize(text)
        
        # All PII should be redacted
        assert "john@example.com" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "4532123456789010" not in sanitized
        
        # Redaction markers should be present
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_CC]" in sanitized
    
    def test_sanitize_empty_string(self, detector):
        """Test sanitizing empty string."""
        assert detector.sanitize("") == ""
        assert detector.sanitize(None) is None


class TestPIIDetect:
    """Test detailed PII detection with extraction."""
    
    def test_detect_extracts_matches(self, detector):
        """Test that detect method extracts actual PII values."""
        text = "Email: user@example.com, Phone: 555-123-4567"
        matches = detector.detect(text)
        
        assert "email" in matches
        assert "phone" in matches
        assert "user@example.com" in matches["email"]
    
    def test_detect_empty_on_clean_text(self, detector):
        """Test that detect returns empty dict for clean text."""
        text = "No PII here"
        matches = detector.detect(text)
        
        assert matches == {}


class TestDictionarySanitization:
    """Test sanitizing dictionaries."""
    
    def test_sanitize_dict_simple(self, detector):
        """Test sanitizing a simple dictionary."""
        data = {
            "name": "John",
            "email": "john@example.com",
            "message": "Clean text"
        }
        
        sanitized = detector.sanitize_dict(data)
        
        assert sanitized["name"] == "John"
        assert sanitized["email"] == "[REDACTED_EMAIL]"
        assert sanitized["message"] == "Clean text"
    
    def test_sanitize_dict_nested(self, detector):
        """Test sanitizing nested dictionaries."""
        data = {
            "user": {
                "name": "John",
                "contact": {
                    "email": "john@example.com",
                    "phone": "555-123-4567"
                }
            }
        }
        
        sanitized = detector.sanitize_dict(data)
        
        assert sanitized["user"]["name"] == "John"
        assert "[REDACTED_EMAIL]" in sanitized["user"]["contact"]["email"]
        assert "[REDACTED_PHONE]" in sanitized["user"]["contact"]["phone"]
    
    def test_sanitize_dict_with_lists(self, detector):
        """Test sanitizing dictionaries containing lists."""
        data = {
            "emails": ["user1@example.com", "user2@example.com"],
            "messages": ["Clean message", "Contact: 555-123-4567"]
        }
        
        sanitized = detector.sanitize_dict(data)
        
        assert all("[REDACTED_EMAIL]" in email for email in sanitized["emails"])
        assert sanitized["messages"][0] == "Clean message"
        assert "[REDACTED_PHONE]" in sanitized["messages"][1]


class TestCustomPatterns:
    """Test custom PII patterns."""
    
    def test_custom_pattern(self):
        """Test adding a custom PII pattern."""
        # Create detector with custom pattern for account numbers
        custom = PIIPattern(
            name="account_number",
            regex=r"\bACCT-\d{8}\b",
            replacement="[REDACTED_ACCT]"
        )
        
        detector = PIIDetector(custom_patterns=[custom])
        
        text = "Account number: ACCT-12345678"
        assert detector.contains_pii(text) is True
        
        sanitized = detector.sanitize(text)
        assert "ACCT-12345678" not in sanitized
        assert "[REDACTED_ACCT]" in sanitized
    
    def test_from_config(self):
        """Test creating detector from policy config."""
        config = {
            "enabled": True,
            "patterns": [
                {
                    "name": "custom_id",
                    "regex": r"\bID-\d{6}\b"
                }
            ]
        }
        
        detector = PIIDetector.from_config(config)
        
        text = "Customer ID-123456"
        assert detector.contains_pii(text) is True
        
        sanitized = detector.sanitize(text)
        assert "ID-123456" not in sanitized
        assert "[REDACTED_CUSTOM_ID]" in sanitized
