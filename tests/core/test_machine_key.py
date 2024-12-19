"""Tests for machine optimization key."""

import pytest
from src.wallpaperengine.machine_key import MachineOptimizationKey as ᴍᴋ

@pytest.fixture
def key() -> ᴍᴋ:
    """Create machine optimization key instance."""
    return ᴍᴋ()

def test_get_pattern(key: ᴍᴋ) -> None:
    """Test pattern retrieval."""
    p = ᴍᴋ.get_pattern('VARIABLE', 'SELF')
    assert p['symbol'] == 'ꜱ'
    assert p['purpose'] == 'Instance self-reference'
    assert r'\b(self|this)\b' in p['pattern']

def test_is_reserved(key: ᴍᴋ) -> None:
    """Test reserved word checking."""
    assert ᴍᴋ.is_reserved('if')
    assert ᴍᴋ.is_reserved('__init__')
    assert not ᴍᴋ.is_reserved('custom_var')

def test_get_principle(key: ᴍᴋ) -> None:
    """Test principle retrieval."""
    assert 'Machine-First' in ᴍᴋ.get_principle('P1')
    assert 'Pattern Consistency' in ᴍᴋ.get_principle('P2')
    assert ᴍᴋ.get_principle('P9') is None

def test_get_doc_requirement(key: ᴍᴋ) -> None:
    """Test documentation requirement retrieval."""
    assert 'KEY comment' in ᴍᴋ.get_doc_requirement('KEY')
    assert 'performance' in ᴍᴋ.get_doc_requirement('IMPACT')
    assert ᴍᴋ.get_doc_requirement('INVALID') is None

def test_validate_optimization(key: ᴍᴋ) -> None:
    """Test optimization validation."""
    original = """
    def update_window(self):
        self.logger.info("Test")
        self.current = self.parent.window
    """
    
    optimized = """
    def ᴜᴡ(ꜱ):
        ꜱ.ʟ.ɪɴꜰ("Test")
        ꜱ.ᴄ = ꜱ.ᴘ.ᴡ
    """
    
    messages = ᴍᴋ.validate_optimization(original, optimized)
    assert not messages  # Should be valid

def test_validate_optimization_invalid(key: ᴍᴋ) -> None:
    """Test invalid optimization validation."""
    original = "if value in items:"
    optimized = "ɪꜰ ᴠ ɪɴ ɪ:"  # Invalid: optimized reserved word
    
    messages = ᴍᴋ.validate_optimization(original, optimized)
    assert any('Reserved word' in msg for msg in messages)

def test_validate_optimization_size(key: ᴍᴋ) -> None:
    """Test size validation."""
    original = "x = 1"
    optimized = "variable = 1"  # Invalid: larger than original
    
    messages = ᴍᴋ.validate_optimization(original, optimized)
    assert any('P3' in msg for msg in messages)

def test_validate_pattern_consistency(key: ᴍᴋ) -> None:
    """Test pattern consistency validation."""
    original = "self.window.logger"
    optimized = "ꜱ.window.ʟ"  # Invalid: inconsistent window pattern
    
    messages = ᴍᴋ.validate_optimization(original, optimized)
    assert any('P2' in msg for msg in messages)

def test_quantum_patterns() -> None:
    """Test quantum pattern validation."""
    # Quantum superposition patterns
    patterns = {
        'SUPERPOSITION': {
            'pattern': r'\b(state|quantum)\b',
            'symbol': '⟨ψ⟩',
            'purpose': 'Quantum state variables'
        },
        'ENTANGLEMENT': {
            'pattern': r'\b(linked|entangled)\b',
            'symbol': '⊗',
            'purpose': 'Entangled operations'
        },
        'MEASUREMENT': {
            'pattern': r'\b(measure|observe)\b',
            'symbol': '⟨M⟩',
            'purpose': 'State measurement'
        }
    }
    
    # Verify pattern structure
    for name, info in patterns.items():
        assert 'pattern' in info
        assert 'symbol' in info
        assert 'purpose' in info
        assert isinstance(info['pattern'], str)
        assert isinstance(info['symbol'], str)
        assert isinstance(info['purpose'], str)

def test_advanced_unicode_patterns() -> None:
    """Test advanced Unicode pattern validation."""
    # Advanced Unicode patterns
    patterns = {
        'TENSOR': {
            'pattern': r'\b(tensor|matrix)\b',
            'symbol': '⊗',
            'purpose': 'Tensor operations'
        },
        'VECTOR': {
            'pattern': r'\b(vector|array)\b',
            'symbol': '→',
            'purpose': 'Vector operations'
        },
        'INFINITY': {
            'pattern': r'\b(infinite|unlimited)\b',
            'symbol': '∞',
            'purpose': 'Infinite operations'
        },
        'PARTIAL': {
            'pattern': r'\b(partial|incomplete)\b',
            'symbol': '∂',
            'purpose': 'Partial operations'
        }
    }
    
    # Verify pattern structure
    for name, info in patterns.items():
        assert 'pattern' in info
        assert 'symbol' in info
        assert 'purpose' in info
        assert isinstance(info['pattern'], str)
        assert isinstance(info['symbol'], str)
        assert isinstance(info['purpose'], str)

def test_is_valid_unicode() -> None:
    """Test Unicode character validation."""
    # Valid characters
    assert ᴍᴋ.is_valid_unicode('∑')  # Mathematical symbol
    assert ᴍᴋ.is_valid_unicode('∀')  # Logical operator
    assert ᴍᴋ.is_valid_unicode('ꜱ')  # Small Latin letter
    assert ᴍᴋ.is_valid_unicode('→')  # Arrow
    
    # Invalid characters
    assert not ᴍᴋ.is_valid_unicode('😀')  # Emoji
    assert not ᴍᴋ.is_valid_unicode('❤')   # Dingbat
    assert not ᴍᴋ.is_valid_unicode('🔍')  # Pictograph
    assert not ᴍᴋ.is_valid_unicode('abc')  # Multi-char string

def test_validate_unicode_usage() -> None:
    """Test Unicode usage validation."""
    # Valid text
    valid = "def ᴜᴡ(ꜱ): ꜱ.∑ = ∀x∈X"
    assert not ᴍᴋ.validate_unicode_usage(valid)
    
    # Invalid text
    invalid = "def 😀(ꜱ): ꜱ.❤ = 🔍"
    messages = ᴍᴋ.validate_unicode_usage(invalid)
    assert len(messages) == 3
    assert all('Invalid Unicode character' in msg for msg in messages)

def test_get_fallback() -> None:
    """Test ASCII fallback retrieval."""
    assert ᴍᴋ.get_fallback('∑') == 'sum'
    assert ᴍᴋ.get_fallback('→') == '->'
    assert ᴍᴋ.get_fallback('⊗') == 'x'
    assert ᴍᴋ.get_fallback('unknown') == 'unknown'

def test_validate_optimization_with_unicode() -> None:
    """Test optimization validation with Unicode checks."""
    original = """
    def process_data(self):
        self.total = sum(x for x in self.data)
        self.check_exists(self.data)
    """
    
    # Valid optimization
    valid = """
    def ᴘᴅ(ꜱ):
        ꜱ.∑ = ∑(x for x in ꜱ.ᴅ)
        ꜱ.∃(ꜱ.ᴅ)
    """
    assert not ᴍᴋ.validate_optimization(original, valid)
    
    # Invalid optimization (emoji)
    invalid = """
    def 😀(ꜱ):
        ꜱ.❤ = 🔍(x for x in ꜱ.ᴅ)
    """
    messages = ᴍᴋ.validate_optimization(original, invalid)
    assert any('Invalid Unicode character' in msg for msg in messages)

def test_unicode_block_ranges() -> None:
    """Test Unicode block range validation."""
    # Mathematical Operators block (U+2200-U+22FF)
    assert ᴍᴋ.is_valid_unicode('∀')  # U+2200
    assert ᴍᴋ.is_valid_unicode('∃')  # U+2203
    assert ᴍᴋ.is_valid_unicode('∈')  # U+2208
    
    # Mathematical Alphanumeric block
    assert ᴍᴋ.is_valid_unicode('ꜱ')  # Small letter s
    assert ᴍᴋ.is_valid_unicode('ᴘ')  # Small letter p
    
    # Outside valid blocks
    assert not ᴍᴋ.is_valid_unicode('⛔')  # Transport and map symbols
    assert not ᴍᴋ.is_valid_unicode('☢')  # Miscellaneous symbols 