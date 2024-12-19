"""Tests for Python reserved words and operands validation."""

import ast
import pytest
from typing import Set, List, Dict
from qumo.core.machine_key import MachineOptimizationKey as MOK

class TestReservedWords:
    """Test suite for Python reserved words and operands validation."""
    
    @pytest.fixture
    def python_keywords(self) -> Set[str]:
        """Get Python keywords using ast module."""
        return set(ast.Name(id=kw, ctx=ast.Load()).__class__.__dict__['keywords'])
    
    @pytest.fixture
    def python_operators(self) -> Set[str]:
        """Get Python operators."""
        return {
            '+', '-', '*', '/', '//', '%', '**',  # Arithmetic
            '==', '!=', '<', '>', '<=', '>=',     # Comparison
            'and', 'or', 'not', 'is', 'in',       # Logical
            '&', '|', '^', '~', '<<', '>>',       # Bitwise
            '=', '+=', '-=', '*=', '/=', '//=',   # Assignment
            '%=', '**=', '&=', '|=', '^=',        # Compound assignment
            '<<=', '>>=', ':=',                   # Walrus and shifts
        }
    
    def test_all_keywords_preserved(self, python_keywords: Set[str]) -> None:
        """Test that all Python keywords are in reserved words."""
        assert python_keywords.issubset(MOK.RESERVED_WORDS), \
            f"Missing keywords: {python_keywords - MOK.RESERVED_WORDS}"
    
    def test_operator_patterns_valid(self, python_operators: Set[str]) -> None:
        """Test that operator patterns don't conflict with Python operators."""
        all_patterns: Dict[str, str] = {}
        
        # Collect all patterns
        for pattern_dict in [
            MOK.VARIABLE_PATTERNS,
            MOK.OPERATION_PATTERNS,
            MOK.METHOD_PATTERNS,
            MOK.CLASS_PATTERNS,
            MOK.QUANTUM_PATTERNS,
            MOK.UNICODE_PATTERNS,
            MOK.MATH_PATTERNS
        ]:
            for pattern_info in pattern_dict.values():
                symbol = pattern_info.get('symbol', '')
                if symbol:
                    all_patterns[symbol] = pattern_info.get('purpose', '')
        
        # Check for conflicts
        conflicts = []
        for op in python_operators:
            if op in all_patterns:
                conflicts.append(f"Pattern '{op}' conflicts with Python operator")
            for pattern in all_patterns:
                if op in pattern:
                    conflicts.append(
                        f"Pattern '{pattern}' contains Python operator '{op}'"
                    )
        
        assert not conflicts, "\n".join(conflicts)
    
    def test_unicode_safety(self) -> None:
        """Test Unicode pattern safety."""
        unsafe_combinations = []
        
        for pattern_dict in [
            MOK.VARIABLE_PATTERNS,
            MOK.QUANTUM_PATTERNS,
            MOK.UNICODE_PATTERNS,
            MOK.MATH_PATTERNS
        ]:
            for name, pattern_info in pattern_dict.items():
                symbol = pattern_info.get('symbol', '')
                if symbol:
                    # Check for combining characters
                    if any(ord(c) >= 0x300 and ord(c) <= 0x36F for c in symbol):
                        unsafe_combinations.append(
                            f"Pattern '{name}' uses combining characters"
                        )
                    
                    # Check for bidirectional text
                    if any(ord(c) >= 0x2000 and ord(c) <= 0x200F for c in symbol):
                        unsafe_combinations.append(
                            f"Pattern '{name}' uses bidirectional controls"
                        )
                    
                    # Check for homoglyphs
                    homoglyph_ranges = [
                        (0x0400, 0x04FF),  # Cyrillic
                        (0x0500, 0x052F),  # Cyrillic Supplement
                        (0x2E80, 0x2EFF),  # CJK Radicals Supplement
                    ]
                    for start, end in homoglyph_ranges:
                        if any(start <= ord(c) <= end for c in symbol):
                            unsafe_combinations.append(
                                f"Pattern '{name}' uses potential homoglyphs"
                            )
        
        assert not unsafe_combinations, "\n".join(unsafe_combinations)
    
    def test_pattern_uniqueness(self) -> None:
        """Test that patterns don't overlap or conflict."""
        all_patterns: Dict[str, List[str]] = {}
        
        # Collect all patterns and their sources
        for category, pattern_dict in [
            ("Variable", MOK.VARIABLE_PATTERNS),
            ("Operation", MOK.OPERATION_PATTERNS),
            ("Method", MOK.METHOD_PATTERNS),
            ("Class", MOK.CLASS_PATTERNS),
            ("Quantum", MOK.QUANTUM_PATTERNS),
            ("Unicode", MOK.UNICODE_PATTERNS),
            ("Math", MOK.MATH_PATTERNS)
        ]:
            for name, pattern_info in pattern_dict.items():
                symbol = pattern_info.get('symbol', '')
                if symbol:
                    if symbol not in all_patterns:
                        all_patterns[symbol] = []
                    all_patterns[symbol].append(f"{category}.{name}")
        
        # Check for duplicates
        duplicates = {
            symbol: sources
            for symbol, sources in all_patterns.items()
            if len(sources) > 1
        }
        
        assert not duplicates, \
            "Duplicate patterns found:\n" + "\n".join(
                f"'{symbol}' used in: {', '.join(sources)}"
                for symbol, sources in duplicates.items()
            )
    
    def test_fallback_compatibility(self) -> None:
        """Test that fallback representations are valid Python."""
        invalid_fallbacks = []
        
        for symbol, fallback in MOK.get_fallback_map().items():
            try:
                # Try to parse as Python code
                ast.parse(fallback)
            except SyntaxError:
                invalid_fallbacks.append(
                    f"Fallback '{fallback}' for '{symbol}' is not valid Python"
                )
        
        assert not invalid_fallbacks, "\n".join(invalid_fallbacks)
    
    def test_pattern_reversibility(self) -> None:
        """Test that all patterns can be reversed without loss."""
        test_cases = [
            ("def process_data(self):", "def ᴘᴅ(ꜱ):"),
            ("total = sum(x for x in data)", "∑ = ∑(x for x in ᴅ)"),
            ("if exists and not empty:", "if ∃ and not ∅:"),
        ]
        
        for original, optimized in test_cases:
            # Forward transformation
            transformed = MOK.optimize_code(original)
            assert transformed == optimized, \
                f"Forward transform failed: {original} -> {transformed} != {optimized}"
            
            # Reverse transformation
            reversed_code = MOK.reverse_optimization(optimized)
            assert reversed_code == original, \
                f"Reverse transform failed: {optimized} -> {reversed_code} != {original}" 