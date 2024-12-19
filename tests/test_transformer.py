"""Tests for the QUMO transformer module."""

import ast
import pytest
from qumo.transformer import QuantumTransformer, FlowOptimizer, optimize_code

def test_name_optimization():
    """Test basic name optimization patterns."""
    source = """
def test_function(self):
    path = "/tmp"
    window = None
    return path
"""
    optimized = optimize_code(source)
    assert "ꜱ" in optimized  # self -> ꜱ
    assert "ᴘ" in optimized  # path -> ᴘ
    assert "ᴡ" in optimized  # window -> ᴡ

def test_quantum_operations():
    """Test quantum operation transformations."""
    source = """
def quantum_calc():
    return expectation_value(psi)
"""
    optimized = optimize_code(source)
    assert "⟨" in optimized  # Contains bra
    assert "⟩" in optimized  # Contains ket

def test_flow_optimization():
    """Test flow control optimization."""
    source = """
def flow_test():
    for i in range(10):
        if i > 5:
            yield i
"""
    flow_optimizer = FlowOptimizer()
    ast_tree = ast.parse(source)
    optimized_ast = flow_optimizer.optimize_flow(ast_tree)
    assert isinstance(optimized_ast, ast.AST)
    
def test_invalid_code():
    """Test handling of invalid code."""
    with pytest.raises(SyntaxError):
        optimize_code("this is not valid python") 