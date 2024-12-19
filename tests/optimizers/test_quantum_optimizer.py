"""Tests for the QUMO-based wallpaper engine optimizer."""

import ast
import pytest
from src.wallpaperengine.quantum_optimizer import (
    QuantumWallpaperTransformer,
    FlowGraphOptimizer,
    optimize_wallpaper_code
)

def test_name_optimization():
    """Test quantum name optimization patterns."""
    source = """
class WallpaperEngine:
    def __init__(self):
        self.path = None
        self.window = None
        self.current = None
        self.display = None
    
    def next_wallpaper(self):
        return self.random_choice()
"""
    optimized = optimize_wallpaper_code(source)
    
    # Check core patterns
    assert "ꜱ" in optimized  # self -> ꜱ
    assert "ᴘ" in optimized  # path -> ᴘ
    assert "ᴡ" in optimized  # window -> ᴡ
    
    # Check state patterns
    assert "ᴄ" in optimized  # current -> ᴄ
    assert "ᴅ" in optimized  # display -> ᴅ
    
    # Check flow patterns
    assert "ɴ" in optimized  # next -> ɴ
    assert "ʀ" in optimized  # random -> ʀ

def test_quantum_operations():
    """Test quantum operation transformations."""
    source = """
def optimize_operations(self):
    return compose(
        tensor(op1, op2),
        direct_sum(op3, op4),
        conjugate(op5)
    )
"""
    optimized = optimize_wallpaper_code(source)
    
    # Check quantum operations
    assert "∘" in optimized  # compose -> ∘
    assert "⊗" in optimized  # tensor -> ⊗
    assert "⊕" in optimized  # direct_sum -> ⊕
    assert "†" in optimized  # conjugate -> †

def test_flow_optimization():
    """Test flow control optimization."""
    source = """
def process_wallpapers(self):
    for wallpaper in self.wallpapers:
        if wallpaper.enabled:
            yield wallpaper
        elif wallpaper.pending:
            continue
"""
    # Create flow optimizer
    optimizer = FlowGraphOptimizer()
    tree = ast.parse(source)
    
    # Analyze flow
    optimizer.analyze_flow(tree)
    
    # Verify graph construction
    assert len(optimizer.ɢ.nodes()) > 0
    assert len(optimizer.ɢ.edges()) > 0
    
    # Optimize and verify
    optimized = optimizer.optimize_flow(tree)
    assert isinstance(optimized, ast.AST)

def test_invalid_code():
    """Test handling of invalid code."""
    with pytest.raises(SyntaxError):
        optimize_wallpaper_code("this is not valid python")

def test_complex_transformation():
    """Test complex code transformation with multiple patterns."""
    source = """
class WallpaperManager:
    def __init__(self):
        self.current = None
        self.paths = []
        self.window = None
    
    def load_next(self):
        if self.current:
            self.current = compose(
                tensor(self.paths, self.window),
                direct_sum(self.load(), self.validate())
            )
        return self.current
"""
    optimized = optimize_wallpaper_code(source)
    
    # Check combined patterns
    assert all(pattern in optimized for pattern in [
        "ꜱ",  # self
        "ᴄ",  # current
        "ᴘ",  # paths
        "ᴡ",  # window
        "∘",  # compose
        "⊗",  # tensor
        "⊕",  # direct_sum
        "ʟ",  # load
        "ᴠ",  # validate
    ]) 