"""
QUMO - Quantum Unicode Machine Optimizer
A next-generation code optimization framework combining quantum patterns and Unicode optimization.
"""

__version__ = "0.1.0"

from typing import Final

# Core Unicode optimization patterns
PATTERNS: Final[dict[str, str]] = {
    "self": "ꜱ",
    "path": "ᴘ",
    "window": "ᴡ",
    "list": "ʟ",
    "quantum": "ψ",
    "delta": "∆",
    "sum": "∑",
    "product": "∏",
    "integral": "∫",
    "partial": "∂",
    "gradient": "∇",
    "infinity": "∞",
}

# Quantum operation symbols
QUANTUM_OPS: Final[dict[str, str]] = {
    "bra": "⟨",
    "ket": "⟩",
    "tensor": "⊗",
    "direct_sum": "⊕",
    "conjugate": "†",
    "expectation": "⟨·⟩",
}

# Flow control patterns
FLOW_PATTERNS: Final[dict[str, str]] = {
    "implies": "⇒",
    "maps_to": "↦",
    "compose": "∘",
    "forall": "∀",
    "exists": "∃",
    "not_exists": "∄",
    "element_of": "∈",
    "subset": "⊂",
    "superset": "⊃",
    "intersection": "∩",
    "union": "∪",
} 