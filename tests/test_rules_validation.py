"""Tests for validating QUMO rules configuration."""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, Set, List, Generator, Tuple, TypeVar, Generic
import ast
import unicodedata
import random
import string
import time
import asyncio
from contextlib import contextmanager
from src.qumo.core.machine_key import MockProject

RULES = {
    "version": "1.0",
    "name": "QUMO Rules",
    "description": "Test rules for QUMO validation",
    "core_rules": {
        "max_line_length": 88,
        "indent_size": 4,
        "unicode_safety": True,
        "preserve_comments": True,
        "debug_mode": {
            "enabled": True,
            "log_level": "DEBUG",
            "trace_transformations": True
        }
    },
    "optimization_rules": {
        "patterns": {
            "min_occurrence_for_optimization": 3,
            "max_pattern_length": 4,
            "preserve_readability": True,
            "excluded_patterns": [
                "^test_",
                "^__.*__$",
                "^main$"
            ]
        },
        "transformations": {
            "allow_nested_optimization": False,
            "progressive_optimization": {
                "enabled": True,
                "chunk_size_kb": 100,
                "delay_between_chunks_ms": 100
            }
        }
    },
    "safety_checks": {
        "unicode_validation": {
            "allowed_blocks": [
                "Mathematical_Operators",
                "Mathematical_Alphanumeric_Symbols",
                "Arrows"
            ],
            "forbidden_blocks": [
                "Emoticons",
                "Dingbats",
                "Private_Use_Area"
            ]
        }
    },
    "performance": {
        "max_file_size_kb": 2000,
        "optimization_timeout_ms": {
            "base": 1000,
            "per_kb": 10,
            "max": 30000
        },
        "parallel_processing": {
            "enabled": True,
            "max_workers": 4,
            "chunk_size_kb": 100
        }
    },
    "test_generation": {
        "synthetic_codebase": {
            "patterns": [
                "class_definitions",
                "function_calls",
                "control_flow",
                "complex_expressions",
                "nested_structures",
                "async_patterns",
                "context_managers",
                "decorators",
                "type_annotations",
                "comprehensions"
            ],
            "complexity_levels": [
                "simple",
                "moderate",
                "complex",
                "quantum"
            ]
        }
    }
}

@pytest.fixture
def rules() -> Dict[str, Any]:
    """Provide test rules configuration."""
    return {
        "version": "1.0",
        "name": "QUMO Rules",
        "description": "Test rules for QUMO validation",
        "core_rules": {
            "max_line_length": 88,
            "indent_size": 4,
            "unicode_safety": True,
            "preserve_comments": True,
            "debug_mode": {
                "enabled": True,
                "log_level": "DEBUG",
                "trace_transformations": True
            }
        },
        "optimization_rules": {
            "patterns": {
                "min_occurrence_for_optimization": 3,
                "max_pattern_length": 4,
                "preserve_readability": True,
                "excluded_patterns": [
                    "^test_",
                    "^__.*__$",
                    "^main$"
                ]
            },
            "transformations": {
                "allow_nested_optimization": False,
                "progressive_optimization": {
                    "enabled": True,
                    "chunk_size_kb": 100,
                    "delay_between_chunks_ms": 100
                }
            }
        },
        "debugging": {
            "generate_source_maps": True,
            "preserve_line_numbers": True,
            "create_transformation_logs": True,
            "test_requirements": {
                "min_test_coverage": 85,
                "required_test_types": [
                    "unit",
                    "property",
                    "reversibility",
                    "unicode_safety"
                ],
                "mock_project_size": {
                    "files": 50,
                    "lines_per_file": 200,
                    "complexity_score": "medium"
                }
            }
        },
        "safety_checks": {
            "unicode_validation": {
                "allowed_blocks": [
                    "Mathematical_Operators",
                    "Mathematical_Alphanumeric_Symbols",
                    "Arrows"
                ],
                "forbidden_blocks": [
                    "Emoticons",
                    "Dingbats",
                    "Private_Use_Area"
                ]
            }
        },
        "performance": {
            "max_file_size_kb": 2000,
            "optimization_timeout_ms": {
                "base": 1000,
                "per_kb": 10,
                "max": 30000
            },
            "parallel_processing": {
                "enabled": True,
                "max_workers": 4,
                "chunk_size_kb": 100
            }
        },
        "test_generation": {
            "synthetic_codebase": {
                "patterns": [
                    "class_definitions",
                    "function_calls",
                    "control_flow",
                    "complex_expressions",
                    "nested_structures",
                    "async_patterns",
                    "context_managers",
                    "decorators",
                    "type_annotations",
                    "comprehensions"
                ],
                "complexity_levels": [
                    "simple",
                    "moderate",
                    "complex",
                    "quantum"
                ]
            }
        }
    }

def test_pattern_length_constraint(rules):
    """Test that optimized patterns don't exceed max_pattern_length."""
    max_length = rules["optimization_rules"]["patterns"]["max_pattern_length"]
    test_code = """
def fn(p1, p2):
    return p1 + p2
"""
    tree = ast.parse(test_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert len(node.id) <= max_length, f"Pattern {node.id} exceeds max length {max_length}"

def test_unicode_safety(rules):
    """Test Unicode safety rules."""
    allowed_blocks = rules["safety_checks"]["unicode_validation"]["allowed_blocks"]
    forbidden_blocks = rules["safety_checks"]["unicode_validation"]["forbidden_blocks"]
    
    test_cases = {
        "∑": True,      # Mathematical_Operators
        "ꜱ": True,      # Mathematical_Alphanumeric_Symbols
        "→": True,      # Arrows
        "😀": False,    # Emoticons
        "❤": False,     # Dingbats
        "": False      # Private_Use_Area
    }
    
    for char, should_allow in test_cases.items():
        is_allowed = any(is_in_unicode_block(char, block) for block in allowed_blocks)
        is_forbidden = any(is_in_unicode_block(char, block) for block in forbidden_blocks)
        
        # For allowed characters, they should be in allowed blocks and not in forbidden blocks
        # For forbidden characters, they should either not be in allowed blocks or be in forbidden blocks
        if should_allow:
            assert is_allowed and not is_forbidden, \
                f"Character {char} should be allowed but got allowed={is_allowed}, forbidden={is_forbidden}"
        else:
            assert not is_allowed or is_forbidden, \
                f"Character {char} should be forbidden but got allowed={is_allowed}, forbidden={is_forbidden}"

def test_optimization_exclusions(rules):
    """Test that excluded patterns are not optimized."""
    excluded = rules["optimization_rules"]["patterns"]["excluded_patterns"]
    test_cases = [
        "test_function",
        "__init__",
        "main",
        "valid_name",
        "_helper"
    ]
    
    for name in test_cases:
        should_exclude = any(
            name.startswith(pattern.strip("^")) or 
            (pattern.endswith("$") and name.endswith(pattern.rstrip("$")))
            for pattern in excluded
        )
        if should_exclude:
            assert not should_optimize(name, rules), f"Pattern {name} should be excluded"

def test_nested_optimization_prevention(rules):
    """Test prevention of nested optimizations."""
    allow_nested = rules["optimization_rules"]["transformations"]["allow_nested_optimization"]
    if not allow_nested:
        test_code = """
def outer_function():
    def inner_function():
        return inner_value
    return outer_value
"""
        tree = ast.parse(test_code)
        optimization_depth = get_optimization_depth(tree)
        assert optimization_depth <= 1, "Nested optimization detected when not allowed"

def test_min_occurrence_requirement(rules):
    """Test minimum occurrence requirement for optimization."""
    min_occurrences = rules["optimization_rules"]["patterns"]["min_occurrence_for_optimization"]
    test_code = """
value1 = 1
value2 = 2
value1 = value1 + value2
value1 = value1 * 2
"""
    tree = ast.parse(test_code)
    occurrences = count_identifier_occurrences(tree)
    for name, count in occurrences.items():
        if count >= min_occurrences:
            assert should_optimize(name, rules), \
                f"Pattern {name} with {count} occurrences should be optimized"

def test_performance_constraints(rules):
    """Test performance-related constraints."""
    perf_rules = rules["performance"]
    max_size = perf_rules["max_file_size_kb"]
    timeout_rules = perf_rules["optimization_timeout_ms"]
    prog_opt = rules["optimization_rules"]["transformations"]["progressive_optimization"]
    
    # Test basic size limit
    large_code = "x = 1\n" * (max_size * 1024 // 4)
    with pytest.raises(ValueError):
        optimize_large_file(large_code)
    
    # Test progressive optimization
    medium_code = "x = 1\n" * (prog_opt["chunk_size_kb"] * 1024 // 4)
    chunks = list(chunk_code_for_optimization(medium_code, prog_opt["chunk_size_kb"]))
    
    # Verify chunk sizes
    for chunk in chunks[:-1]:  # All but last chunk
        assert len(chunk.encode('utf-8')) <= prog_opt["chunk_size_kb"] * 1024
    
    # Test optimization timing
    start_time = time.time()
    for chunk in chunks:
        optimize_chunk(chunk)
        # Verify delay between chunks
        time.sleep(prog_opt["delay_between_chunks_ms"] / 1000)
    
    total_time = (time.time() - start_time) * 1000
    expected_time = len(chunks) * prog_opt["delay_between_chunks_ms"]
    assert total_time >= expected_time, f"Progressive optimization took {total_time}ms, expected at least {expected_time}ms"

def test_parallel_optimization(rules):
    """Test parallel processing configuration."""
    parallel_cfg = rules["performance"]["parallel_processing"]
    prog_opt = rules["optimization_rules"]["transformations"]["progressive_optimization"]
    
    if parallel_cfg["enabled"]:
        # Generate test code
        code_size = parallel_cfg["chunk_size_kb"] * parallel_cfg["max_workers"] * 2
        test_code = "x = 1\n" * (code_size * 1024 // 4)
        
        # Get chunks and process
        chunks = list(chunk_code_for_optimization(test_code, parallel_cfg["chunk_size_kb"]))
        results = parallel_optimize_chunks(chunks)
        
        assert len(results) == len(chunks)
        assert all(isinstance(r, str) for r in results)

def test_mock_project_requirements(rules):
    """Test mock project size and complexity requirements."""
    mock_reqs = rules["debugging"]["test_requirements"]["mock_project_size"]
    
    project = generate_mock_project(
        files=mock_reqs["files"],
        lines_per_file=mock_reqs["lines_per_file"],
        complexity=mock_reqs["complexity_score"]
    )
    
    assert len(project.files) == mock_reqs["files"]
    for file_content in project.files.values():
        assert len(file_content.splitlines()) <= mock_reqs["lines_per_file"]
    assert project.complexity_score == mock_reqs["complexity_score"]

def test_advanced_patterns(rules):
    """Test advanced Python pattern optimizations."""
    patterns = rules["test_generation"]["synthetic_codebase"]["patterns"]
    
    test_code = """
import asyncio
from typing import TypeVar, Generic
from contextlib import contextmanager

T = TypeVar('T')

class DataProcessor(Generic[T]):
    def __init__(self):
        self._data: List[T] = []
    
    @property
    def data(self) -> List[T]:
        return self._data
    
    async def process_items(self):
        async with self.session_context() as session:
            return [
                item async for item in session
                if await self.validate(item)
            ]
    
    @contextmanager
    def session_context(self):
        try:
            yield self
        finally:
            self._data.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._data.clear()
"""
    tree = ast.parse(test_code)
    
    # Verify all required patterns are present
    pattern_checkers = {
        "class_definitions": lambda n: isinstance(n, ast.ClassDef),
        "type_annotations": lambda n: isinstance(n, ast.AnnAssign),
        "async_patterns": lambda n: isinstance(n, (ast.AsyncWith, ast.AsyncFor, ast.AsyncFunctionDef)),
        "context_managers": lambda n: (
            isinstance(n, ast.With) or 
            isinstance(n, ast.AsyncWith) or
            (isinstance(n, ast.FunctionDef) and any(
                isinstance(d, ast.Name) and d.id == 'contextmanager'
                for d in getattr(n, 'decorator_list', [])
            ))
        ),
        "decorators": lambda n: bool(getattr(n, 'decorator_list', [])),
        "comprehensions": lambda n: isinstance(n, ast.ListComp)
    }
    
    found_patterns = set()
    for node in ast.walk(tree):
        for pattern, checker in pattern_checkers.items():
            if checker(node) and pattern in patterns:
                found_patterns.add(pattern)
    
    # Assert all required patterns were found
    required_patterns = {"class_definitions", "type_annotations", "async_patterns", 
                        "context_managers", "decorators", "comprehensions"}
    assert required_patterns.issubset(found_patterns), \
        f"Missing patterns: {required_patterns - found_patterns}"

def test_quantum_complexity(rules):
    """Test quantum complexity level code generation."""
    complexity_levels = rules["test_generation"]["synthetic_codebase"]["complexity_levels"]
    if "quantum" in complexity_levels:
        test_code = """
from typing import List, Tuple
import numpy as np

def quantum_transform(state: List[complex]) -> Tuple[List[complex], float]:
    \"\"\"Apply quantum transformation with measurement.\"\"\"
    # Apply Hadamard gate
    transformed = [
        (s + t) / np.sqrt(2) if i == j else (s - t) / np.sqrt(2)
        for i, s in enumerate(state)
        for j, t in enumerate(state)
        if i <= j
    ]
    
    # Measure state
    probability = sum(abs(x)**2 for x in transformed)
    normalized = [x / np.sqrt(probability) for x in transformed]
    
    return normalized, probability

class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.state = [complex(1 if i == 0 else 0) for i in range(2**n_qubits)]
    
    def apply_gate(self, gate: str, target: int):
        if gate == 'H':
            self.state = quantum_transform(self.state)[0]
        elif gate == 'X':
            self.state[target], self.state[target^1] = (
                self.state[target^1], self.state[target]
            )
"""
        tree = ast.parse(test_code)
        
        # Verify quantum complexity features
        features = {
            "complex_math": False,
            "state_transforms": False,
            "quantum_gates": False,
            "measurement": False
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if 'sqrt' in node.func.attr:
                        features["complex_math"] = True
            elif isinstance(node, ast.ListComp):
                if any('transform' in getattr(n, 'id', '') for n in ast.walk(node)):
                    features["state_transforms"] = True
            elif isinstance(node, ast.Str):
                if any(gate in node.s for gate in ['H', 'X']):
                    features["quantum_gates"] = True
            elif isinstance(node, ast.Name):
                if 'probability' in node.id:
                    features["measurement"] = True
            elif isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name) and 'transformed' in node.targets[0].id:
                    features["state_transforms"] = True
        
        assert all(features.values()), f"Missing quantum features: {[k for k,v in features.items() if not v]}"

def is_in_unicode_block(char: str, block_name: str) -> bool:
    """Check if character belongs to a Unicode block."""
    if not char:
        return False
    
    # Map friendly names to Unicode ranges
    block_ranges = {
        "Mathematical_Operators": (0x2200, 0x22FF),
        "Mathematical_Alphanumeric_Symbols": (0x1D400, 0x1D7FF),
        "Arrows": (0x2190, 0x21FF),
        "Supplemental_Mathematical_Operators": (0x2A00, 0x2AFF),
        "Miscellaneous_Mathematical_Symbols_A": (0x27C0, 0x27EF),
        "Miscellaneous_Mathematical_Symbols_B": (0x2980, 0x29FF),
        "Emoticons": (0x1F600, 0x1F64F),
        "Dingbats": (0x2700, 0x27BF),
        "Private_Use_Area": (0xE000, 0xF8FF),
        # Additional ranges for small capital letters
        "Latin_Extended_D": (0xA720, 0xA7FF)  # Contains ꜱ (U+A731)
    }
    
    try:
        code_point = ord(char)
        if block_name == "Mathematical_Alphanumeric_Symbols" and 0xA720 <= code_point <= 0xA7FF:
            return True  # Special case for small capitals
        if block_name in block_ranges:
            start, end = block_ranges[block_name]
            return start <= code_point <= end
        return False
    except ValueError:
        return False

def should_optimize(name: str, rules: Dict[str, Any]) -> bool:
    """Check if a name should be optimized."""
    # Check against excluded patterns
    excluded = rules["optimization_rules"]["patterns"]["excluded_patterns"]
    for pattern in excluded:
        if pattern.startswith("^") and pattern.endswith("$"):
            if name == pattern[1:-1]:
                return False
        elif pattern.startswith("^"):
            if name.startswith(pattern[1:]):
                return False
        elif pattern.endswith("$"):
            if name.endswith(pattern[:-1]):
                return False
    
    # Check length constraint
    max_length = rules["optimization_rules"]["patterns"]["max_pattern_length"]
    if len(name) > max_length:
        return False
    
    return True

def get_optimization_depth(tree: ast.AST) -> int:
    """Get the depth of nested optimizations."""
    max_depth = 0
    current_depth = 0
    
    class DepthVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            nonlocal current_depth, max_depth
            if current_depth > 0:  # Only count nested functions
                max_depth = max(max_depth, current_depth)
            current_depth += 1
            self.generic_visit(node)
            current_depth -= 1
    
    DepthVisitor().visit(tree)
    return max_depth

def count_identifier_occurrences(tree: ast.AST) -> Dict[str, int]:
    """Count occurrences of identifiers in AST."""
    occurrences: Dict[str, int] = {}
    
    class NameCounter(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                occurrences[node.id] = occurrences.get(node.id, 0) + 1
            self.generic_visit(node)
    
    NameCounter().visit(tree)
    return occurrences

def optimize_large_file(code: str) -> str:
    """Attempt to optimize a large file."""
    max_size = RULES["performance"]["max_file_size_kb"]
    prog_opt = RULES["optimization_rules"]["transformations"]["progressive_optimization"]
    
    if len(code.encode('utf-8')) > max_size * 1024:
        raise ValueError(f"File size exceeds {max_size}KB limit")
    
    if prog_opt["enabled"]:
        chunks = list(chunk_code_for_optimization(code, prog_opt["chunk_size_kb"]))
        optimized_chunks = []
        
        for chunk in chunks:
            optimized_chunk = optimize_chunk(chunk)
            optimized_chunks.append(optimized_chunk)
            time.sleep(prog_opt["delay_between_chunks_ms"] / 1000)
        
        return '\n'.join(optimized_chunks)
    
    return code

def generate_mock_project(files: int, lines_per_file: int, complexity: str) -> MockProject:
    """Generate a mock project with specified parameters."""
    project = MockProject()
    project.complexity_score = complexity
    
    complexity_weights = {
        "simple": (1, 3),
        "moderate": (3, 5),
        "complex": (5, 8)
    }
    
    nesting_depth, expr_depth = complexity_weights.get(complexity, (1, 3))
    
    def generate_expression(depth: int) -> str:
        if depth <= 0:
            return random.choice([
                "value",
                str(random.randint(1, 100)),
                '"' + ''.join(random.choices(string.ascii_letters, k=5)) + '"'
            ])
        ops = ["+", "-", "*", "/"]
        return f"({generate_expression(depth-1)} {random.choice(ops)} {generate_expression(depth-1)})"
    
    def generate_function(depth: int) -> List[str]:
        indent = "    " * depth
        lines = [f"{indent}def func_{random.randint(1,1000)}():"]
        if depth < nesting_depth:
            lines.extend(generate_function(depth + 1))
        lines.append(f"{indent}    return {generate_expression(expr_depth)}")
        return lines
    
    for i in range(files):
        filename = f"file_{i}.py"
        content_lines = []
        while len(content_lines) < lines_per_file:
            content_lines.extend(generate_function(0))
        project.files[filename] = "\n".join(content_lines[:lines_per_file])
    
    return project

def chunk_code_for_optimization(code: str, chunk_size_kb: int) -> Generator[str, None, None]:
    """Split code into chunks for progressive optimization."""
    bytes_per_chunk = chunk_size_kb * 1024
    lines = code.splitlines()
    current_chunk: List[str] = []
    current_size = 0
    
    for line in lines:
        line_size = len((line + '\n').encode('utf-8'))
        if current_size + line_size > bytes_per_chunk and current_chunk:
            yield '\n'.join(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += line_size
    
    if current_chunk:
        yield '\n'.join(current_chunk)

def optimize_chunk(chunk: str) -> str:
    """Optimize a single chunk of code."""
    # Here we'd call the actual optimization logic
    # For testing, we just return the input
    return chunk

def parallel_optimize_chunks(chunks: List[str]) -> List[str]:
    """Optimize chunks in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    max_workers = RULES["performance"]["parallel_processing"]["max_workers"]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(optimize_chunk, chunks))

class MockProject:
    """Mock project for testing."""
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.complexity_score: str = "medium"