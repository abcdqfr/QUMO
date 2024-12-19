"""Test configuration and shared fixtures."""

import pytest
from hypothesis import settings, Verbosity, Phase, HealthCheck
import hypothesis.strategies as st
from typing import Generator, Any

# Configure Hypothesis for thorough testing
settings.register_profile(
    "quantum",
    max_examples=2000,
    deadline=1000,
    derandomize=True,
    verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

settings.load_profile("quantum")

# Custom strategies for generating Python code
@st.composite
def python_identifiers(draw: Any) -> str:
    """Generate valid Python identifiers."""
    first = draw(st.characters(whitelist_categories=('Lu', 'Ll', '_')))
    rest = draw(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', '_')), min_size=0, max_size=10))
    return first + rest

@st.composite
def python_expressions(draw: Any) -> str:
    """Generate valid Python expressions."""
    return draw(st.one_of(
        st.integers().map(str),
        st.text(min_size=1).map(lambda x: f"'{x}'"),
        python_identifiers(),
        st.lists(python_identifiers()).map(lambda x: f"[{', '.join(x)}]"),
    ))

@st.composite
def python_statements(draw: Any) -> str:
    """Generate valid Python statements."""
    var = draw(python_identifiers())
    expr = draw(python_expressions())
    return f"{var} = {expr}"

@pytest.fixture
def code_generator() -> Generator[Any, None, None]:
    """Fixture for generating test code."""
    yield {
        'identifiers': python_identifiers(),
        'expressions': python_expressions(),
        'statements': python_statements(),
    }

@pytest.fixture
def quantum_patterns() -> Generator[dict, None, None]:
    """Fixture for quantum patterns."""
    yield {
        'superposition': '⟨ψ⟩',
        'tensor': '⊗',
        'direct_sum': '⊕',
        'conjugate': '†',
        'expectation': '⟨·⟩',
    }

@pytest.fixture
def unicode_patterns() -> Generator[dict, None, None]:
    """Fixture for Unicode patterns."""
    yield {
        'self': 'ꜱ',
        'path': 'ᴘ',
        'window': 'ᴡ',
        'list': 'ʟ',
        'sum': '∑',
        'forall': '∀',
        'exists': '∃',
        'empty': '∅',
    } 