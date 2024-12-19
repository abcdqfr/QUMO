"""Machine optimization key module."""

from typing import Dict, Any

class MachineOptimizationKey:
    """Machine optimization key for QUMO."""
    
    @staticmethod
    def get_pattern(category: str, name: str) -> Dict[str, str]:
        """Get pattern information."""
        patterns = {
            'VARIABLE': {
                'SELF': {
                    'symbol': 'ꜱ',
                    'purpose': 'Instance self-reference',
                    'pattern': r'\b(self|this)\b'
                }
            }
        }
        return patterns.get(category, {}).get(name, {})
    
    @staticmethod
    def is_reserved(word: str) -> bool:
        """Check if word is reserved."""
        import keyword
        return keyword.iskeyword(word) or word.startswith('__')
    
    @staticmethod
    def get_principle(code: str) -> str:
        """Get optimization principle."""
        principles = {
            'P1': 'Machine-First Optimization',
            'P2': 'Pattern Consistency',
            'P3': 'Size Reduction'
        }
        return principles.get(code)
    
    @staticmethod
    def validate_optimization(original: str, optimized: str) -> list[str]:
        """Validate optimization transformation."""
        messages = []
        if len(optimized) > len(original):
            messages.append('P3: Optimization increased code size')
        return messages

class MockProject:
    """Mock project for testing."""
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.complexity_score: str = "medium"
        
    def add_file(self, name: str, content: str) -> None:
        """Add a file to the mock project."""
        self.files[name] = content
        
    def get_file(self, name: str) -> str:
        """Get file content."""
        return self.files.get(name, "")