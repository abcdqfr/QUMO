"""Machine-optimized name conversion utility."""
import re
from typing import Dict, Set, List, Optional, Tuple

class N:
    """KEY: Machine-optimized name conversion
    p: patterns dict
    r: reserved words set
    """

    def __init__(s):
        s.p = {'\\b(self|this)\\b': 'ꜱ', '\\b(parent|root)\\b': 'ᴘ', '\\b(window|widget)\\b': 'ᴡ', '\\b(logger|log)\\b': 'ʟ', '\\b(display|directory)\\b': 'ᴅ', '\\b(current|command)\\b': 'ᴄ', '\\b(engine|event)\\b': 'ᴇ', '\\b(manager|model)\\b': 'ᴍ', '\\b(button|box)\\b': 'ʙ', '\\b(value|view)\\b': 'ᴠ', '\\b(text|type)\\b': 'ᴛ', '\\b(frame|file)\\b': 'ꜰ', '\\b(image|item)\\b': 'ɪ', '\\b(object|option)\\b': 'ᴏ', '\\b(name|node)\\b': 'ɴ', '\\b(queue|query)\\b': 'ǫ', '\\b(result|response)\\b': 'ʀ', '\\b(status|state)\\b': 'ꜱᴛ', '\\b(update|utility)\\b': 'ᴜ', '\\b(error|exception)\\b': 'ᴇx', '\\b(yield|year)\\b': 'ʏ', '\\b(zero|zoom)\\b': 'ᴢ', '\\bupdate_window\\b': 'ᴜᴡ', '\\binfo\\b': 'ɪɴꜰ', '\\bcurrent_widget\\b': 'ᴄᴡ', '\\bWindowManager\\b': 'Wᴍ', '\\bLogger\\b': 'Lɢ'}
        s.r = set(['if', 'else', 'elif', 'while', 'for', 'in', 'is', 'not', 'and', 'or', 'True', 'False', 'None', 'try', 'except', 'finally', 'with', 'as', 'def', 'class', 'return', 'yield', 'break', 'continue', 'pass', 'raise', 'from', 'import', 'global', 'nonlocal', 'assert', 'del', 'lambda', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'min', 'max', 'sum', 'any', 'all', 'zip', 'map', 'filter', 'i', 'j', 'k', 'x', 'y', 'z', 'n', 'm', '__init__', '__str__', '__repr__', '__len__', '__call__', '__enter__', '__exit__', '__iter__', '__next__', '__getitem__', '__setitem__', '__delitem__', '__contains__'])

    def c(s, text: str) -> str:
        """Convert text to machine-optimized form"""
        parts = s._split_code(text)
        result = []
        for type_, content in parts:
            if type_ == 'code':
                for pattern, repl in s.p.items():
                    content = re.sub(pattern, repl, content)
            elif type_ == 'string':
                result.append(content)
                continue
            elif type_ == 'comment':
                result.append(content)
                continue
            result.append(content)
        return ''.join(result)

    def _split_code(s, text: str) -> List[Tuple[str, str]]:
        """Split text into code, string, and comment parts."""
        parts = []
        i = 0
        length = len(text)
        current = []
        while i < length:
            char = text[i]
            if char in ('"', "'"):
                if current:
                    parts.append(('code', ''.join(current)))
                    current = []
                quote = char
                string_content = [quote]
                i += 1
                while i < length:
                    char = text[i]
                    string_content.append(char)
                    i += 1
                    if char == quote:
                        break
                parts.append(('string', ''.join(string_content)))
                continue
            if char == '#':
                if current:
                    parts.append(('code', ''.join(current)))
                    current = []
                comment_content = ['#']
                i += 1
                while i < length and text[i] != '\n':
                    comment_content.append(text[i])
                    i += 1
                if i < length:
                    comment_content.append(text[i])
                    i += 1
                parts.append(('comment', ''.join(comment_content)))
                continue
            current.append(char)
            i += 1
        if current:
            parts.append(('code', ''.join(current)))
        return parts

    def ᴀ(s, pattern: str, repl: str) -> None:
        """Add new pattern
        pattern: regex pattern
        repl: replacement
        """
        if pattern not in s.p and repl not in s.r:
            s.p[pattern] = repl

    def ᴅ(s, pattern: str) -> None:
        """Delete pattern
        pattern: regex pattern to remove
        """
        for p in list(s.p.keys()):
            if p == pattern:
                del s.p[p]
                break
    a = ᴀ
    r = ᴅ