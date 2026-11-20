import re
import unicodedata

_TERMS = re.compile(r"[\u3400-\u9fff]+|(?:--)?/?[a-z0-9_]+(?:[./:_-][a-z0-9_]+)*")
_PIECES = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are for in of the to what which how with 如何 怎么 什么 是否 可以 哪些 多少".split())


def _matches(text: str):
    return _TERMS.findall(unicodedata.normalize("NFKC", text).casefold())


def _is_technical(word: str) -> bool:
    return any(c in word for c in "/._:-")


def _aliases(word: str):
    yield word
    if "/" in word:
        basename = word.rsplit("/", 1)[-1]
        if _is_technical(basename):
            yield basename


def technical_terms(text: str) -> frozenset[str]:
    return frozenset(alias for word in _matches(text) if _is_technical(word) for alias in _aliases(word))


def terms(text: str, *, query: bool = False):
    for word in _matches(text):
        if ord(word[0]) >= 0x3400:
            sizes = (2, 3) if query and len(word) > 1 else (1, 2, 3)
            candidates = (word[i:i+n] for n in sizes for i in range(len(word)-n+1))
        elif _is_technical(word):
            candidates = (*_aliases(word), *_PIECES.findall(word))
        else:
            candidates = (word,)
        for token in candidates:
            if token not in _STOP:
                # FTS sees one safe token even when the identifier contains punctuation.
                yield "x" + token.encode("utf-8").hex()


def match_expression(query: str) -> str:
    exact = ("x" + word.encode("utf-8").hex() for word in sorted(technical_terms(query)))
    words = list(dict.fromkeys([*exact, *terms(query, query=True)]))[:96]
    return " OR ".join(words)
