"""Text to terms, for lexical retrieval only.

Four rules, all chosen because the *same* function runs over the query and over the span.
Asymmetry between the two sides is the classic lexical-retrieval defect: it makes recall
depend on which side of the comparison a word happened to appear on.

1. **Case folding**, so ``Refund`` and ``refund`` are one term.
2. **Digit-group separators are dropped**, so ``10,000`` and ``10000`` are one term. The
   corpus writes thresholds with separators and a change request writes them without, and
   without this rule the single most important term in a threshold question misses.
3. **Function words are dropped** (:data:`STOPWORDS`). Measured before it was added: two of
   the benchmark's four negative controls returned results, on ``the``, ``is`` and ``with``
   alone - a question the corpus does not answer produced 13 spans (``docs/STATUS.md`` 32).
4. **One trailing plural ``s`` is removed** (:func:`depluralize`), so ``refund`` reaches
   ``Refunds``. Measured before it was added: the benchmark's ``known_lexical_miss``
   category recalled 20% and did not improve with depth, the signature of a vocabulary
   mismatch rather than a ranking problem (``docs/STATUS.md`` 26).

Still absent, still deliberately: **no stemmer, no lemmatizer, no synonym table, no query
expansion, no fuzzy matching**. Rule 4 is a single guarded suffix removal and not the first
step of a Porter stemmer; the difference is that it has one rule with three exceptions and
can be read in full, which is what makes it auditable. It does *not* handle ``-ies``
(``policies`` does not reach ``policy``), irregular plurals (``analyses`` does not reach
``analysis``), verb inflection or comparatives, and those are recorded as limitations rather
than fixed by adding a second rule nobody measured.

Terms are split on any non-alphanumeric character, so ``refund.approval_threshold`` and
``refund approval threshold`` yield the same three terms - which is what lets a registry key
be typed into a query. Measured, that is much less powerful than it sounds: it finds a key
only where a document happens to *spell the key out*, and the benchmark's
``assumption_key_lookup`` queries show it recovering 1 of the 5 artifacts binding
``retention.window``, because the governing policy writes "retained for 90 days".

This is **not** related to assumption keys. Coupling resolves keys through the closed
registry and an approved alias, never through text similarity (checkpoint 4.1.5). Nothing
here may ever be used to decide that two keys mean the same thing.
"""

from __future__ import annotations

import re
from typing import Final

#: Bumped when tokenization changes, because that changes every score. ``tokens-2`` added
#: :data:`STOPWORDS`, ``tokens-3`` adds :func:`depluralize`. The word list and the suffix
#: guards are both part of this version, so editing either bumps it again and invalidates
#: every recall figure measured under the old one.
TOKENIZER_VERSION: Final = "tokens-3"

#: Function words, dropped from **both** the query and the span.
#:
#: **Chosen by grammatical class, not by measuring which words caused noise**, and that is
#: the load-bearing decision rather than the contents. A list assembled from the benchmark's
#: own failures would be fitted to 35 queries over 13 documents, and the improvement it
#: produced would be a description of the fitting. This is the closed-class English function
#: vocabulary - articles, coordinators, the commonest prepositions, personal and possessive
#: pronouns, the auxiliary forms of *be*/*do*/*have*, and the wh-words - assembled without
#: reference to the corpus, so the measured delta is a result rather than an artefact.
#:
#: **Four classes of word a standard stopword list would remove are deliberately kept**,
#: because in an organizational corpus each of them changes what a sentence commits to:
#:
#: * **negation** - ``no``, ``not``, ``nor``, ``never``, ``without``. "a verbal approval is
#:   *not* sufficient" and "may be released *without* manager approval" are the opposite of
#:   the same sentences without them. Dropping negation is the best-known defect of the
#:   off-the-shelf lists and it is not repeated here.
#: * **comparatives and bounds** - ``more``, ``less``, ``above``, ``below``, ``over``,
#:   ``under``, ``least``, ``most``. "10,000 USD *or more*" is a threshold direction, and an
#:   escalation ceiling reads "anything *above* 50,000 USD".
#: * **quantifiers** - ``all``, ``any``, ``each``, ``every``, ``some``, ``none``. "*all*
#:   regions unless a regional policy states otherwise" is a scope statement; "*every*
#:   refund requires manager approval" is a threshold of one.
#: * **modals** - ``may``, ``must``, ``shall``, ``should``, ``can``, ``will``. These carry
#:   the normative force of a policy, and "must" is a term a reviewer legitimately searches
#:   for.
#:
#: There is no second, larger list behind a flag. A stopword list that can be swapped at
#: run time is a scorer that cannot be reproduced from its fingerprint.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # articles and demonstratives
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # coordinators and subordinators
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "as",
        "than",
        "because",
        # the commonest prepositions
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "to",
        "with",
        "within",
        "after",
        "before",
        "during",
        "per",
        "about",
        # personal and possessive pronouns
        "i",
        "it",
        "its",
        "we",
        "us",
        "our",
        "you",
        "your",
        "they",
        "them",
        "their",
        "he",
        "him",
        "his",
        "she",
        "her",
        # auxiliary be / do / have
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        # wh-words and here/there
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "when",
        "where",
        "why",
        "how",
        "here",
        "there",
    }
)

#: Endings that keep their ``s``. The principle is narrow and worth stating precisely: an
#: ending is guarded when a trailing ``s`` there is **never** an English plural marker.
#:
#: * ``-ss`` - no English plural ends in ``ss``: ``access``, ``address``, ``business``,
#:   ``class``, ``loss``, ``process``, and the retrieval term ``less``.
#: * ``-us`` - ``status``, ``corpus``, ``bonus``, ``consensus``, ``previous``. Latin ``-us``
#:   plurals are ``-i``, so an ``-us`` word is a singular.
#: * ``-is`` - ``analysis``, ``basis``, ``crisis``, ``axis``. The plural of ``-is`` is
#:   ``-es``, so a word ending ``-is`` is never itself a plural. Known collateral:
#:   ``apis`` and ``kpis`` are caught by it and do not reach ``api``/``kpi``. Rescuing them
#:   needs a length exception, which would then strip ``axis`` - a real protection traded
#:   for a marginal gain, so the guard stays simple and the miss is recorded.
#:
#: ``-as`` is deliberately **not** guarded, and the reason is the same principle read the
#: other way. There, both classes are real - ``alias`` and ``bias`` are singulars, but
#: ``areas`` and ``ideas`` are plurals - so a guard would buy one and sell the other. The
#: rule keeps the plural reading because it is the commoner one, and accepts ``alias ->
#: alia`` as collateral: the stem is applied to both sides of every match, so it is
#: self-consistent, and it collides with no other term.
GUARDED_ENDINGS: Final = ("ss", "us", "is")

#: The shortest stem the rule will produce. Three characters keeps the merges that matter
#: (``days -> day``, ``apis -> api``, ``sops -> sop``) while refusing the short words where
#: a trailing ``s`` is part of the word rather than a plural: ``bus``, ``gas``, ``its``,
#: ``was``, ``has``. Those are left whole, which is also what stops a two-letter fragment
#: from becoming a term.
MIN_STEM_LENGTH: Final = 3

_DIGIT_SEPARATOR = re.compile(r"(?<=\d)[,_](?=\d)")
_TERM = re.compile(r"[a-z0-9]+")


def depluralize(term: str) -> str:
    """Remove one trailing plural ``s`` from ``term``, or return it unchanged.

    **This is not a stemmer and must not become one.** One suffix, three guarded endings, a
    minimum stem length, and letters only - the whole rule fits in the four conditions
    below and can be checked by reading it. A Porter stemmer would fix more plurals and
    also conflate ``operate``/``operation``, which in this corpus are different claims; the
    trade was measured on the retrieval benchmark and only the plural half of it paid.

    ``term`` is expected already case-folded, because it is: the only caller is
    :func:`tokenize`, after folding.

    What it deliberately does not do, each recorded as a limitation rather than fixed by a
    second rule: ``-ies`` (``policies`` does not reach ``policy``), irregular plurals
    (``analyses`` does not reach ``analysis``), verb inflection (``retained`` does not reach
    ``retain``), and anything at all to a token containing a digit.
    """
    # Letters only, so no quantity is ever touched. `10,000`, `90` and `2025` reach this
    # function and leave it untouched by construction rather than by luck - a number is not
    # a word and a threshold is the term a change request most depends on.
    if not term.isalpha() or not term.endswith("s"):
        return term
    if term.endswith(GUARDED_ENDINGS):
        return term
    stem = term[:-1]
    return stem if len(stem) >= MIN_STEM_LENGTH else term


def tokenize(text: str) -> tuple[str, ...]:
    """Split ``text`` into lexical terms, in order of appearance, folded and normalized.

    Order is preserved rather than sorted, and duplicates are kept: term frequency is
    counted from this.

    Both the stopword filter and the suffix rule live **inside** this function rather than
    beside it, which is the whole reason they are safe. The module's one invariant is that
    the same function runs over the query and over the span, and a caller cannot obtain
    tokens that skipped either step - so neither rule can be applied to one side and
    forgotten on the other. The cost is that the result is no longer positionally faithful
    to the input, so a future phrase strategy needs its own positional variant, and a test
    that the two agree on which terms survive.

    **A term is tested against the stopword list twice**, before and after the suffix rule,
    and both checks are load-bearing. Without the first, ``does`` would be stemmed to
    ``doe`` and escape the list as a content term. Without the second, ``theirs`` and
    ``yours`` would stem onto ``their`` and ``your`` and enter the index as terms no query
    can ever match, since a query spelling them is filtered. So the list means "this word
    and its plural are function words", and it stays a list of singulars.
    """
    folded = _DIGIT_SEPARATOR.sub("", text.casefold())
    terms: list[str] = []
    for raw in _TERM.findall(folded):
        if raw in STOPWORDS:
            continue
        term = depluralize(raw)
        if term in STOPWORDS:
            continue
        terms.append(term)
    return tuple(terms)


def term_frequencies(tokens: tuple[str, ...]) -> dict[str, int]:
    """How often each term occurs. Plain counting, no weighting."""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts
