"""Tokenization, which decides what "matches" means for every lexical result.

The tests worth reading are the symmetry ones. A tokenizer applied differently to the query
and to the span makes recall depend on which side a word appeared on, and that failure is
invisible in a spot check because the obvious queries still work.

Next after those, the two boundary suites, for the same reason in both cases: the *contents*
of a rule matter far less than what it refuses to do.

:class:`TestWordsThatAreNotStopwords` - negation, comparatives, quantifiers and modals all
change what an organizational sentence commits to, and an off-the-shelf English stopword
list removes every one of them. Those tests are what stop a later "let us just use the
standard list" edit from silently making "is not sufficient" mean "is sufficient".

:class:`TestWordsThePluralRuleMustNotTouch` - a wrong merge is far worse than a missed one.
A missed plural costs recall on one query; conflating ``access`` with something else quietly
changes what a citation is about. So the guards carry more tests than the rule.
"""

from __future__ import annotations

import pytest

from app.retrieval.tokens import (
    STOPWORDS,
    TOKENIZER_VERSION,
    depluralize,
    term_frequencies,
    tokenize,
)


class TestTerms:
    def test_case_is_folded(self) -> None:
        assert tokenize("Refund APPROVAL Threshold") == ("refund", "approval", "threshold")

    def test_digit_group_separators_are_dropped(self) -> None:
        """``10,000`` in a policy must match ``10000`` in a change request."""
        assert tokenize("10,000 USD") == ("10000", "usd")
        assert tokenize("10_000 USD") == ("10000", "usd")

    def test_a_decimal_point_still_splits(self) -> None:
        """Only separators *between digits* are dropped, so ``1.5`` is not ``15``."""
        assert tokenize("1.5 hours") == ("1", "5", "hour")

    def test_a_dotted_registry_key_yields_its_parts(self) -> None:
        """So a reviewer can paste a key into a query and reach the prose that binds it."""
        assert tokenize("refund.approval_threshold") == ("refund", "approval", "threshold")

    def test_punctuation_is_never_a_term(self) -> None:
        assert tokenize("--- 90 days? (EU) ---") == ("90", "day", "eu")

    def test_an_empty_string_has_no_terms(self) -> None:
        assert tokenize("") == ()

    def test_a_punctuation_only_string_has_no_terms(self) -> None:
        assert tokenize("!!! ??? ...") == ()

    def test_order_and_repeats_are_preserved(self) -> None:
        """Term frequency is counted from this, so collapsing repeats would lose weight."""
        assert tokenize("policy policy retention") == ("policy", "policy", "retention")

    def test_a_singular_and_its_plural_count_as_the_same_repeat(self) -> None:
        """Which is the point of the suffix rule, seen through the frequency count."""
        assert tokenize("refund refunds") == ("refund", "refund")

    def test_relative_order_survives_a_dropped_function_word(self) -> None:
        """The surviving terms keep their order; only the positions are no longer faithful
        to the input, which is why a future phrase strategy needs its own variant."""
        assert tokenize("90 days after the account is closed") == (
            "90",
            "day",
            "account",
            "closed",
        )


class TestSymmetry:
    @pytest.mark.parametrize(
        "query,span",
        [
            ("refund.approval_threshold", "refund approval threshold"),
            ("10000", "10,000"),
            ("10,000", "10000"),
            ("RETENTION WINDOW", "retention window"),
            ("retention-window", "retention window"),
            # The stopword rule is inside `tokenize`, so it cannot be applied to one side
            # and forgotten on the other - which is the whole reason it lives there.
            ("the retention window", "retention window"),
            ("retention window", "a retention window is"),
            ("is the retention window", "retention window for the"),
        ],
    )
    def test_the_two_sides_of_a_match_tokenize_alike(self, query: str, span: str) -> None:
        assert set(tokenize(query)) == set(tokenize(span))


class TestStopwords:
    """Function words are dropped from both sides of a match (``docs/STATUS.md`` 32)."""

    def test_a_function_word_is_not_a_term(self) -> None:
        assert tokenize("the refund is in the case record") == ("refund", "case", "record")

    def test_a_query_of_nothing_but_function_words_has_no_terms(self) -> None:
        """So it matches nothing and says so, rather than ranking the corpus by "the".

        This is the negative-control behaviour the change was made for: before it, "the
        coffee machine is coming with us" returned 13 admitted spans.
        """
        assert tokenize("what is it about") == ()
        assert tokenize("the coffee machine is coming with us") == ("coffee", "machine", "coming")

    def test_the_list_is_case_folded_like_everything_else(self) -> None:
        """Folding happens first, so a capitalized function word is still dropped - and a
        list containing an uppercase entry could never match anything."""
        assert tokenize("The Refund IS Approved") == ("refund", "approved")
        assert all(word == word.casefold() for word in STOPWORDS)

    def test_no_stopword_could_survive_the_term_pattern_anyway(self) -> None:
        """Each entry must be a single term the splitter can actually produce, or it is a
        dead entry that reads as a rule and enforces nothing."""
        for word in STOPWORDS:
            assert tokenize(f"alpha {word} omega") == ("alpha", "omega"), word

    def test_the_list_stays_small(self) -> None:
        """A closed-class function vocabulary, not a corpus-fitted list. If this grows a
        lot, someone has started tuning against the benchmark."""
        assert len(STOPWORDS) < 100


class TestWordsThatAreNotStopwords:
    """The boundary of the list, which is the part that matters.

    Every word below is removed by a standard English stopword list and is deliberately
    kept, because in an organizational corpus each of them changes what a sentence commits
    to. These are the assertions that make the choice a decision rather than an accident.
    """

    @pytest.mark.parametrize("word", ["no", "not", "nor", "never", "without"])
    def test_negation_is_a_term(self, word: str) -> None:
        """ "a verbal approval is *not* sufficient" is the opposite of the same sentence
        without it, and dropping negation is the best-known defect of the standard lists."""
        assert word in tokenize(f"approval is {word} sufficient")

    @pytest.mark.parametrize(
        "word", ["more", "less", "above", "below", "over", "under", "least", "most"]
    )
    def test_a_comparative_or_a_bound_is_a_term(self, word: str) -> None:
        """ "10,000 USD or *more*" is a threshold direction, not a filler word."""
        assert word in tokenize(f"10,000 USD or {word}")

    @pytest.mark.parametrize("word", ["all", "any", "each", "every", "some", "none"])
    def test_a_quantifier_is_a_term(self, word: str) -> None:
        """ "*all* regions unless a regional policy states otherwise" is a scope statement,
        and "*every* refund requires manager approval" is a threshold of one."""
        assert word in tokenize(f"{word} refunds require approval")

    @pytest.mark.parametrize("word", ["may", "must", "shall", "should", "can", "will"])
    def test_a_modal_is_a_term(self, word: str) -> None:
        """Modals carry the normative force of a policy, and "must" is a term a reviewer
        legitimately searches for."""
        assert word in tokenize(f"a manager {word} approve the refund")

    @pytest.mark.parametrize(
        "word",
        ["refund", "refunds", "retention", "window", "days", "usd", "approval", "policy"],
    )
    def test_a_domain_term_is_never_a_stopword(self, word: str) -> None:
        assert word not in STOPWORDS


class TestPluralRule:
    """One trailing ``s``, three guarded endings, a minimum stem (``docs/STATUS.md`` 26)."""

    @pytest.mark.parametrize(
        "plural,singular",
        [
            ("refunds", "refund"),
            ("Refunds", "refund"),
            ("days", "day"),
            ("minutes", "minute"),
            ("customers", "customer"),
            ("managers", "manager"),
            ("records", "record"),
            ("items", "item"),
            ("months", "month"),
            ("regions", "region"),
            # Acronyms benefit too, and they are exactly the terms a reviewer types.
            ("sops", "sop"),
            ("slas", "sla"),
        ],
    )
    def test_a_plural_reaches_its_singular(self, plural: str, singular: str) -> None:
        assert tokenize(plural) == (singular,)
        assert tokenize(plural) == tokenize(singular)

    def test_the_target_case_of_the_whole_change(self) -> None:
        """``refund`` typed as a query must reach a policy that writes ``Refunds``.

        The recorded defect (``docs/STATUS.md`` 26) in one line, and the reason the rule
        exists at all.
        """
        query = tokenize("refund threshold")
        span = tokenize("Refunds of 10,000 USD or more require manager approval")
        assert "refund" in query
        assert "refund" in span

    def test_only_one_s_is_ever_removed(self) -> None:
        """It is a single suffix rule, not an iterative stemmer."""
        assert depluralize("addresses") == "addresse"
        assert depluralize("depluralize") == "depluralize"

    def test_the_rule_is_idempotent(self) -> None:
        """Applying it twice must equal applying it once, or the index and the query could
        disagree depending on how many times a caller normalized."""
        for word in ["refunds", "days", "access", "analysis", "bus", "apis", "addresses"]:
            assert depluralize(depluralize(word)) == depluralize(word)


class TestWordsThePluralRuleMustNotTouch:
    """The guards. Every case here is a word the rule must leave whole.

    A wrong merge is far worse than a missed one: a missed plural costs recall on one query,
    while conflating ``access`` with something else quietly changes what a citation is
    about. So the guards get more tests than the rule does.
    """

    @pytest.mark.parametrize(
        "word",
        ["access", "address", "business", "class", "process", "loss", "gross", "less"],
    )
    def test_a_double_s_singular_keeps_its_s(self, word: str) -> None:
        """No English plural ends in ``ss``, so the ``s`` there is part of the word."""
        assert depluralize(word) == word
        assert word in tokenize(f"the {word} record")

    @pytest.mark.parametrize("word", ["status", "corpus", "bonus", "consensus", "previous"])
    def test_a_latin_us_singular_keeps_its_s(self, word: str) -> None:
        assert depluralize(word) == word

    @pytest.mark.parametrize("word", ["analysis", "basis", "crisis", "axis", "thesis"])
    def test_an_is_singular_keeps_its_s(self, word: str) -> None:
        """The plural of ``-is`` is ``-es``, so a word ending ``-is`` is never a plural."""
        assert depluralize(word) == word

    @pytest.mark.parametrize("word", ["bus", "gas", "its", "was", "has", "yes"])
    def test_a_short_word_keeps_its_s(self, word: str) -> None:
        """``bus`` is not the plural of ``bu``. The minimum stem length is what stops a
        two-letter fragment from becoming a term."""
        assert depluralize(word) == word

    @pytest.mark.parametrize("word", ["access", "analysis", "status", "class", "basis"])
    def test_a_guarded_word_does_not_merge_with_its_own_es_plural(self, word: str) -> None:
        """The guard is one-directional and that is accepted: ``access`` and ``accesses``
        remain two terms. A missed merge costs recall on a query nobody asked; a wrong
        merge would change what a result is about."""
        plural = f"{word}es"
        assert depluralize(word) != depluralize(plural)

    @pytest.mark.parametrize("word", ["not", "no", "nor", "never", "without"])
    def test_negation_is_untouched(self, word: str) -> None:
        """None of these end in ``s``, so the rule cannot reach them - asserted anyway,
        because negation surviving tokenization is the property that matters, not the
        mechanism that happens to deliver it today."""
        assert depluralize(word) == word
        assert word in tokenize(f"approval is {word} sufficient")

    @pytest.mark.parametrize(
        "word", ["more", "less", "above", "below", "over", "under", "least", "most"]
    )
    def test_a_comparative_or_bound_is_untouched(self, word: str) -> None:
        assert depluralize(word) == word
        assert word in tokenize(f"10,000 USD or {word}")

    @pytest.mark.parametrize("word", ["all", "any", "each", "every", "some", "none"])
    def test_a_quantifier_is_untouched(self, word: str) -> None:
        assert depluralize(word) == word

    @pytest.mark.parametrize("word", ["may", "must", "shall", "should", "can", "will"])
    def test_a_modal_is_untouched(self, word: str) -> None:
        assert depluralize(word) == word
        assert word in tokenize(f"a manager {word} approve the refund")

    @pytest.mark.parametrize("quantity", ["90", "10000", "7500", "365", "45", "2025", "24", "15"])
    def test_a_quantity_is_untouched(self, quantity: str) -> None:
        """Structurally, not incidentally: the rule requires an all-letters token, so a
        number cannot reach the suffix logic at all. A threshold is the term a change
        request most depends on."""
        assert depluralize(quantity) == quantity
        assert quantity in tokenize(f"{quantity} days")

    @pytest.mark.parametrize("token", ["v2s", "sha256s", "p1s"])
    def test_a_token_containing_a_digit_is_untouched(self, token: str) -> None:
        """An identifier is not a word. Leaving it whole is the conservative choice, and it
        keeps the rule's domain easy to state: letters only."""
        assert depluralize(token) == token

    def test_no_two_unrelated_corpus_terms_collapse_onto_one_stem(self) -> None:
        """The check that would have caught a bad guard, over real vocabulary.

        Each group below is a set of terms that share a stem. Every one must be a genuine
        singular/plural pair - if a future rule change merges two unrelated words, one of
        these groups gains a member that does not belong.
        """
        vocabulary = [
            "access",
            "accesses",
            "analysis",
            "analyses",
            "status",
            "class",
            "classes",
            "business",
            "loss",
            "basis",
            "bus",
            "gas",
            "refund",
            "refunds",
            "day",
            "days",
            "policy",
            "policies",
            "alias",
            "aliases",
        ]
        groups: dict[str, set[str]] = {}
        for term in vocabulary:
            groups.setdefault(depluralize(term), set()).add(term)

        merged = {stem: sorted(raw) for stem, raw in groups.items() if len(raw) > 1}
        assert merged == {
            "refund": ["refund", "refunds"],
            "day": ["day", "days"],
        }


class TestKnownLimitations:
    """What the rule does not do, stated as tests so nobody assumes otherwise.

    Each of these is a *missed* merge. They are recorded rather than fixed because fixing
    them means a second rule, a third, and eventually a stemmer that conflates
    ``operate``/``operation`` - which in this corpus are different claims.
    """

    def test_an_ies_plural_does_not_reach_its_singular(self) -> None:
        assert depluralize("policies") == "policie"
        assert depluralize("policies") != depluralize("policy")

    def test_an_irregular_plural_does_not_reach_its_singular(self) -> None:
        assert depluralize("analyses") != depluralize("analysis")

    def test_verb_inflection_is_not_handled(self) -> None:
        """``retained`` does not reach ``retain``, and ``retention`` reaches neither."""
        assert depluralize("retained") == "retained"
        assert depluralize("retention") == "retention"

    @pytest.mark.parametrize("acronym", ["apis", "kpis"])
    def test_an_acronym_ending_in_is_is_caught_by_the_analysis_guard(self, acronym: str) -> None:
        """Collateral of the ``-is`` guard, and accepted rather than worked around.

        ``apis`` does not reach ``api``, because the guard cannot tell it from ``basis``.
        Rescuing it would mean a length exception on the guard, which would then strip
        ``axis`` - trading a real protection for a marginal gain. The guard stays simple
        and this is written down instead.
        """
        assert depluralize(acronym) == acronym

    def test_a_third_person_verb_is_treated_as_a_plural_noun(self) -> None:
        """``requires`` becomes ``require``, which is a lucky accident rather than a design:
        the rule cannot tell a verb from a noun and does not try. It is harmless here
        because it merges the two forms of one word either way."""
        assert depluralize("requires") == "require"


class TestFrequencies:
    def test_repeats_are_counted(self) -> None:
        assert term_frequencies(tokenize("90 days is 90 days")) == {"90": 2, "day": 2}

    def test_no_terms_is_an_empty_mapping(self) -> None:
        assert term_frequencies(()) == {}


def test_the_tokenizer_is_versioned() -> None:
    """It feeds every score, so a change to it must be attributable."""
    assert TOKENIZER_VERSION
