"""Authorization on the lexical retrieval path.

``docs/SECURITY.md`` 4 requires authorization *before* retrieval output enters an agent, and
the requirement has two halves that pull against each other: an unauthorized span must be
excluded **and** the exclusion must be reported. Reporting it richly would defeat the point,
so the whole design is in what the trace is allowed to say - a count, and nothing else.

The tests worth reading first:

* :func:`TestNoLeakage.test_no_identifier_of_a_denied_document_appears_anywhere_in_the_outcome`
  serializes the entire outcome and asserts that no id, title, uri or span of the restricted
  document occurs in it.
* :func:`TestNoOracle.test_the_permission_count_does_not_depend_on_the_query` pins the reason
  authorization runs *first*: applied last, the count would answer "how many restricted
  documents mention my search terms?".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalTrace,
)
from app.domain.value_objects.enums import SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.retrieval.lexical import LexicalRetriever
from tests.fixtures.retrieval import AS_OF, OPEN_SCOPE, OTHER_TENANT, TENANT, candidate

OPEN_TEXT = "Refunds of 10,000 USD or more require manager approval before release."
PAYMENTS_TEXT = "Refunds above 10,000 USD are escalated to the payments fraud desk."
PRIVACY_TEXT = "Refunds for closed accounts retain personal data for 90 days."

PAYMENTS = PermissionScope.of("payments")
PRIVACY = PermissionScope.of("privacy")
BOTH = PermissionScope.of("payments", "privacy")


#: One unlabelled span and two labelled ones, all matching the word "refunds", so a scope
#: is the only thing that can decide which of them a requester sees.
def corpus() -> tuple[EvidenceCandidate, ...]:
    return (
        candidate(OPEN_TEXT, n=1),
        candidate(PAYMENTS_TEXT, n=2, permissions=("payments",), title="Payments Fraud Playbook"),
        candidate(PRIVACY_TEXT, n=3, permissions=("privacy",), title="Privacy Retention Note"),
    )


def query(
    text: str = "refunds", *, perm: PermissionScope = OPEN_SCOPE, **over: object
) -> RetrievalQuery:
    """A query under ``perm``.

    The permission argument is called ``perm`` and not ``scope`` on purpose:
    ``RetrievalQuery.scope`` is the *assumption* scope filter, and letting the two share a
    name in a test helper is how an authorization test ends up asserting nothing.
    """
    return RetrievalQuery(tenant_id=TENANT, text=text, permission_scope=perm, **over)


def retrieve(perm: PermissionScope, text: str = "refunds", **over: object) -> RetrievalOutcome:
    return LexicalRetriever(corpus()).retrieve(query(text, perm=perm, **over))


# ---------------------------------------------------------------------------
# Allowed, denied, mixed
# ---------------------------------------------------------------------------


class TestAllowed:
    def test_an_unlabelled_span_is_returned_to_the_least_privileged_scope(self) -> None:
        outcome = retrieve(OPEN_SCOPE)
        assert [r.candidate.text for r in outcome.results] == [OPEN_TEXT]

    def test_a_matching_grant_returns_the_labelled_span(self) -> None:
        outcome = retrieve(PAYMENTS)
        assert PAYMENTS_TEXT in [r.candidate.text for r in outcome.results]

    def test_a_scope_holding_every_grant_sees_the_whole_corpus(self) -> None:
        outcome = retrieve(BOTH)
        assert len(outcome.results) == 3
        assert outcome.trace.excluded_by_permission == 0

    def test_one_grant_reads_a_document_carrying_several_labels(self) -> None:
        """The labels are an ACL read disjunctively: holding any one of them is enough.

        Pinned here as well as on ``PermissionScope.permits`` because a clearance reading
        ("hold all of them") is a plausible wrong implementation that a single-label corpus
        cannot tell apart.
        """
        shared = candidate(PAYMENTS_TEXT, n=2, permissions=("payments", "privacy", "audit"))
        for grant in ("payments", "privacy", "audit"):
            outcome = LexicalRetriever((shared,)).retrieve(query(perm=PermissionScope.of(grant)))
            assert outcome.evidence_ids == (shared.evidence_id,), grant
        denied = LexicalRetriever((shared,)).retrieve(query(perm=PermissionScope.of("facilities")))
        assert denied.is_empty

    def test_case_does_not_decide_access(self) -> None:
        spans = (candidate(PAYMENTS_TEXT, n=2, permissions=("Payments",)),)
        outcome = LexicalRetriever(spans).retrieve(query(perm=PermissionScope.of("PAYMENTS")))
        assert outcome.results


class TestDenied:
    def test_a_labelled_span_is_withheld_from_a_scope_without_the_grant(self) -> None:
        outcome = retrieve(OPEN_SCOPE)
        texts = [r.candidate.text for r in outcome.results]
        assert PAYMENTS_TEXT not in texts
        assert PRIVACY_TEXT not in texts

    def test_the_denial_is_counted_rather_than_silent(self) -> None:
        outcome = retrieve(OPEN_SCOPE)
        assert outcome.trace.excluded_by_permission == 2

    def test_a_corpus_the_requester_cannot_read_at_all_returns_nothing(self) -> None:
        spans = (
            candidate(PAYMENTS_TEXT, n=2, permissions=("payments",)),
            candidate(PRIVACY_TEXT, n=3, permissions=("privacy",)),
        )
        outcome = LexicalRetriever(spans).retrieve(query(perm=PermissionScope.of("facilities")))
        assert outcome.is_empty
        assert outcome.trace.excluded_by_permission == 2
        assert outcome.trace.considered == 2

    def test_an_unrelated_grant_grants_nothing(self) -> None:
        outcome = retrieve(PermissionScope.of("facilities"))
        assert [r.candidate.text for r in outcome.results] == [OPEN_TEXT]
        assert outcome.trace.excluded_by_permission == 2


class TestMixed:
    def test_each_scope_sees_exactly_its_own_slice(self) -> None:
        assert {r.candidate.text for r in retrieve(PAYMENTS).results} == {OPEN_TEXT, PAYMENTS_TEXT}
        assert {r.candidate.text for r in retrieve(PRIVACY).results} == {OPEN_TEXT, PRIVACY_TEXT}

    def test_widening_a_scope_only_ever_adds(self) -> None:
        narrow = set(retrieve(PAYMENTS).evidence_ids)
        wide = set(retrieve(BOTH).evidence_ids)
        assert narrow < wide

    def test_the_permission_bucket_tracks_the_scope(self) -> None:
        assert retrieve(OPEN_SCOPE).trace.excluded_by_permission == 2
        assert retrieve(PAYMENTS).trace.excluded_by_permission == 1
        assert retrieve(BOTH).trace.excluded_by_permission == 0

    def test_the_trace_still_balances(self) -> None:
        """Every candidate is charged to exactly one bucket, authorization included."""
        for scope in (OPEN_SCOPE, PAYMENTS, PRIVACY, BOTH):
            trace = retrieve(scope).trace
            assert trace.excluded + trace.matched == trace.considered == 3

    def test_ranking_within_an_authorized_slice_is_still_deterministic(self) -> None:
        first = retrieve(BOTH)
        second = LexicalRetriever(tuple(reversed(corpus()))).retrieve(query(perm=BOTH))
        assert first.evidence_ids == second.evidence_ids


# ---------------------------------------------------------------------------
# Authorization is not a content filter
# ---------------------------------------------------------------------------


class TestNotAContentFilter:
    def test_a_denied_span_is_never_charged_to_a_content_bucket(self) -> None:
        """It is excluded on authorization grounds, before source type, as-of or scope."""
        outcome = retrieve(OPEN_SCOPE, source_types=(SourceType.POLICY,))
        assert outcome.trace.excluded_by_permission == 2
        assert outcome.trace.excluded_by_source_type == 0
        assert outcome.trace.excluded_by_as_of == 0
        assert outcome.trace.excluded_by_scope == 0

    def test_a_denied_span_cannot_be_reached_by_any_content_filter(self) -> None:
        for extra in (
            {"source_types": (SourceType.POLICY,)},
            {"as_of": AS_OF},
            {"scope": ScopeSelector.global_scope()},
            {"top_k": 100},
        ):
            outcome = retrieve(OPEN_SCOPE, **extra)
            assert PAYMENTS_TEXT not in [r.candidate.text for r in outcome.results]

    def test_a_score_does_not_depend_on_a_document_the_requester_cannot_read(self) -> None:
        """Corpus statistics are derived after authorization, so an unreadable span cannot
        move an authorized one's score - and two scopes legitimately rank differently."""
        alone = LexicalRetriever((candidate(OPEN_TEXT, n=1),)).retrieve(query(perm=OPEN_SCOPE))
        crowded = retrieve(OPEN_SCOPE)
        assert [r.score for r in alone.results] == [r.score for r in crowded.results]

    def test_the_same_span_may_score_differently_under_a_wider_scope(self) -> None:
        """Not a defect: a wider scope is a different corpus, which is what the hash records."""
        narrow = retrieve(OPEN_SCOPE).results[0]
        wide = next(r for r in retrieve(BOTH).results if r.evidence_id == narrow.evidence_id)
        assert narrow.score != wide.score


# ---------------------------------------------------------------------------
# The count must not become an oracle
# ---------------------------------------------------------------------------


class TestNoOracle:
    def test_the_permission_count_does_not_depend_on_the_query(self) -> None:
        """Authorization runs before scoring, so the count reports how much of the corpus is
        closed to the requester - never whether any of it matched their words."""
        matching = retrieve(OPEN_SCOPE, text="refunds")
        missing = retrieve(OPEN_SCOPE, text="bicycle helmet procurement")
        assert matching.trace.excluded_by_permission == missing.trace.excluded_by_permission == 2

    def test_the_permission_count_does_not_depend_on_a_content_filter(self) -> None:
        broad = retrieve(OPEN_SCOPE)
        narrowed = retrieve(OPEN_SCOPE, source_types=(SourceType.EMAIL_TEMPLATE,))
        assert broad.trace.excluded_by_permission == narrowed.trace.excluded_by_permission == 2

    def test_a_targeted_probe_for_restricted_wording_reveals_nothing(self) -> None:
        """The classic probe: query the exact words of a document you may not read."""
        probe = retrieve(OPEN_SCOPE, text="payments fraud desk escalated")
        assert probe.is_empty
        assert probe.trace.excluded_by_permission == 2
        assert probe.trace.excluded_zero_score == 1


# ---------------------------------------------------------------------------
# Nothing about a denied document escapes
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_no_identifier_of_a_denied_document_appears_anywhere_in_the_outcome(self) -> None:
        """Serialized whole, and searched for every field a denied document has."""
        denied = candidate(
            PAYMENTS_TEXT, n=2, permissions=("payments",), title="Payments Fraud Playbook"
        )
        outcome = retrieve(OPEN_SCOPE, text="refunds")
        serialized = outcome.model_dump_json()

        for secret in (
            denied.evidence_id,
            denied.chunk_id,
            denied.source_id,
            denied.source_version_id,
            denied.document.title,
            denied.document.uri,
            denied.text,
            "escalated",
            "fraud",
        ):
            assert secret not in serialized, secret

    def test_a_probe_echoes_only_the_requesters_own_words_back(self) -> None:
        """A query naming restricted wording is answered with that wording in the echoed
        query and nowhere else. The caller already knows what they typed; what they must not
        learn is whether anything matched it."""
        denied = candidate(PAYMENTS_TEXT, n=2, permissions=("payments",))
        outcome = retrieve(OPEN_SCOPE, text="payments fraud desk escalated")
        assert outcome.is_empty

        without_query = outcome.model_dump_json(exclude={"query"})
        for word in ("payments", "fraud", "escalated", denied.evidence_id, denied.text):
            assert word not in without_query, word

    def test_the_trace_exposes_a_count_and_no_identifier(self) -> None:
        trace = retrieve(OPEN_SCOPE).trace
        dumped = trace.model_dump()
        assert dumped["excluded_by_permission"] == 2
        assert all(isinstance(value, int) for value in dumped.values())

    def test_the_trace_has_no_field_that_could_carry_an_identifier(self) -> None:
        """Structural: every field on the trace is a count, so there is nowhere to put one."""
        for name, field in RetrievalTrace.model_fields.items():
            assert field.annotation is int, name

    def test_a_denied_span_is_absent_from_the_result_list_itself(self) -> None:
        outcome = retrieve(OPEN_SCOPE, text="refunds", top_k=100)
        denied_ids = {c.evidence_id for c in corpus()[1:]}
        assert not denied_ids & set(outcome.evidence_ids)


# ---------------------------------------------------------------------------
# The scope hash on the outcome
# ---------------------------------------------------------------------------


class TestScopeHashOnOutcome:
    def test_the_outcome_records_the_scope_it_was_computed_under(self) -> None:
        outcome = retrieve(PAYMENTS)
        assert outcome.permission_scope_hash == PAYMENTS.scope_hash

    def test_different_scopes_produce_different_hashes(self) -> None:
        hashes = {retrieve(scope).permission_scope_hash for scope in (OPEN_SCOPE, PAYMENTS, BOTH)}
        assert len(hashes) == 3

    def test_the_same_scope_produces_the_same_hash_across_runs(self) -> None:
        assert retrieve(PAYMENTS).permission_scope_hash == retrieve(PAYMENTS).permission_scope_hash

    def test_an_equivalent_scope_spelled_differently_produces_the_same_hash(self) -> None:
        """Deterministic for the effective scope: a cache key must not fragment on spelling."""
        spelled = PermissionScope(grants=("Privacy", "payments", "payments"))
        assert retrieve(spelled).permission_scope_hash == retrieve(BOTH).permission_scope_hash
        assert retrieve(spelled).evidence_ids == retrieve(BOTH).evidence_ids

    def test_a_relabelled_outcome_is_rejected(self) -> None:
        """Results computed under a wide scope must not be servable to a narrow one."""
        wide = retrieve(BOTH)
        with pytest.raises(ValidationError, match="does not match the scope"):
            RetrievalOutcome(
                query=query(perm=OPEN_SCOPE),
                results=wide.results,
                trace=wide.trace,
                permission_scope_hash=BOTH.scope_hash,
                scorer_version=wide.scorer_version,
            )

    def test_an_outcome_cannot_omit_the_hash(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalOutcome(  # type: ignore[call-arg]
                query=query(),
                trace=RetrievalTrace(considered=0, matched=0, returned=0),
                scorer_version="lexical-bm25-1",
            )


# ---------------------------------------------------------------------------
# The query cannot be built without a scope
# ---------------------------------------------------------------------------


class TestQueryRequiresAScope:
    def test_a_query_without_a_permission_scope_is_rejected(self) -> None:
        """Forgetting the parameter must be an error, never the unrestricted case."""
        with pytest.raises(ValidationError, match="permission_scope"):
            RetrievalQuery(tenant_id=TENANT, text="refunds")  # type: ignore[call-arg]

    def test_a_null_permission_scope_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(tenant_id=TENANT, text="refunds", permission_scope=None)

    def test_the_query_exposes_the_hash_it_will_be_answered_under(self) -> None:
        assert query(perm=PAYMENTS).permission_scope_hash == PAYMENTS.scope_hash


# ---------------------------------------------------------------------------
# Tenant isolation is unchanged by any of this
# ---------------------------------------------------------------------------


class TestTenantIsolationStillHolds:
    def test_a_grant_does_not_cross_a_tenant_boundary(self) -> None:
        """Holding the label a foreign document carries must not make it readable."""
        theirs = candidate(PAYMENTS_TEXT, n=9, tenant_id=OTHER_TENANT, permissions=("payments",))
        outcome = LexicalRetriever((theirs, *corpus())).retrieve(query(perm=PAYMENTS))
        assert theirs.evidence_id not in outcome.evidence_ids
        assert outcome.trace.excluded_by_tenant == 1

    def test_a_foreign_span_is_charged_to_the_tenant_bucket_not_the_permission_one(self) -> None:
        theirs = candidate(OPEN_TEXT, n=9, tenant_id=OTHER_TENANT)
        outcome = LexicalRetriever((theirs, *corpus())).retrieve(query(perm=BOTH))
        assert outcome.trace.excluded_by_tenant == 1
        assert outcome.trace.excluded_by_permission == 0

    def test_a_foreign_tenants_restricted_span_leaks_nothing(self) -> None:
        theirs = candidate(
            PAYMENTS_TEXT,
            n=9,
            tenant_id=OTHER_TENANT,
            permissions=("payments",),
            title="Their Fraud Playbook",
        )
        outcome = LexicalRetriever((theirs, *corpus())).retrieve(
            query(text="refunds payments fraud", perm=PAYMENTS)
        )
        serialized = outcome.model_dump_json()
        assert theirs.evidence_id not in serialized
        assert theirs.document.title not in serialized


def test_authorization_is_enforced_by_narrowing_the_index_not_by_a_late_check() -> None:
    """A structural assertion, so the ordering cannot be undone by a later edit.

    ``retrieve`` must reach the content filters with an index that already excludes
    unreadable spans; if authorization were one more entry in ``_first_failing_filter`` the
    guarantee would rest on the order of a chain of ``if`` statements.

    The membership test is a substring scan and not ``in co_names``, which is how the first
    version of this test let a per-candidate check through: ``"permission"`` is not equal to
    ``"permission_scope"``, so exact membership passed while the mutation sat there.
    """
    from app.retrieval import lexical

    assert "_authorized_slice" in dir(lexical)
    reachable = lexical._first_failing_filter.__code__.co_names
    leaked = [name for name in reachable if "permission" in name or "readable" in name]
    assert not leaked, f"authorization decided inside the content filters: {leaked}"
