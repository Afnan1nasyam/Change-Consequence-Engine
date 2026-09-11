"""The confidence-input assembler: what each factor is measured from, and what it cannot see.

Confidence was wired in the judge slice and every score was an honest zero, because nothing
measured the factors. This is the measurement. It changes neither the factor list (checkpoint
9.1) nor the weights - only where each number comes from - so the tests are almost entirely
about the *sources* being the right ones and about three things being unreachable:

* **agent identity, count, model and agreement** - checkpoint 9.3, enforced by the data not
  existing at this boundary rather than by a rule;
* **evidence the requester may not read** - the assembler measures only what the retrievers
  returned, so a narrower scope can never score higher;
* **a candidate's own divergence** - a finding that reports a contradiction must not be
  penalized for the thing it found.

The tests carrying the most weight:

* :func:`TestAgentIdentityCannotEnter.test_the_signature_has_no_agent_shaped_parameter`
* :func:`TestAgentIdentityCannotEnter.test_identical_evidence_scores_identically_whoever_reported_it`
* :func:`TestAuthorization.test_a_narrower_scope_can_never_score_higher`
* :func:`TestCorroboration.test_duplicate_source_versions_collapse_to_one`
* :func:`TestTheSampleCorpus` - the corpus-level measurement, which contradicted a
  prediction recorded in ``docs/STATUS.md`` and is documented as measured.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import inspect
from typing import Any

import pytest

from app.application.orchestration.retrieval_service import (
    RetrievalRequest,
    RetrievalService,
)
from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.contradictions.divergence import DivergenceClass, DivergenceItem
from app.contradictions.hunter import AssessedDivergence
from app.domain.entities.assumption import AssumptionRecord
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.source import (
    EvidenceItem,
    SourceDocument,
    SourceVersion,
)
from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
)
from app.domain.services.confidence import ConfidenceInputs
from app.domain.services.confidence import compute as compute_confidence
from app.domain.value_objects.enums import (
    AUTHORITY_RANK,
    AuthorityClass,
    BindingRole,
    ConsequenceClass,
    Explicitness,
    RetrievalStrategy,
    SourceType,
    ValueType,
)
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.typed_value import TypedValue
from app.evaluation.corpus import CorpusStores, load_corpus
from app.infrastructure.embeddings.hashing import HashingEmbedder
from app.judgment.confidence_inputs import (
    CONTRADICTION_WEIGHTS,
    FRESHNESS_HORIZON,
    INFERENCE_DEPTH,
    ConfidenceAssemblyError,
    assemble,
)

TENANT = "tn_" + "0" * 31 + "1"
FOREIGN = "tn_" + "0" * 31 + "2"
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
CLASS_A = ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
CLASS_B = ConsequenceClass.CLASS_B_BEHAVIORAL

#: The restricted document in the sample corpus, and the grant that opens it.
RESTRICTED_URI = "corpus://policy/data-retention-policy.md"
PRIVACY_GRANT = "privacy-office"

#: A query the sample corpus answers from several documents.
BROAD_QUERY = "personal data is retained for 90 days after the account is closed"
#: A query the sample corpus answers from exactly one span.
NARROW_QUERY = "token expiry"


# ---------------------------------------------------------------------------
# Synthetic candidates, for the per-factor tests
# ---------------------------------------------------------------------------


def _hex_id(prefix: str, key: str) -> str:
    """A well-formed identifier derived from a readable tag.

    Ids are lowercase hex by contract, so a tag like ``"shared"`` cannot be padded into one.
    Hashing keeps the test readable while producing a legal id, and two calls with one tag
    produce one id - which is what lets two spans share a version.
    """
    return f"{prefix}_{hashlib.sha256(key.encode()).hexdigest()[:32]}"


def candidate(
    *,
    tag: str,
    authority: AuthorityClass = AuthorityClass.POLICY,
    source_type: SourceType = SourceType.POLICY,
    effective_from: dt.datetime | None = None,
    tenant_id: str = TENANT,
    version_tag: str | None = None,
) -> EvidenceCandidate:
    """One retrievable span, joined to a document and a version.

    ``version_tag`` lets two spans share a version, which is how the duplicate-version test
    builds a document quoted twice.
    """
    version_key = version_tag or tag
    source_id = _hex_id("src", version_key)
    version_id = _hex_id("sv", version_key)
    document = SourceDocument(
        source_id=source_id,
        tenant_id=tenant_id,
        uri=f"corpus://{version_key}.md",
        title=f"Document {version_key}",
        source_type=source_type,
        authority_class=authority,
    )
    version = SourceVersion(
        version_id=version_id,
        source_id=source_id,
        content_hash="a" * 64,
        version_label="v1",
        effective_from=effective_from,
    )
    evidence = EvidenceItem(
        evidence_id=_hex_id("ev", tag),
        source_version_id=version_id,
        chunk_id=_hex_id("ch", tag),
        text_span=f"span {tag} states a retention window of 90 days.",
        char_start=0,
        char_end=48,
    )
    return EvidenceCandidate(evidence=evidence, document=document, version=version)


def outcome(
    *candidates: EvidenceCandidate,
    strategy: RetrievalStrategy = RetrievalStrategy.LEXICAL,
    tenant_id: str = TENANT,
    scope: PermissionScope | None = None,
) -> RetrievalOutcome:
    """A well-formed outcome over the given candidates, as one strategy would return them."""
    query = RetrievalQuery(
        tenant_id=tenant_id,
        text="retention window",
        permission_scope=scope or PermissionScope(),
    )
    results = tuple(
        RetrievalResult(
            candidate=item,
            strategy=strategy,
            score=1.0 / (index + 1),
            rank=index + 1,
        )
        for index, item in enumerate(candidates)
    )
    return RetrievalOutcome(
        query=query,
        results=results,
        strategies_executed=(strategy,),
        trace=RetrievalTrace(
            considered=len(candidates),
            matched=len(candidates),
            returned=len(candidates),
        ),
        permission_scope_hash=query.permission_scope_hash,
        scorer_version="test-scorer-1",
    )


def binding(
    *,
    tag: str,
    explicitness: Explicitness = Explicitness.EXPLICIT,
    authority: AuthorityClass = AuthorityClass.POLICY,
) -> ResolvedBinding:
    record = AssumptionRecord(
        assumption_id=_hex_id("as", tag),
        tenant_id=TENANT,
        key="retention.window",
        registry_version="registry-1",
        value=TypedValue(value_type=ValueType.DURATION, raw=90, unit="d"),
        role=BindingRole.DECLARES,
        explicitness=explicitness,
        holder_source_version_id=_hex_id("sv", tag),
        evidence_id=_hex_id("ev", tag),
        extractor_version="extractor-1",
        span_verified=True,
        value_parsed=True,
        key_resolved=True,
    )
    return ResolvedBinding(
        record=record,
        authority_class=authority,
        source_id=_hex_id("src", tag),
        source_title=f"Document {tag}",
    )


def divergence(
    divergence_class: DivergenceClass = DivergenceClass.MATERIAL_CONTRADICTION,
    *,
    participants: tuple[str, ...] = ("as_" + "8" * 32, "as_" + "9" * 32),
) -> AssessedDivergence:
    is_class_a = divergence_class in {
        DivergenceClass.MATERIAL_CONTRADICTION,
        DivergenceClass.STALE_INVALIDATION,
        DivergenceClass.UNAUTHORIZED_OVERRIDE,
    }
    relationship = {
        DivergenceClass.MATERIAL_CONTRADICTION: "conflicts_with",
        DivergenceClass.STALE_INVALIDATION: "invalidates_assumption",
        DivergenceClass.UNAUTHORIZED_OVERRIDE: "unauthorized_override",
    }.get(divergence_class)
    from app.domain.value_objects.taxonomy import Relationship

    item = DivergenceItem(
        item_id="d1",
        divergence_class=divergence_class,
        assumption_key="retention.window",
        scope="global",
        as_of=AS_OF,
        participants=participants,
        evidence_ids=("ev_" + "1" * 32,),
        detail="two declarations diverge",
        relation_id=("dr_" + "1" * 32) if is_class_a else None,
        relationship=Relationship(relationship) if relationship else None,
    )
    return AssessedDivergence(item=item)


def measure(**over: Any) -> Any:
    """Assemble with sensible defaults, one field overridable per test."""
    kwargs: dict[str, Any] = {
        "supporting_evidence_ids": (),
        "outcomes": (),
        "consequence_class": CLASS_A,
        "tenant_id": TENANT,
        "as_of": AS_OF,
    }
    kwargs.update(over)
    return assemble(**kwargs)


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------


class TestAgentIdentityCannotEnter:
    def test_the_signature_has_no_agent_shaped_parameter(self) -> None:
        """Checkpoint 9.3, enforced by the data not existing at this boundary.

        Corroboration counts sources; several agents reading one span are one source. This
        function cannot be told how many agents there were, so it cannot be talked into
        treating their agreement as corroboration.
        """
        parameters = set(inspect.signature(assemble).parameters)

        assert parameters == {
            "supporting_evidence_ids",
            "outcomes",
            "consequence_class",
            "tenant_id",
            "as_of",
            "bindings",
            "own_input_record_ids",
            "divergences",
            "alias_mediated",
        }
        assert (
            not {
                "agent",
                "agents",
                "agent_count",
                "agreeing_agents",
                "model",
                "provenance",
                "votes",
                "confidence",
                "severity",
            }
            & parameters
        )

    def test_identical_evidence_scores_identically_whoever_reported_it(self) -> None:
        """Two runs over one evidence set must produce one number.

        The realistic version of the rule: the same span reported by one agent and by four is
        the same evidence, and the assembler has no way to tell the difference - so this test
        is about there being nothing to vary.
        """
        spans = (candidate(tag="a"), candidate(tag="b"))
        first = measure(
            supporting_evidence_ids=[s.evidence_id for s in spans], outcomes=(outcome(*spans),)
        )
        second = measure(
            supporting_evidence_ids=[s.evidence_id for s in spans], outcomes=(outcome(*spans),)
        )

        assert first.inputs == second.inputs

    def test_the_inputs_type_still_forbids_an_agent_field(self) -> None:
        """Belt and braces: the assembler could not populate one even if it wanted to."""
        with pytest.raises(Exception, match=r"forbidden|extra"):
            ConfidenceInputs(agreeing_agents=4)  # type: ignore[call-arg]

    def test_the_assembler_is_deterministic_over_input_ordering(self) -> None:
        """Requirement 7. Two orderings of the same evidence give the same number."""
        spans = (candidate(tag="a"), candidate(tag="b"), candidate(tag="c"))
        forward = measure(
            supporting_evidence_ids=[s.evidence_id for s in spans],
            outcomes=(outcome(*spans),),
        )
        reverse = measure(
            supporting_evidence_ids=[s.evidence_id for s in reversed(spans)],
            outcomes=(outcome(*reversed(spans)),),
        )

        assert forward.inputs == reverse.inputs


# ---------------------------------------------------------------------------
# Corroboration
# ---------------------------------------------------------------------------


class TestCorroboration:
    def test_no_supporting_evidence_measures_no_corroboration(self) -> None:
        assembled = measure()

        assert assembled.inputs.corroborating_source_version_ids == ()
        assert assembled.inputs.corroborating_strategies == ()
        assert assembled.coverage.matched_candidates == 0
        assert "corroboration" in assembled.coverage.unmeasured

    def test_distinct_source_versions_corroborate(self) -> None:
        spans = (candidate(tag="a"), candidate(tag="b"), candidate(tag="c"))
        assembled = measure(
            supporting_evidence_ids=[s.evidence_id for s in spans],
            outcomes=(outcome(*spans),),
        )

        assert assembled.inputs.corroboration_count == 3
        assert assembled.coverage.distinct_versions == 3

    def test_duplicate_source_versions_collapse_to_one(self) -> None:
        """A statement quoted twice in one document is one source, not two.

        The whole mechanism rests on this: without it, a widely-copied paragraph would
        outscore a genuinely triangulated claim.
        """
        spans = (
            candidate(tag="a1", version_tag="shared"),
            candidate(tag="a2", version_tag="shared"),
            candidate(tag="a3", version_tag="shared"),
        )
        assembled = measure(
            supporting_evidence_ids=[s.evidence_id for s in spans],
            outcomes=(outcome(*spans),),
        )

        assert assembled.coverage.matched_candidates == 3
        assert assembled.inputs.corroboration_count == 1

    def test_three_distinct_versions_score_above_three_copies_of_one(self) -> None:
        """The consequence of the rule above, at the score."""
        distinct = (candidate(tag="a"), candidate(tag="b"), candidate(tag="c"))
        copies = (
            candidate(tag="a1", version_tag="shared"),
            candidate(tag="a2", version_tag="shared"),
            candidate(tag="a3", version_tag="shared"),
        )
        scores = [
            compute_confidence(
                measure(
                    supporting_evidence_ids=[s.evidence_id for s in spans],
                    outcomes=(outcome(*spans),),
                ).inputs,
                consequence_id="cq_" + "7" * 32,
                computed_at=AS_OF,
            ).confidence
            for spans in (distinct, copies)
        ]

        assert scores[0] > scores[1]

    def test_a_cited_id_no_strategy_returned_contributes_nothing(self) -> None:
        """The requester never received it, so it cannot support their claim."""
        span = candidate(tag="a")
        assembled = measure(
            supporting_evidence_ids=[span.evidence_id, "ev_" + "f" * 32],
            outcomes=(outcome(span),),
        )

        assert assembled.coverage.supporting_ids == 2
        assert assembled.coverage.matched_candidates == 1
        assert assembled.coverage.unmatched_ids == 1
        assert assembled.inputs.corroboration_count == 1

    def test_only_strategies_that_returned_a_supporting_span_are_counted(self) -> None:
        """A strategy that ran and returned nothing relevant has corroborated nothing.

        Counting it would let an empty second strategy widen the corroboration gate for
        free, which is exactly the kind of free lunch the gate exists to refuse.
        """
        span = candidate(tag="a")
        other = candidate(tag="z")
        assembled = measure(
            supporting_evidence_ids=[span.evidence_id],
            outcomes=(
                outcome(span, strategy=RetrievalStrategy.LEXICAL),
                outcome(other, strategy=RetrievalStrategy.VECTOR),
            ),
        )

        assert assembled.inputs.corroborating_strategies == ("lexical",)
        assert assembled.coverage.strategies_executed == 2


# ---------------------------------------------------------------------------
# Strategy reach and agreement
# ---------------------------------------------------------------------------


class TestStrategyReach:
    def test_both_strategies_reaching_every_span_is_full_agreement(self) -> None:
        spans = (candidate(tag="a"), candidate(tag="b"))
        assembled = measure(
            supporting_evidence_ids=[s.evidence_id for s in spans],
            outcomes=(
                outcome(*spans, strategy=RetrievalStrategy.LEXICAL),
                outcome(*spans, strategy=RetrievalStrategy.VECTOR),
            ),
        )

        assert assembled.inputs.retrieval_agreement == 1.0
        assert assembled.inputs.corroborating_strategies == ("lexical", "vector")

    def test_partial_reach_is_partial_agreement(self) -> None:
        first, second = candidate(tag="a"), candidate(tag="b")
        assembled = measure(
            supporting_evidence_ids=[first.evidence_id, second.evidence_id],
            outcomes=(
                outcome(first, second, strategy=RetrievalStrategy.LEXICAL),
                outcome(first, strategy=RetrievalStrategy.VECTOR),
            ),
        )

        assert assembled.inputs.retrieval_agreement == 0.5

    def test_one_strategy_cannot_agree_with_itself(self) -> None:
        """Zero, and reported as unmeasured - which is different from measured disagreement."""
        span = candidate(tag="a")
        assembled = measure(supporting_evidence_ids=[span.evidence_id], outcomes=(outcome(span),))

        assert assembled.inputs.retrieval_agreement == 0.0
        assert "retrieval_agreement" in assembled.coverage.unmeasured

    def test_agreement_is_reach_not_score(self) -> None:
        """A BM25 score and a cosine similarity are not on a common scale.

        Two outcomes with wildly different score magnitudes but the same reach agree fully.
        Comparing the numbers would need a normalization neither scorer supports, and any
        value produced that way would describe the normalization.
        """
        span = candidate(tag="a")
        lexical = outcome(span, strategy=RetrievalStrategy.LEXICAL)
        vector = outcome(span, strategy=RetrievalStrategy.VECTOR)
        loud = lexical.model_copy(
            update={"results": (lexical.results[0].model_copy(update={"score": 42.0}),)}
        )

        assembled = measure(supporting_evidence_ids=[span.evidence_id], outcomes=(loud, vector))
        assert assembled.inputs.retrieval_agreement == 1.0


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------


class TestAuthority:
    @pytest.mark.parametrize(
        "authority", [AuthorityClass.REGULATORY, AuthorityClass.POLICY, AuthorityClass.CHAT]
    )
    def test_authority_is_the_normalized_holder_rank(self, authority: AuthorityClass) -> None:
        span = candidate(tag="a", authority=authority)
        assembled = measure(supporting_evidence_ids=[span.evidence_id], outcomes=(outcome(span),))

        expected = AUTHORITY_RANK[authority] / max(AUTHORITY_RANK.values())
        assert assembled.inputs.source_authority == pytest.approx(expected)

    def test_a_regulation_outscores_a_chat_message(self) -> None:
        high = candidate(tag="a", authority=AuthorityClass.REGULATORY)
        low = candidate(tag="b", authority=AuthorityClass.CHAT)

        scores = [
            measure(
                supporting_evidence_ids=[span.evidence_id], outcomes=(outcome(span),)
            ).inputs.source_authority
            for span in (high, low)
        ]
        assert scores[0] > scores[1]

    def test_a_weak_citation_dilutes_rather_than_being_carried(self) -> None:
        """Mean, not max. One strong citation must not carry a pile of weak ones."""
        policy = candidate(tag="a", authority=AuthorityClass.POLICY)
        chat = candidate(tag="b", authority=AuthorityClass.CHAT)

        alone = measure(supporting_evidence_ids=[policy.evidence_id], outcomes=(outcome(policy),))
        together = measure(
            supporting_evidence_ids=[policy.evidence_id, chat.evidence_id],
            outcomes=(outcome(policy, chat),),
        )

        assert together.inputs.source_authority < alone.inputs.source_authority


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


class TestFreshness:
    def test_a_recent_version_is_fresher_than_an_old_one(self) -> None:
        fresh = candidate(tag="a", effective_from=AS_OF - dt.timedelta(days=30))
        stale = candidate(tag="b", effective_from=AS_OF - dt.timedelta(days=365 * 4))

        scores = [
            measure(
                supporting_evidence_ids=[span.evidence_id], outcomes=(outcome(span),)
            ).inputs.source_freshness
            for span in (fresh, stale)
        ]
        assert scores[0] > scores[1]
        assert scores[0] == pytest.approx(1.0 - 30 / FRESHNESS_HORIZON.days, abs=1e-5)

    def test_a_version_older_than_the_horizon_scores_zero(self) -> None:
        ancient = candidate(tag="a", effective_from=AS_OF - dt.timedelta(days=365 * 9))
        assembled = measure(
            supporting_evidence_ids=[ancient.evidence_id], outcomes=(outcome(ancient),)
        )

        assert assembled.inputs.source_freshness == 0.0
        # Measured as absent, not unmeasurable - the version *is* dated.
        assert "source_freshness" not in assembled.coverage.unmeasured

    def test_a_version_not_yet_in_force_is_not_stale(self) -> None:
        """Full freshness rather than negative age. Whether it governs is a filter's job."""
        future = candidate(tag="a", effective_from=AS_OF + dt.timedelta(days=90))
        assembled = measure(
            supporting_evidence_ids=[future.evidence_id], outcomes=(outcome(future),)
        )

        assert assembled.inputs.source_freshness == 1.0

    def test_an_undated_version_is_excluded_rather_than_scored_stale(self) -> None:
        """Scoring it zero would report an undated document as old, which the corpus does not
        say. It is excluded from the mean and reported as unmeasured instead."""
        undated = candidate(tag="a", effective_from=None)
        assembled = measure(
            supporting_evidence_ids=[undated.evidence_id], outcomes=(outcome(undated),)
        )

        assert assembled.inputs.source_freshness == 0.0
        assert "source_freshness" in assembled.coverage.unmeasured
        assert assembled.coverage.versions_dated == 0

    def test_a_dated_version_beside_an_undated_one_still_measures(self) -> None:
        dated = candidate(tag="a", effective_from=AS_OF - dt.timedelta(days=30))
        undated = candidate(tag="b", effective_from=None)
        assembled = measure(
            supporting_evidence_ids=[dated.evidence_id, undated.evidence_id],
            outcomes=(outcome(dated, undated),),
        )

        assert assembled.inputs.source_freshness > 0.9
        assert assembled.coverage.versions_dated == 1
        assert "source_freshness" not in assembled.coverage.unmeasured

    def test_freshness_averages_versions_not_spans(self) -> None:
        """Two quotations from one document describe one document's age."""
        twice = (
            candidate(tag="a1", version_tag="shared", effective_from=AS_OF - dt.timedelta(days=30)),
            candidate(tag="a2", version_tag="shared", effective_from=AS_OF - dt.timedelta(days=30)),
        )
        once = candidate(tag="b", effective_from=AS_OF - dt.timedelta(days=30))

        pair = measure(
            supporting_evidence_ids=[s.evidence_id for s in twice],
            outcomes=(outcome(*twice),),
        )
        single = measure(supporting_evidence_ids=[once.evidence_id], outcomes=(outcome(once),))
        assert pair.inputs.source_freshness == single.inputs.source_freshness


# ---------------------------------------------------------------------------
# Explicitness, depth, contradiction, alias
# ---------------------------------------------------------------------------


class TestTheRemainingFactors:
    @pytest.mark.parametrize(
        ("explicitness", "expected"),
        [
            (Explicitness.EXPLICIT, 1.0),
            (Explicitness.DERIVED_FROM_TEXT, 2 / 3),
            (Explicitness.IMPLIED, 1 / 3),
        ],
    )
    def test_explicitness_is_the_normalized_binding_rank(
        self, explicitness: Explicitness, expected: float
    ) -> None:
        held = binding(tag="a", explicitness=explicitness)
        assembled = measure(
            bindings={held.assumption_id: held}, own_input_record_ids=(held.assumption_id,)
        )

        assert assembled.inputs.explicitness == pytest.approx(expected)

    def test_explicitness_is_scoped_to_the_candidates_own_bindings(self) -> None:
        """An explicit binding elsewhere in the corpus must not raise this claim."""
        mine = binding(tag="a", explicitness=Explicitness.IMPLIED)
        theirs = binding(tag="b", explicitness=Explicitness.EXPLICIT)
        assembled = measure(
            bindings={mine.assumption_id: mine, theirs.assumption_id: theirs},
            own_input_record_ids=(mine.assumption_id,),
        )

        assert assembled.inputs.explicitness == pytest.approx(1 / 3)
        assert assembled.coverage.bindings_supplied == 1

    def test_no_bindings_leaves_explicitness_unmeasured(self) -> None:
        assembled = measure()

        assert assembled.inputs.explicitness == 0.0
        assert "explicitness" in assembled.coverage.unmeasured

    @pytest.mark.parametrize("consequence_class", [CLASS_A, CLASS_B])
    def test_inference_depth_is_counted_from_the_producer(
        self, consequence_class: ConsequenceClass
    ) -> None:
        """A derivation is one hop; a hypothesis is inferred beyond the evidence layer."""
        assembled = measure(consequence_class=consequence_class)
        assert assembled.inputs.inference_depth == INFERENCE_DEPTH[consequence_class]

    def test_a_hypothesis_scores_below_an_identically_evidenced_derivation(self) -> None:
        span = candidate(tag="a", effective_from=AS_OF - dt.timedelta(days=30))
        scores = [
            compute_confidence(
                measure(
                    supporting_evidence_ids=[span.evidence_id],
                    outcomes=(outcome(span),),
                    consequence_class=klass,
                ).inputs,
                consequence_id="cq_" + "7" * 32,
                computed_at=AS_OF,
            ).confidence
            for klass in (CLASS_A, CLASS_B)
        ]
        assert scores[0] > scores[1]

    @pytest.mark.parametrize("divergence_class", sorted(DivergenceClass))
    def test_contradiction_strength_follows_the_divergence_class(
        self, divergence_class: DivergenceClass
    ) -> None:
        assembled = measure(divergences=(divergence(divergence_class),))
        assert assembled.inputs.contradiction_strength == (CONTRADICTION_WEIGHTS[divergence_class])

    def test_the_two_non_contradictions_weigh_nothing(self) -> None:
        """The checkpoint says a legitimate override and a consumer divergence are not
        contradictions, so they must not lower confidence in anything."""
        for divergence_class in (
            DivergenceClass.LEGITIMATE_SCOPED_OVERRIDE,
            DivergenceClass.NON_MATERIAL_DIVERGENCE,
        ):
            assembled = measure(divergences=(divergence(divergence_class),))
            assert assembled.inputs.contradiction_strength == 0.0

    def test_a_candidates_own_divergence_does_not_penalize_it(self) -> None:
        """A finding that *reports* a contradiction has it as evidence, not as doubt.

        Without the exclusion every contradiction finding would be marked down for the thing
        it found, which inverts the factor.
        """
        own = ("as_" + "1" * 32, "as_" + "2" * 32)
        mine = divergence(DivergenceClass.MATERIAL_CONTRADICTION, participants=own)
        elsewhere = divergence(DivergenceClass.MATERIAL_CONTRADICTION)

        assert (
            measure(divergences=(mine,), own_input_record_ids=own).inputs.contradiction_strength
            == 0.0
        )
        assert (
            measure(
                divergences=(elsewhere,), own_input_record_ids=own
            ).inputs.contradiction_strength
            == 1.0
        )

    def test_the_strongest_other_divergence_wins(self) -> None:
        assembled = measure(
            divergences=(
                divergence(DivergenceClass.STALE_INVALIDATION),
                divergence(DivergenceClass.MATERIAL_CONTRADICTION),
            )
        )
        assert assembled.inputs.contradiction_strength == 1.0

    def test_alias_mediation_is_carried_through(self) -> None:
        assert measure(alias_mediated=True).inputs.alias_mediated is True
        assert measure().inputs.alias_mediated is False


# ---------------------------------------------------------------------------
# Tenant and permission boundaries
# ---------------------------------------------------------------------------


class TestAuthorization:
    def test_a_cross_tenant_candidate_raises(self) -> None:
        """An isolation defect must be loud, not a slightly different number."""
        foreign = candidate(tag="a", tenant_id=FOREIGN)

        with pytest.raises(ConfidenceAssemblyError, match="tenant boundary"):
            measure(
                supporting_evidence_ids=[foreign.evidence_id],
                outcomes=(outcome(foreign, tenant_id=FOREIGN),),
                tenant_id=TENANT,
            )

    def test_the_assembler_reads_no_store(self) -> None:
        """It measures the outcomes it was handed and cannot reach past the retrievers.

        Parsed rather than grepped: a substring search would match the docstring explaining
        the rule, which is a test that reads correctly and asserts nothing.
        """
        import ast
        import pathlib

        source = pathlib.Path("app/judgment/confidence_inputs.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        assert not any(m.startswith("app.infrastructure") for m in imported)
        assert not any("repositories" in m or "store" in m for m in imported)

    def test_a_narrower_scope_can_never_score_higher(self, corpus: CorpusStores) -> None:
        """The authorization effect, measured over the real corpus.

        A withheld span never reaches the outcome, so it cannot enter the measurement. The
        narrower scope sees fewer source versions and lower authority, and therefore a lower
        or equal confidence - never higher.
        """
        narrow = _measure_query(corpus, BROAD_QUERY, PermissionScope())
        wide = _measure_query(corpus, BROAD_QUERY, PermissionScope.of(PRIVACY_GRANT))

        assert narrow.coverage.distinct_versions < wide.coverage.distinct_versions
        assert narrow.inputs.source_authority <= wide.inputs.source_authority
        assert _confidence(narrow) <= _confidence(wide)

    def test_the_withheld_document_is_absent_from_the_narrow_measurement(
        self, corpus: CorpusStores
    ) -> None:
        """Not merely down-weighted. It is not in the requester's result set at all."""
        restricted = corpus.sources.find_document_by_uri(corpus.tenant_id, RESTRICTED_URI)
        assert restricted is not None

        narrow_outcome = _retrieve(corpus, BROAD_QUERY, PermissionScope())
        wide_outcome = _retrieve(corpus, BROAD_QUERY, PermissionScope.of(PRIVACY_GRANT))

        narrow_uris = {r.candidate.document.uri for r in narrow_outcome.results}
        wide_uris = {r.candidate.document.uri for r in wide_outcome.results}
        assert RESTRICTED_URI not in narrow_uris
        assert RESTRICTED_URI in wide_uris


# ---------------------------------------------------------------------------
# The sample corpus, measured
# ---------------------------------------------------------------------------


class TestTheSampleCorpus:
    """The corpus-level measurement, and a prediction it falsified.

    ``docs/STATUS.md`` recorded, at the end of the judge slice, that corroboration on this
    corpus "should be low, because lexical and vector reach the same source versions and the
    strategy gate is designed to say so."

    **That was wrong about which number carries the signal.** Corroboration counts *distinct
    source versions*, so a broad query returning spans from seven documents saturates it at
    1.0 and the strategy gate never binds. The dependence between the two strategies is real
    and it lands on ``retrieval_agreement`` instead, at 0.875-1.0 - which the current formula
    scores as a *positive*. Recorded in ``docs/EVALUATION.md`` and filed as an open issue,
    because it is a calibration problem rather than a measurement one and the weights are
    frozen for this slice.
    """

    def test_a_broad_query_saturates_corroboration(self, corpus: CorpusStores) -> None:
        assembled = _measure_query(corpus, BROAD_QUERY, PermissionScope())

        assert assembled.coverage.distinct_versions >= 3
        assert assembled.inputs.corroboration_count >= 3
        # Saturation: the factor reaches 1.0 and stops discriminating.
        score = compute_confidence(
            assembled.inputs, consequence_id="cq_" + "7" * 32, computed_at=AS_OF
        )
        assert score.factor_inputs["corroboration"] == 1.0

    def test_a_single_hit_query_is_the_genuinely_low_corroboration_case(
        self, corpus: CorpusStores
    ) -> None:
        """One document, one version. This is what low corroboration actually looks like."""
        assembled = _measure_query(corpus, NARROW_QUERY, PermissionScope())

        assert assembled.inputs.corroboration_count == 1
        score = compute_confidence(
            assembled.inputs, consequence_id="cq_" + "7" * 32, computed_at=AS_OF
        )
        assert score.factor_inputs["corroboration"] == pytest.approx(1 / 3)
        assert score.confidence < 0.5

    def test_the_two_strategies_are_near_duplicates_on_this_corpus(
        self, corpus: CorpusStores
    ) -> None:
        """The real low-independence signal, and it is not on the corroboration factor.

        ``HashingEmbedder`` is feature hashing over a bag of words, so the vector strategy
        reaches nearly the same spans as lexical. Agreement is therefore high - which the
        formula rewards, and which on this embedder means redundancy rather than reliability.
        """
        assembled = _measure_query(corpus, BROAD_QUERY, PermissionScope())

        assert assembled.inputs.retrieval_agreement >= 0.85
        assert assembled.inputs.corroborating_strategies == ("lexical", "vector")

    def test_every_source_version_in_the_corpus_is_dated(self, corpus: CorpusStores) -> None:
        """So freshness is measurable here, and a zero would mean old rather than unknown."""
        assembled = _measure_query(corpus, BROAD_QUERY, PermissionScope())

        assert assembled.coverage.versions_dated == assembled.coverage.distinct_versions
        assert "source_freshness" not in assembled.coverage.unmeasured
        assert assembled.inputs.source_freshness > 0.7

    def test_the_measurement_is_reproducible(self, corpus: CorpusStores) -> None:
        """Requirement 7, end to end over the real corpus."""
        first = _measure_query(corpus, BROAD_QUERY, PermissionScope())
        second = _measure_query(corpus, BROAD_QUERY, PermissionScope())

        assert first.inputs == second.inputs
        assert _confidence(first) == _confidence(second)

    def test_explicitness_is_the_one_factor_this_path_cannot_reach(
        self, corpus: CorpusStores
    ) -> None:
        """Retrieval returns spans, not bindings, so a retrieval-only caller measures no
        explicitness. Reported rather than defaulted, so the zero is not read as 'implied'."""
        assembled = _measure_query(corpus, BROAD_QUERY, PermissionScope())

        assert assembled.coverage.unmeasured == ("explicitness",)


# ---------------------------------------------------------------------------
# Corpus fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> CorpusStores:
    """The sample corpus, with the retention policy restricted to the privacy grant.

    The overlay matters: without a document only some requesters may read, the authorization
    tests would run over a corpus where nothing is ever withheld and would prove nothing.
    """
    stores = asyncio.run(load_corpus())
    assert not stores.extraction_failures, stores.extraction_failures
    document = stores.sources.find_document_by_uri(stores.tenant_id, RESTRICTED_URI)
    assert document is not None
    stores.sources.put_document(document.model_copy(update={"permissions": (PRIVACY_GRANT,)}))
    return stores


def _service(corpus: CorpusStores) -> RetrievalService:
    use_case = RetrieveEvidenceUseCase(sources=corpus.sources, assumptions=corpus.assumptions)
    return RetrievalService(retrieval=use_case, embedder=HashingEmbedder())


def _retrieve(
    corpus: CorpusStores,
    text: str,
    scope: PermissionScope,
    strategy: RetrievalStrategy = RetrievalStrategy.LEXICAL,
) -> RetrievalOutcome:
    query = RetrievalQuery(tenant_id=corpus.tenant_id, text=text, permission_scope=scope, top_k=10)
    finalized = asyncio.run(
        _service(corpus).retrieve(RetrievalRequest(query=query, strategy=strategy))
    )
    return finalized.outcome


def _measure_query(corpus: CorpusStores, text: str, scope: PermissionScope) -> Any:
    """Run both strategies for real, then measure the lexical result set as the support."""
    lexical = _retrieve(corpus, text, scope, RetrievalStrategy.LEXICAL)
    vector = _retrieve(corpus, text, scope, RetrievalStrategy.VECTOR)
    return assemble(
        supporting_evidence_ids=list(lexical.evidence_ids),
        outcomes=(lexical, vector),
        consequence_class=CLASS_A,
        tenant_id=corpus.tenant_id,
        as_of=AS_OF,
    )


def _confidence(assembled: Any) -> float:
    return compute_confidence(
        assembled.inputs, consequence_id="cq_" + "7" * 32, computed_at=AS_OF
    ).confidence
