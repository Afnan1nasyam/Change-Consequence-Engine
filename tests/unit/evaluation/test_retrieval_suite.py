"""The suite schema and the shipped label file, without loading a corpus.

Two jobs. The first is that a malformed label is a **load** error and not a silently wrong
measurement: a query naming an undeclared scope must not run under some default
authorization, and a mistyped tenant must not quietly measure recall over an empty corpus.
The second is that the shipped suite covers the query shapes the slice was asked to
measure, asserted by reading the file rather than by trusting the reviewer who wrote it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from app.domain.interfaces.retrieval import MAX_TOP_K
from app.domain.value_objects.enums import SourceType
from app.evaluation.retrieval_benchmark import (
    MEASURE_TOP_K,
    SUITE_PATH,
    BenchmarkQuery,
    RetrievalSuite,
    load_suite,
)

TENANT = "tn_" + "0" * 31 + "1"

#: The categories the task named. Asserted to exist so a later edit cannot quietly drop the
#: shape of query that motivated the benchmark.
REQUIRED_CATEGORIES = {
    "exact_terminology",
    "numeric_value",
    "scoped_concept",
    "version_currency",
    "paraphrase",
    "permission_restricted",
    "known_lexical_miss",
    "coverage_gap",
    "negative_control",
}


def suite_data(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": "test-1",
        "retrieval_config_version": "retrieval-1",
        "scorer_version": "scorer-x",
        "tenants": {"primary": TENANT},
        "scopes": {"open": [], "privacy": ["privacy-office"]},
        "queries": [
            {
                "id": "q_one",
                "category": "exact_terminology",
                "text": "manager approval",
                "tenant": "primary",
                "permission_scope": "open",
            }
        ],
    }
    base.update(over)
    return base


@pytest.fixture(scope="module")
def shipped() -> RetrievalSuite:
    return load_suite()


# ---------------------------------------------------------------------------
# The schema refuses what it cannot measure
# ---------------------------------------------------------------------------


class TestSuiteValidation:
    def test_a_minimal_suite_loads(self) -> None:
        assert RetrievalSuite.model_validate(suite_data()).queries[0].id == "q_one"

    def test_an_undeclared_permission_scope_is_a_load_error(self) -> None:
        """The failure mode this prevents: a typo running under the wrong authorization."""
        data = suite_data()
        data["queries"][0]["permission_scope"] = "priavcy"
        with pytest.raises(ValidationError, match="undeclared scope"):
            RetrievalSuite.model_validate(data)

    def test_an_undeclared_tenant_is_a_load_error(self) -> None:
        data = suite_data()
        data["queries"][0]["tenant"] = "secondary"
        with pytest.raises(ValidationError, match="undeclared tenant"):
            RetrievalSuite.model_validate(data)

    def test_a_mistyped_tenant_id_is_a_load_error(self) -> None:
        """Otherwise it measures recall over a corpus nobody owns and reports zero."""
        with pytest.raises(ValidationError):
            RetrievalSuite.model_validate(suite_data(tenants={"primary": "tenant-1"}))

    def test_a_duplicate_query_id_is_a_load_error(self) -> None:
        data = suite_data()
        data["queries"] = [data["queries"][0], dict(data["queries"][0])]
        with pytest.raises(ValidationError, match="duplicate query id"):
            RetrievalSuite.model_validate(data)

    def test_an_unparseable_binding_scope_is_a_load_error(self) -> None:
        data = suite_data()
        data["queries"][0]["scope"] = "region"
        with pytest.raises(ValidationError):
            RetrievalSuite.model_validate(data)

    def test_an_unknown_field_is_refused(self) -> None:
        """``extra="forbid"`` throughout: a misspelled key must not be ignored label data."""
        data = suite_data()
        data["queries"][0]["expected"] = []
        with pytest.raises(ValidationError):
            RetrievalSuite.model_validate(data)

    def test_an_expectation_needs_both_a_source_and_a_quote(self) -> None:
        data = suite_data()
        data["queries"][0]["expect"] = [{"source": "policy/a.md"}]
        with pytest.raises(ValidationError):
            RetrievalSuite.model_validate(data)

    def test_a_suite_with_no_queries_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalSuite.model_validate(suite_data(queries=[]))


class TestScopesAndTenants:
    def test_the_empty_scope_is_the_least_privileged_one_and_not_a_wildcard(self) -> None:
        suite = RetrievalSuite.model_validate(suite_data())
        assert suite.scope("open").grants == ()
        assert suite.scope("open").permits(()) is True
        assert suite.scope("open").permits(("privacy-office",)) is False

    def test_a_named_scope_carries_its_grants(self) -> None:
        suite = RetrievalSuite.model_validate(suite_data())
        assert suite.scope("privacy").permits(("privacy-office",)) is True

    def test_all_grants_unions_the_scopes_and_the_document_labels(self) -> None:
        """The classifier holds these, because a benchmark has to be able to see the span
        it withheld in order to report that it withheld it."""
        suite = RetrievalSuite.model_validate(
            suite_data(document_labels=[{"source": "policy/a.md", "labels": ["hr-only"]}])
        )
        assert suite.all_grants == ("hr-only", "privacy-office")

    def test_document_labels_may_not_be_empty(self) -> None:
        """An empty label set restricts nothing, so asking for one is asking for no ACL."""
        with pytest.raises(ValidationError):
            RetrievalSuite.model_validate(
                suite_data(document_labels=[{"source": "policy/a.md", "labels": []}])
            )


class TestResolveQuery:
    def test_a_labelled_query_becomes_the_production_contract(self) -> None:
        suite = RetrievalSuite.model_validate(
            suite_data(
                queries=[
                    {
                        "id": "q_two",
                        "category": "scoped_concept",
                        "text": "personal data",
                        "tenant": "primary",
                        "permission_scope": "privacy",
                        "as_of": "2026-01-01",
                        "scope": "region:EU",
                        "source_types": ["policy"],
                    }
                ]
            )
        )
        resolved = suite.resolve_query(suite.queries[0])

        assert resolved.tenant_id == TENANT
        assert resolved.text == "personal data"
        assert resolved.permission_scope.grants == ("privacy-office",)
        assert resolved.as_of == dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        # Case-folded by the scope parser, so a label may be written the way a person
        # writes it and still be the same scope the corpus bound.
        assert resolved.scope is not None and resolved.scope.render() == "region:eu"
        assert resolved.source_types == (SourceType.POLICY,)

    def test_every_query_runs_at_the_measurement_top_k(self) -> None:
        """Recall@k is derived by truncating one full ranking, which is exact because the
        ordering is total - and it lets a miss be reported as *shares no term* rather than
        merely *ranked below k*."""
        suite = RetrievalSuite.model_validate(suite_data())
        assert MEASURE_TOP_K == MAX_TOP_K
        assert suite.resolve_query(suite.queries[0]).top_k == MEASURE_TOP_K

    def test_an_absent_as_of_stays_absent_rather_than_becoming_today(self) -> None:
        """A benchmark whose temporal filter moved with the wall clock would not be
        reproducible."""
        suite = RetrievalSuite.model_validate(suite_data())
        assert suite.queries[0].as_of is None
        assert suite.resolve_query(suite.queries[0]).as_of is None


# ---------------------------------------------------------------------------
# The shipped suite
# ---------------------------------------------------------------------------


class TestShippedSuite:
    def test_it_loads(self, shipped: RetrievalSuite) -> None:
        assert SUITE_PATH.exists()
        assert shipped.version == "retrieval-benchmark-1"

    def test_it_is_a_small_labelled_suite(self, shipped: RetrievalSuite) -> None:
        assert len(shipped.queries) >= 30
        assert sum(len(q.expect) for q in shipped.queries) >= 50

    def test_it_covers_every_query_shape_the_slice_had_to_measure(
        self, shipped: RetrievalSuite
    ) -> None:
        categories = {q.category for q in shipped.queries}
        assert categories >= REQUIRED_CATEGORIES, sorted(REQUIRED_CATEGORIES - categories)

    def test_every_query_names_a_tenant_and_a_permission_scope(
        self, shipped: RetrievalSuite
    ) -> None:
        """Neither has a default, here or in the production contract."""
        assert all(q.tenant and q.permission_scope for q in shipped.queries)

    def test_it_probes_more_than_one_tenant_and_more_than_one_scope(
        self, shipped: RetrievalSuite
    ) -> None:
        assert len({q.tenant for q in shipped.queries}) > 1
        assert len({q.permission_scope for q in shipped.queries}) > 1

    def test_it_restricts_at_least_one_document(self, shipped: RetrievalSuite) -> None:
        """The sample corpus ships unlabelled, so authorization recall is unmeasurable
        without the suite's own ACL overlay."""
        assert shipped.document_labels

    def test_a_query_with_no_expectations_is_declared_as_a_control(
        self, shipped: RetrievalSuite
    ) -> None:
        """Otherwise an expectation lost in an edit would look like an intended control."""
        unlabelled = [q for q in shipped.queries if not q.expect]
        assert unlabelled
        assert all(q.category in {"negative_control", "tenant_isolation"} for q in unlabelled)

    def test_the_temporal_queries_are_the_ones_carrying_an_as_of(
        self, shipped: RetrievalSuite
    ) -> None:
        """As-of is set where a temporal question is asked and nowhere else, so the report
        can say which figures a date filter touched."""
        dated = [q for q in shipped.queries if q.as_of is not None]
        assert dated
        assert all(q.category == "version_currency" for q in dated)

    def test_the_version_currency_queries_come_in_dated_pairs(
        self, shipped: RetrievalSuite
    ) -> None:
        """One question, two dates. A single date could not show a stale version being
        excluded rather than merely absent."""
        dated = [q for q in shipped.queries if q.category == "version_currency"]
        by_text: dict[str, list[BenchmarkQuery]] = {}
        for query in dated:
            by_text.setdefault(query.text, []).append(query)
        assert by_text
        for text, pair in by_text.items():
            assert len({q.as_of for q in pair}) == len(pair) >= 2, text

    def test_the_permission_pair_differs_only_in_the_grant_held(
        self, shipped: RetrievalSuite
    ) -> None:
        """The A/B that isolates authorization: same text, same labels, different scope."""
        by_id = {q.id: q for q in shipped.queries}
        withheld = by_id["restricted_retention_least_privilege"]
        granted = by_id["restricted_retention_privacy_reader"]

        assert withheld.text == granted.text
        assert withheld.expect == granted.expect
        assert withheld.permission_scope != granted.permission_scope

    def test_the_numeric_pair_differs_only_in_a_digit_separator(
        self, shipped: RetrievalSuite
    ) -> None:
        by_id = {q.id: q for q in shipped.queries}
        separated = by_id["numeric_refund_threshold_with_separator"]
        plain = by_id["numeric_refund_threshold_without_separator"]

        assert separated.text.replace(",", "") == plain.text
        assert separated.expect == plain.expect

    def test_every_expectation_names_a_corpus_path_and_a_quote(
        self, shipped: RetrievalSuite
    ) -> None:
        """Never an ``evidence_id``: an opaque id cannot be reviewed, and a stale one would
        be indistinguishable from evidence that was never admitted."""
        for query in shipped.queries:
            for expected in query.expect:
                assert not expected.source.startswith("ev_")
                assert expected.source.endswith((".md", ".json"))
                assert expected.quote.strip() == expected.quote
