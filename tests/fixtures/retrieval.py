"""Builders for retrieval candidates.

A leaf module: it imports only ``tests.fixtures.ids`` and production code, so it cannot take
part in the fixture import cycle that ``ids.py`` exists to prevent.

:func:`candidate` builds one internally consistent span - evidence, version and document all
agreeing - from a text and whatever metadata a test cares about, so a negative test can
change exactly one thing and attribute the failure to it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.domain.entities.source import EvidenceItem, SourceDocument, SourceVersion
from app.domain.interfaces.retrieval import EvidenceCandidate
from app.domain.value_objects.enums import AuthorityClass, SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from tests.fixtures.ids import SHA, make_id

TENANT = make_id("tenant", 1)
OTHER_TENANT = make_id("tenant", 2)

#: A date inside every fixture's validity interval unless a test moves it.
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

#: The least-privileged scope: it reads unlabelled documents and nothing else. Used as the
#: default in these fixtures so a test that means to exercise authorization has to say so.
OPEN_SCOPE = PermissionScope()


def candidate(
    text: str,
    *,
    n: int = 1,
    tenant_id: str = TENANT,
    source_type: SourceType = SourceType.POLICY,
    authority_class: AuthorityClass = AuthorityClass.POLICY,
    scopes: tuple[str, ...] = ("global",),
    assumption_keys: tuple[str, ...] = ("refund.approval_threshold",),
    effective_from: dt.datetime | None = None,
    effective_to: dt.datetime | None = None,
    title: str = "Refund Approval Policy",
    permissions: tuple[str, ...] = (),
    **over: Any,
) -> EvidenceCandidate:
    """One retrievable span whose evidence, version and document agree.

    ``n`` seeds every id, so two candidates built with different ``n`` are distinct and two
    built with the same ``n`` collide - which is what a deduplication test needs.
    """
    source_id = make_id("source", n)
    version_id = make_id("source_version", n)

    return EvidenceCandidate(
        evidence=EvidenceItem(
            evidence_id=make_id("evidence", n),
            source_version_id=version_id,
            chunk_id=make_id("chunk", n),
            text_span=text,
            char_start=0,
            char_end=max(len(text), 1),
        ),
        document=SourceDocument(
            source_id=source_id,
            tenant_id=tenant_id,
            source_type=source_type,
            title=title,
            uri=f"corpus://fixture/{n}.md",
            authority_class=authority_class,
            permissions=permissions,
            current_version_id=version_id,
        ),
        version=SourceVersion(
            version_id=version_id,
            source_id=source_id,
            content_hash=SHA,
            version_label="v1",
            effective_from=effective_from,
            effective_to=effective_to,
        ),
        scopes=tuple(ScopeSelector.parse(scope) for scope in scopes),
        assumption_keys=assumption_keys,
        **over,
    )
