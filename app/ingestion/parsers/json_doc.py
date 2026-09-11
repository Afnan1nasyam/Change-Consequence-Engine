"""JSON parser.

JSON is not prose, so a span quoted from raw JSON ("24") is useless to a reviewer and
ambiguous to the span verifier. This parser renders the document into a deterministic
flat form instead:

```
auth.reset_token.ttl_hours: 24
auth.reset_token.single_use: true
```

That rendering **is** the canonical text for the version: offsets index into it, evidence
spans quote it, and the content hash is taken over it. Two consequences, both intended:

* whitespace and key order in the original file do not create a new source version, which
  is what makes re-ingesting a reformatted config idempotent;
* an evidence citation reads as a path and a value, which is what a reviewer needs to
  confirm the binding by hand.

The original bytes are still hashed separately (``raw_sha256``) so the artifact in object
storage remains verifiable against what was fetched.
"""

from __future__ import annotations

import json
from typing import Any

from app.domain.value_objects.enums import SourceType
from app.ingestion.identity import PARSER_VERSION, bytes_hash, canonical_text
from app.ingestion.parsers.base import ParsedDocument, ParseError, Section, decode

_MAX_DEPTH = 32


class JsonParser:
    name = "json"

    def supports(self, *, source_type: SourceType, suffix: str) -> bool:
        return suffix == ".json"

    def parse(self, payload: bytes) -> ParsedDocument:
        try:
            loaded = json.loads(decode(payload))
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}") from exc

        lines = list(_flatten(loaded, path=()))
        if not lines:
            raise ParseError("JSON document contains no leaf values")

        text_parts: list[str] = []
        sections: list[Section] = []
        cursor = 0
        for pointer, rendered in lines:
            line = f"{pointer}: {rendered}\n"
            text_parts.append(line)
            sections.append(
                Section(
                    path=tuple(pointer.split(".")),
                    char_start=cursor,
                    char_end=cursor + len(line),
                    level=pointer.count(".") + 1,
                )
            )
            cursor += len(line)

        text = canonical_text("".join(text_parts))
        return ParsedDocument(
            text=text,
            sections=tuple(sections),
            metadata={"raw_sha256": bytes_hash(payload), "leaf_count": str(len(lines))},
            parser_name=self.name,
            parser_version=PARSER_VERSION,
        )


def _flatten(node: Any, *, path: tuple[str, ...], depth: int = 0) -> list[tuple[str, str]]:
    """Depth-first, key-sorted flattening to ``(dotted.path, rendered_value)`` pairs.

    Sorting is what makes the rendering canonical. Depth is bounded so that a pathological
    or hostile document cannot exhaust the stack during ingestion.
    """
    if depth > _MAX_DEPTH:
        raise ParseError(f"JSON nesting exceeds {_MAX_DEPTH} levels at {'.'.join(path)}")

    if isinstance(node, dict):
        out: list[tuple[str, str]] = []
        for key in sorted(str(k) for k in node):
            out.extend(_flatten(node[key], path=(*path, key), depth=depth + 1))
        return out

    if isinstance(node, list):
        out = []
        for index, item in enumerate(node):
            out.extend(_flatten(item, path=(*path, str(index)), depth=depth + 1))
        return out

    return [(".".join(path) or "(root)", _render_leaf(node))]


def _render_leaf(value: Any) -> str:
    """Render a scalar the way the source declared it.

    ``json.dumps`` is used for strings so that an embedded newline or quote cannot break
    the one-leaf-per-line invariant that offsets depend on.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(str(value))
