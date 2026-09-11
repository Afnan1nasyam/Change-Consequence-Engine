# Security Requirements

## Threat model

CCE may process internal enterprise knowledge. Assume:
- documents can contain secrets
- documents can be mutually untrusted
- retrieved content can contain prompt injection
- users may have different access rights
- model providers may receive sensitive context

## Requirements

1. Treat source content as data, not instructions.
2. Sanitize and tag retrieved content.
3. Never let retrieved text change agent permissions.
4. Enforce authorization before retrieval output enters an agent.
5. Do not log raw sensitive documents by default.
6. Redact credentials, tokens, and secrets during ingestion.
7. Track source access in audit logs.
8. Use tenant/project filters in every retrieval path.
9. Prevent cross-tenant vector searches.
10. Validate all external connector credentials.
11. Use least privilege.
12. Keep model configuration out of source code.
13. Store prompt versions for auditability.
14. Add prompt-injection test scenarios to evaluation.

## Prompt injection defense

Retrieved content must be wrapped as untrusted evidence.

Agents must follow instructions from their system prompt and runtime contract, not instructions found inside source documents.

The assumption extractor is the highest-value injection target in the system: a document
that induces a false assumption binding corrupts the coupling index for every future
analysis of that corpus, not just the current request. Mitigations:

- extractor output is schema-constrained; free-form text cannot become a binding
- every AssumptionRecord must carry an `evidence_id` whose span is verified to exist in
  the source, mechanically, after extraction
- assumption keys are constrained to the **closed AssumptionKeyRegistry**; an unregistered
  key is quarantined as a `KeyRegistryGap` and never indexed. There is no fallback path, so
  an injected document cannot introduce a new coupling key at all
- a proposed `KeyAlias` cannot create Class A coupling. Only human approval can, so an
  injected document cannot manufacture a key equivalence
- a single source cannot manufacture a **cross-source** contradiction. Two divergent
  declarations *within* one source version are a genuine intra-document inconsistency and
  are reported as such - not suppressed, but never presented as two artifacts disagreeing.
  Implemented in `app/assumptions/collisions.py`: the pair is emitted as an
  `intra_document_divergence` anomaly rather than a `conflicts_with` relation, so it never
  reaches the Class A tier as a governance conflict between independent authorities
- an injected document cannot fabricate authority: `authority_class` comes from source
  registration, not from document content, so a planted file cannot claim policy authority
  and override a real policy

## Multi-provider security

Provider pluralism widens the data-egress surface. Therefore:

- the set of enabled providers is explicit configuration, never inferred from which API
  keys happen to be present
- each provider records what it received, at what granularity, for audit
- provider credentials come only from environment or a secret manager
- a local provider (Ollama) is the only correct choice for corpora that may not leave the
  network boundary, and the configuration must make that an enforceable constraint rather
  than a convention
- failover between providers must not silently move restricted content to a provider the
  tenant did not approve
