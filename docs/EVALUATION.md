# CCE Evaluation Plan

## Objective

Demonstrate that multi-agent evidence arbitration finds more correct hidden consequences with fewer unsupported claims than simpler approaches.

## Experimental conditions

Baseline A — Search-assisted analyst:
retrieve evidence, one LLM call, answer.

Baseline B — Single-agent RAG:
one agent retrieves and reasons.

Baseline C — Single-agent + self-critique:
initial answer, critique, revision.

Baseline D — Multi-agent without challenge:
specialists + synthesis.

System E — Full CCE:
specialists + consequence hunter + contradiction hunter + challenger + blind spot + judge.

## Scenario construction

Each scenario includes:
- proposed change
- authoritative source
- stale source
- contradictory source
- irrelevant source
- direct consequence
- hidden consequence
- historical analogue
- one or more unresolved questions
- expected assumption keys and their coupling sets

Avoid leaking benchmark labels into the corpus.

### The corpus, as built

`data/sample_corpus/` holds 13 synthetic organizational documents. Registration metadata
(tenant, source type, **authority class**, owner, effective interval) lives in
`manifest.yaml` beside them, because authority must come from registration and never from
document content. Expectations live separately in
`data/benchmarks/ingestion_expectations.yaml`, and nothing under `app/` loads that file, so
a gold label cannot be read by the system under test.

The corpus deliberately contains:

| shape | where |
|---|---|
| authoritative declaration | refund policy: `refund.approval_threshold = 10,000 USD @global` |
| consumer that agrees with its authority | training deck repeating 10,000 as `consumes`; silent until the threshold moves |
| same-scope contradiction | payments standard declaring `7,500 USD @global` |
| legitimate scoped override | EU data-protection policy, `regulatory`, 30d @`region:EU` |
| unauthorized scoped override | EMEA team wiki, `wiki`, 45d @`region:EU` |
| **standing stale consumer** | CLV metric definition `consumes retention.window = 365d` where the policy declares 90d |
| **equal-authority ambiguity** | two `sop_runbook` peers declaring `sla.p1_response_time` = 15 and 30 min |
| cross-medium coupling with no cross-reference | IdP JSON config and the reset runbook, joined only by `auth.reset_token.ttl` |
| historical narration | INC-4471, bound as `describes`, which never invalidates |
| historical analogue with a recorded outcome | INC-4471's observed queue-depth tripling |
| unregistered key | `refund.escalation_ceiling` in the wiki |
| irrelevant document (negative control) | Dublin office-move FAQ, which must bind nothing |
| prompt injection | vendor notes demanding `regulatory` authority, a new registry key, an alias approval and a confidence number |
| embedded credentials | sandbox keys in the same vendor notes |

Two documents were added when the coupling resolver landed, because nothing in the original
eleven produced the corresponding case:

- **the standing stale consumer.** Every consumer in the original corpus agreed with its
  governing declarer, so the `declares`/`consumes` staleness rule only fired on a delta.
  Section A of the knowledge-decay report needs a divergence that exists with **no change
  request at all**, and the CLV metric definition supplies it.
- **the equal-authority ambiguity, deliberately with the newer document disagreeing.** The
  incident-response runbook is effective 2025-09-01; the password-reset runbook it
  contradicts is effective 2025-02-01. An implementation with a newest-wins fallback would
  resolve to the newer 30-minute value and pass every other check in the suite. This pair
  is the corpus-level guard against exactly that, and `resolve_authority` must answer
  `Ambiguous`.

Resolver expectations live in `data/benchmarks/coupling_expectations.yaml`, separate from
the ingestion labels, and are read only by tests.

Labelled retrieval queries live in `data/benchmarks/retrieval_queries.yaml`, again separate.
That file is read by `app/evaluation/retrieval_benchmark.py` — the harness, not the system
under test — and the separation is structural rather than promised:
`tests/contract/test_layering.py` restricts `app/domain`, `app/ingestion`, `app/assumptions`,
`app/retrieval` and `app/application` to importing only each other, so no module on the
retrieval path can import `app.evaluation` and therefore none can reach a label file.

### Replay extraction

`app/evaluation/replay.py` substitutes recorded proposals
(`data/fixtures/extraction_proposals.json`) for the model call. Everything else — parsers,
chunker, identity, registry, admission gate, stores — is real.

This exists because a comparison that varies the registry version, an alias approval or the
chunker cannot be read if the extractor output moves at the same time. Replay holds
extraction fixed so the deterministic layer is the only variable. It is also how the
ingestion path runs in CI with no credentials and no network.

Recorded proposals carry a **quote, not a chunk ordinal**: the ordinal is resolved by
locating the quote, so a fixture stays valid across a chunker change and cannot smuggle in
offsets the real extractor would never supply. A recorded quote that no longer matches the
corpus is reported as an extraction failure, not skipped — otherwise a drifted fixture would
look like a recall problem in the gate.

### Three anti-flattery rules

The benchmark is synthetic and authored by the same people who designed the ontology.
Without the following, recall degenerates into "did the pipeline find what we planted for
it, in the categories we planted it in".

1. **Ontology separation.** Scenarios are authored before the relationship taxonomy and
   assumption key namespace are frozen, ideally by someone who has not read them.
2. **Mandatory negative controls, ~25% of scenarios.** Changes whose correct answer is
   "little or nothing downstream". A system that always finds twelve consequences is
   useless, and only negative controls can detect that.
3. **Corpus-ablation variants.** See below.

### Statistical honesty

30-50 scenarios cannot separate five experimental conditions under human judging.
Differences of a few points are noise and must not be reported as wins.

Therefore:
- report confidence intervals, never bare point estimates
- for the headline claim (E beats D), power the comparison on that pair alone rather than
  spreading scenarios across all five conditions
- treat the initial 30-50 scenario suite as a development harness, and gate any published
  comparison on a separately authored held-out set

## Corpus ablation (primary evidence-grounding test)

For each scenario, produce a variant with the document that supports a known consequence
removed.

Pass condition: the system moves that consequence to `unresolved` or `speculative` and
reports corpus insufficiency. Fail condition: it still asserts the consequence, which
means the finding was coming from model priors rather than from the corpus.

This single test demonstrates evidence-grounding more convincingly than the entire
baseline table, and it directly exercises the failure behavior in `IDEA.md` 17.

### Ablation of a deterministic finding, as implemented

For Class A the pass condition is sharper than "degrades to `unresolved`", and stating it
loosely would let the test pass for the wrong reason. Removing the data-retention policy
from the corpus leaves the vendor wiki page as the only global declarer of
`retention.window`, so a staleness finding against the CLV metric definition *still exists*
— correctly, because the corpus still contains a governing declarer that disagrees with it.

What must be true is that **nothing survives unchanged**:

- no finding derived from the removed document retains its `finding_id`;
- no finding cites any evidence span belonging to the removed document;
- the surviving staleness finding names a **different** authority.

A finding whose identity and evidence are identical after its supporting document is
deleted was not derived from that document at all. All three conditions are asserted in
`tests/evaluation/test_class_a_benchmark.py`.

## Metrics

### Seeded recall (sanity metric)
Fraction of gold consequences surfaced. This checks that the pipeline is not broken. It is
**not** the product metric, because the gold set was authored with the system's own
ontology in mind.

### Unseeded materiality (product metric)
Human-judged usefulness of findings that are correct but absent from the gold set.

These must **not** be scored as false positives. Scoring them as errors punishes exactly
the capability the product claims - finding consequences nobody thought to plant - and
will drive the system toward conservative uselessness. Precision is computed only over
findings judged incorrect, not over findings judged unlisted.

### Consequence precision
Fraction of final material consequence findings judged correct, where "incorrect" means
the mechanism does not hold - not merely "not in the gold set".

### Class A vs Class B breakdown
All precision and recall metrics are reported separately for assumption-mediated (Class A)
and behavioral (Class B) consequences. They have different mechanisms, different expected
reliability, and pooling them hides the interesting result.

### Hidden consequence recall
Fraction of seeded non-explicit consequences surfaced.

### Coupling recall (mechanism metric)
Fraction of expected assumption keys correctly extracted, and fraction of expected
coupling sets correctly reconstructed. This isolates the core mechanism from the agent
layer: if coupling recall is low, no amount of agent tuning will help.

### Contradiction detection, deterministic vs inferred
Reported separately for collisions found by key comparison and those found by an agent.
The first should approach 1.0. If it does not, the fault is in extraction or in registry
coverage, and the `KeyRegistryGap` volume per domain localizes which.
A key normalizer is deliberately excluded from the MVP, so there is nothing to retune
there: canonical form is declared per registry entry (`DOMAIN_CHECKPOINT.md` 4.1.5).

### Deterministic resolver metrics, as measured

Scored by `tests/evaluation/test_class_a_benchmark.py` against
`data/benchmarks/coupling_expectations.yaml`, over the real corpus with **zero live model
calls**.

| metric | definition | threshold | measured |
|---|---|---|---|
| Class A recall | seeded `must_find` cases the resolver produced | 1.0 | **1.0** (4/4) |
| false Class A rate | reported `(artifact, rule)` pairs the labels forbid | 0.0 | **0.0** |
| unauthorized-override precision | flagged overrides that are labelled defects | 1.0 | **1.0** |
| contradiction precision | same-scope conflicts that are labelled conflicts | 1.0 | **1.0** |
| stale-consumer detection | labelled stale consumers found | 1.0 | **1.0** |
| authority ambiguity rate | keys where authority cannot be established | reported | 1 of 9 keys |
| `class_b_rediscovery_rate` | Class B hypotheses suppressed as Class A duplicates | reported | n/a (no agentic layer yet) |
| model calls in the Class A path | — | 0 | **0** |

**The false-positive metric is scored per `(artifact, rule)`, not per artifact**, and the
distinction matters. The EU data-protection policy must never be flagged as an
*unauthorized override* — but it is legitimately reported as a same-scope contradiction
against the EMEA wiki page. Scoring by artifact alone would count that true positive as a
false one, and the metric would then punish correct behaviour.

`class_b_rediscovery_rate` is wired and tested but has nothing to measure until the agentic
layer exists. When it does, a **high** value is a warning that the hunter's evidence slice
is misconfigured — not evidence that it is being productive.

### Mutation testing of the resolver

A green suite proves nothing unless the assertions are wired to the logic, so eight
plausible wrong implementations were introduced one at a time and the suite had to fail on
each: a newest-wins fallback in `resolve_authority`, scope containment conferring override
rights without the authority test, a unit mismatch weakening instead of blocking, key-only
(scope-blind) coupling, `describes` bindings participating in collisions, pairwise instead
of authority-based staleness, adjacent intervals treated as overlapping, and legitimate
overrides reported as findings. **All eight were caught.**

### Retrieval recall, as measured

Scored by `app/evaluation/retrieval_benchmark.py` against
`data/benchmarks/retrieval_queries.yaml`, over the real corpus, with **zero model calls**
and no network. Reproduce with `make retrieval-benchmark`.

This measures the **lexical retriever only** — Okapi BM25 over admitted evidence spans,
with no stemmer, no lemmatizer, no synonym table, no query expansion, no fuzzy matching, no
vector search and no reranking. It exists before the second strategy on purpose: a fusion
weight tuned against no number is not a decision.

Three scorers have been measured against it, and each change after the first was made
*because* of a number this suite produced:

| scorer | change | recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|---|
| `tokens-1` | as the benchmark first found the retriever | 33.9% | 62.5% | 78.6% | 80.4% |
| `tokens-2` | stopword list (`docs/STATUS.md` 32) | 35.7% | 67.9% | 80.4% | 80.4% |
| **`tokens-3`** | **guarded plural rule (`docs/STATUS.md` 26)** | **35.7%** | **75.0%** | **89.3%** | **91.1%** |

Every figure in this section is `tokens-3` unless it says otherwise. The two tuning changes
are written up in "The stopword list, measured" and "The plural rule, measured" below;
`k1=1.2` / `b=0.75` have never been touched.

#### The suite

35 queries, 62 expectations, over the 13-document corpus and its 21 admitted spans.

An expectation names a **source path and a quote**, never an `evidence_id`, and the runner
resolves the pair to the span containing it. An opaque id cannot be reviewed, and — the
decisive reason — a stale id would be indistinguishable from a span that was never
admitted, which is the central distinction below. A quote must occur exactly once in its
document.

`expect` is a **relevance judgment about the question**. It is not a prediction of what the
retriever will do and not a statement about who may read the evidence, which is what lets
one expectation be a hit under a scope holding a grant and a correct withholding under one
that does not, with the label unchanged.

Each query specifies its text, tenant, permission scope, an as-of date where a temporal
question is being asked, its expected evidence, and a category. Categories, with their
expectation counts:

| category | queries | expectations | what it measures |
|---|---|---|---|
| `exact_terminology` | 4 | 5 | the corpus's own words |
| `numeric_value` | 4 | 10 | thresholds, with and without a digit separator |
| `assumption_key_lookup` | 2 | 11 | can text search reconstruct a coupling set? |
| `paraphrase` | 3 | 7 | the question a person actually types |
| `known_lexical_miss` | 2 | 5 | the recorded singular/plural and vocabulary defects |
| `scoped_concept` | 2 | 4 | scope intersection, and a disjoint sibling |
| `version_currency` | 4 | 7 | one question at two dates, stale vs in force |
| `permission_restricted` | 4 | 5 | the same label under two scopes |
| `coverage_gap` | 4 | 4 | relevant text no strategy can reach |
| `cross_medium` | 1 | 2 | a JSON config and a prose runbook |
| `source_type_filter` | 1 | 2 | a filter that selects without re-weighting |
| `negative_control` | 3 | 0 | the correct answer is nothing |
| `tenant_isolation` | 1 | 0 | a foreign tenant reaches nothing |

The suite ships its own **ACL overlay**, because the sample corpus carries no permission
labels and authorization recall cannot be measured against a corpus that has none. It
declares one label on the retention policy and the runner applies it through the store
port, which is what an operator applying an ACL does. `manifest.yaml` is untouched:
registration metadata is operator input.

#### Reachability of the expected evidence

The denominator is the whole point. An expected span can fail to arrive for three reasons
with three different owners, and pooling them produces a number that measures nothing:

| | count | |
|---|---|---|
| labelled expectations | 62 | |
| **retrievable** | **56** | the recall denominator |
| never admitted | 4 | no verified binding covers the text; no strategy can reach it |
| withheld by authorization | 2 | relevant, unreadable by that requester; correct behaviour |
| removed by a query filter | 0 | non-zero here would mean a label and a query disagree |
| label defects | 0 | |
| **coverage** | **90.3%** | of the labelled evidence, what retrieval can reach at all |

#### Recall

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **recall** (of reachable evidence) | **35.7%** | **75.0%** | **89.3%** | **91.1%** |
| end-to-end recall (of all labelled evidence) | 32.3% | 67.7% | 80.6% | 82.3% |
| mean per query (reachable) | 53.5% | 85.5% | 94.1% | 94.8% |

Both denominators are reported, always. Quoting only the first flatters the retriever;
quoting only the second blames it for the ingestion boundary. The pooled and per-query
figures differ because queries carry unequal numbers of labels — the two
`assumption_key_lookup` queries hold 11 of the 62 expectations between them and drag the
pooled number down.

| category | @1 | @3 | @5 | @10 | coverage | @10 under `tokens-2` |
|---|---|---|---|---|---|---|
| `exact_terminology` | 80.0% | 100% | 100% | 100% | 100% | 100% |
| `numeric_value` | 20.0% | 60.0% | 100% | 100% | 100% | 100% |
| `cross_medium` | 50.0% | 100% | 100% | 100% | 100% | 100% |
| `version_currency` | 57.1% | 100% | 100% | 100% | 100% | 100% |
| `scoped_concept` | 50.0% | 100% | 100% | 100% | 100% | 100% |
| `source_type_filter` | 50.0% | 100% | 100% | 100% | 100% | 100% |
| `permission_restricted` | 66.7% | 100% | 100% | 100% | 60.0% | 100% |
| `paraphrase` | 14.3% | 85.7% | 100% | 100% | 100% | 71.4% |
| `known_lexical_miss` | 20.0% | 60.0% | 80.0% | 80.0% | 100% | 20.0% |
| `assumption_key_lookup` | 18.2% | 36.4% | 54.5% | 63.6% | 100% | 54.5% |
| `coverage_gap` | — | — | — | — | **0.0%** | — |
| `negative_control`, `tenant_isolation` | — | — | — | — | — | — |

A dash is a ratio over an empty denominator and is reported as unknown rather than as zero.
A negative-control category scoring "0.0%" would read as a failure when emptiness is the
intended answer.

#### Authorization-filtered recall

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall under a restricting scope | 50.0% | 100% | 100% | 100% |

Computed over the 3 queries where the scope actually removed something — read from the
trace, not from the scope, because a scope that happens to read everything is not a
restricted run. 2 expectations were withheld and 2 were reachable; `@1` is 50% because one
of those two queries has two relevant spans and only one can occupy rank 1.

**Withheld evidence returned anyway: 0.** The only metric here whose acceptable value is
fixed in advance.

The substantive result is a comparison, and it was not a foregone conclusion. Corpus
statistics are recomputed over the authorized slice, so a restricted requester scores every
span differently by construction. Run on the identical query with and without the grant
(`exact_manager_approval_before_release` / `restricted_refund_threshold_least_privilege`),
the evidence the requester is entitled to comes back **at the same ranks**: the different
statistics did not reorder it. On this corpus, authorization costs nothing in recall for
what the requester may read.

#### Negative controls

4 queries whose correct answer is nothing. **3 silent, 5 results returned** — down from
2 silent and 24 results under `tokens-1`.

The survivor is "pack your personal items into the crate", which returns 5 spans, and it
survives for a reason no stopword list can address: it shares the *content* words
`personal` (three retention spans) and `items` ("at most 100 items per page") with the
corpus. That is content-word ambiguity, and it is the floor for this class of fix rather
than evidence of an unfinished list. Reported as a number instead of being tuned away.

#### The stopword list, measured

The one change made because of a figure this suite produced. `docs/STATUS.md` 32 recorded
that two of four negative controls returned results on `the`, `is` and `with` alone; the
tokenizer now drops a closed-class list of function words from **both** the query and the
span (`app/retrieval/tokens.py`, `TOKENIZER_VERSION = tokens-2`).

| | `tokens-1` | `tokens-2` | |
|---|---|---|---|
| recall@1 | 33.9% | **35.7%** | +1.8 |
| recall@3 | 62.5% | **67.9%** | +5.4 |
| recall@5 | 78.6% | **80.4%** | +1.8 |
| recall@10 | 80.4% | **80.4%** | — |
| end-to-end recall@1 / @3 / @5 | 30.6 / 56.5 / 71.0% | **32.3 / 61.3 / 72.6%** | |
| mean per query @3 | 74.8% | **79.5%** | +4.7 |
| authorization-filtered recall (all *k*) | 50 / 100 / 100 / 100% | 50 / 100 / 100 / 100% | — |
| negative controls silent | 2 of 4 | **3 of 4** | +1 |
| negative-control results returned | 24 | **5** | −19 |
| expectations retrieved / missed | 45 / 11 | 45 / 11 | — |
| coverage | 90.3% | 90.3% | — |

**The last two rows are the point.** Recall rose at 1, 3 and 5, was unchanged at 10, and the
outcome counts are identical: the same 45 expectations come back and the same 11 are missed.
Nothing was found that was not found before and nothing was lost — every gain is an
expectation that was already arriving, arriving *higher up*, because spans can no longer earn
score for matching `the`. A change that traded recall for precision would look completely
different here, and that is why the outcome counts are asserted alongside the recall figures
in `tests/evaluation/test_retrieval_benchmark.py`.

Per-query, the value is mostly in results that stopped coming back — which a recall suite
cannot see, so it is pinned individually:

| query | results returned |
|---|---|
| `paraphrase_password_reset_link_validity` | 15 → 3 |
| `numeric_retention_ninety_days` | 15 → 8 |
| `negative_coffee_machine` | 13 → 0 |
| `negative_pack_personal_items` | 11 → 5 |
| `paraphrase_how_long_we_keep_customer_data` | 12 → 8 |
| `coverage_refund_billing_defect_exception` | 11 → 7 |
| `paraphrase_who_signs_off_large_refunds` | 7 → 2 |
| `exact_revenue_close_lag` | 5 → 1 |

Two ranking effects worth naming, one good and one not:

- **The governing policy now outranks the unofficial wiki.** Asked "how long do we keep
  customer data once an account is closed", `tokens-1` ranked the EMEA wiki page — an
  unauthorized override at `wiki` authority — 2nd and the governing retention policy 3rd.
  Under `tokens-2` the policy is 1st and the wiki 4th. Retrieval does not know about
  authority and must not; this is a side effect of function words no longer contributing
  score, not a feature. But an agent reads results in order, so it is pinned by a test.
- **The two equal-authority peers swapped places.** In `as_of_p1_target_2026_01` the
  15-minute runbook was 1st and the 30-minute one 2nd; they are now reversed. Both are
  returned either way, so no recall figure moves, and neither order is *correct* — they are
  peers at the same authority class and `resolve_authority` answers `Ambiguous` for exactly
  this pair. Recorded rather than presented as an improvement.

**The list was chosen by grammatical class, not by measuring which words caused the noise**,
and that is the methodological point rather than the contents. A list assembled from this
suite's own failures would be fitted to 35 queries over 13 documents, and the improvement it
produced would be a description of the fitting. It is the closed-class English function
vocabulary — articles, coordinators, common prepositions, personal and possessive pronouns,
the auxiliary forms of *be*/*do*/*have*, and the wh-words — assembled without reference to
the corpus.

**Four classes a standard English stopword list removes are deliberately kept**, because
each changes what an organizational sentence commits to: **negation** (`not`, `without` — "a
verbal approval is *not* sufficient"), **comparatives and bounds** (`more`, `above` —
"10,000 USD *or more*"), **quantifiers** (`all`, `every` — "*every* refund requires manager
approval"), and **modals** (`may`, `must`). Dropping negation is the best-known defect of the
off-the-shelf lists and it is not repeated here. `tests/unit/retrieval/test_tokens.py`
asserts the boundary word by word, so a later "just use the standard list" edit fails
loudly.

The list is part of `TOKENIZER_VERSION`, which is part of the scorer fingerprint the suite
pins — so the version guard fired on this change, and the figures above were re-measured
rather than reprinted. That is the guard doing its job on its first real test.

What the stopword list did **not** change: coverage, the unreachable count, the withheld
count, authorization-filtered recall, the leak count, `known_lexical_miss` (still 20% flat),
and `assumption_key_lookup` (still 54.5% @10). None of those failures are caused by function
words — which is what pointed at morphology as the next thing to try.

#### The plural rule, measured

The second change the suite has caused, aimed at `docs/STATUS.md` 26: `refund` did not
reach a policy writing `Refunds`, and `known_lexical_miss` recalled 20% flat across every
*k* — the signature of a vocabulary mismatch rather than a ranking problem, and the stopword
list had moved it by exactly zero.

**The rule, in full.** In `app/retrieval/tokens.py`, `depluralize()` removes **one** trailing
`s` from a token when all four conditions hold:

1. the token is **all letters** — so no quantity can reach the suffix logic at all;
2. it ends in `s`;
3. it does **not** end in `ss`, `us` or `is`;
4. the remaining stem is at least **3** characters.

Applied inside `tokenize()`, after case folding and the stopword filter, so it runs over the
query and the span through the same function and cannot be applied to one side only.

This is **not a stemmer and must not become one**. There is no `-ies` rule, no irregular
plural handling, no verb inflection, no lemmatizer, no synonyms and no fuzzy matching. A
Porter stemmer would fix more plurals and also conflate `operate` with `operation`, which
in this corpus are different claims.

**The guard principle: an ending is guarded when a trailing `s` there is never an English
plural marker.** No plural ends in `ss` (`access`, `address`, `business`, `class`, `loss`,
and the retrieval term `less`); `-us` words are singulars whose Latin plural is `-i`
(`status`, `corpus`, `previous`); the plural of `-is` is `-es`, so `analysis`, `basis` and
`axis` are never plurals. `-as` is deliberately **not** guarded, because there both classes
are real — `alias` and `bias` are singulars but `areas` and `ideas` are plurals — so a guard
would buy one and sell the other. The rule keeps the commoner plural reading and accepts
`alias → alia` as collateral: self-consistent on both sides of every match, and colliding
with no other term.

**Checked against the corpus before it was written.** Over all 736 distinct tokens in the
full canonical text of the 13 documents plus every benchmark query and quote, the rule
changes 99 tokens and produces 40 groups where two raw tokens collapse onto one stem.
**All 40 are genuine singular/plural pairs** — `refund`/`refunds`, `day`/`days`,
`customer`/`customers`, `record`/`records`, and so on. Not one unrelated merge. The words
the guards left whole are exactly `access`, `address`, `basis`, `business`, `class`,
`corpus`, `loss`, `previous`, `regardless`, `status`, `unless`.

| | `tokens-2` | `tokens-3` | |
|---|---|---|---|
| recall@1 | 35.7% | 35.7% | — |
| recall@3 | 67.9% | **75.0%** | +7.1 |
| recall@5 | 80.4% | **89.3%** | +8.9 |
| recall@10 | 80.4% | **91.1%** | +10.7 |
| end-to-end recall@3 / @5 / @10 | 61.3 / 72.6 / 72.6% | **67.7 / 80.6 / 82.3%** | |
| mean per query @10 | 86.1% | **94.8%** | +8.7 |
| `known_lexical_miss` @3 / @5 / @10 | 20 / 20 / 20% | **60 / 80 / 80%** | +40 / +60 / +60 |
| `paraphrase` @3 / @10 | 57.1 / 71.4% | **85.7 / 100%** | +28.6 / +28.6 |
| `assumption_key_lookup` @10 | 54.5% | **63.6%** | +9.1 |
| expectations retrieved | 45 | **51** | +6 |
| expectations missed | 11 | **5** | −6 |
| negative controls silent | 3 of 4 | 3 of 4 | — |
| negative-control results returned | 5 | 5 | — |
| coverage · withheld · leaks | 90.3% / 2 / 0 | 90.3% / 2 / 0 | — |
| authorization-filtered recall | 50 / 100 / 100 / 100% | 50 / 100 / 100 / 100% | — |

**This is a different shape of change from the stopword list.** That one moved evidence up
without altering what came back; this one alters what comes back — six labelled spans that
were reachable and missed are now retrieved, and the miss count fell from 11 to 5.
Crucially, **the negative-control figures are byte-identical**: widening what matches did
not buy recall by making the retriever noisier. Coverage, the withheld count and the leak
count are unchanged, as a tokenizer change must leave them.

The six newly retrieved expectations, all of them plural/singular mismatches:

| query | was | now |
|---|---|---|
| `lexical_miss_refund_singular` | 3 misses of 3 | ranks 2, 3, 5 |
| `paraphrase_who_signs_off_large_refunds` | 2 misses of 2 | ranks 3, 5 |
| `key_refund_approval_threshold` | training deck missed | rank 6 |

**A correction to what this document previously claimed.** The `tokens-2` write-up recorded
"who has to sign off on a large refund" as a paraphrase sharing *no content word* with the
spans that answer it, and called it the clearest case for vector retrieval. That was wrong:
the spans write "Refunds", the query writes "refund", and the plural rule alone recovered
both expectations. The genuinely semantic gap is narrower than it looked — one query,
`lexical_miss_token_expiry_wording`, where the runbook says "the link is valid for 24 hours"
and shares no word *form* with "token expiry" under any inflection.

**One clear ranking regression, and it undoes a stopword-era gain.** Asked "how long do we
keep customer data once an account is closed", `tokens-1` ranked the EMEA wiki page — an
unauthorized override at `wiki` authority — 2nd and the governing retention policy 3rd; the
stopword list made it policy 1st, wiki 4th; the plural rule reverses that to wiki 1st,
policy 2nd, because the wiki span writes "customers" and now matches the query's "customer".
Every expectation is still returned, so no recall figure moves — this is purely the order an
agent reads them in. The aggregate is not one-sided: on the refund-threshold queries the
same effect promoted the governing policy from 3rd to 2nd and the training deck from 5th to
3rd while the wiki fell from 2nd to 4th. Retrieval does not know about authority and must
not; ordering evidence by authority belongs to the consumer, not the tokenizer. Both
directions are pinned by tests so the trade stays visible.

**Known limitations, recorded rather than fixed:** `policies` does not reach `policy`
(`-ies`); `analyses` does not reach `analysis` (irregular); `retained` does not reach
`retain` (verb inflection); `access` does not reach `accesses` (the guard is
one-directional); and `apis`/`kpis` are caught by the `-is` guard, which cannot tell them
from `basis`. Rescuing the last would need a length exception on the guard, which would then
strip `axis` — a real protection traded for a marginal gain.

#### The four findings that matter

1. **Typing an assumption key does not reconstruct its coupling set.** `retention.window`
   is bound by five artifacts. Only one spells the key out — the injected vendor wiki page —
   and the prose that governs writes "retained for 90 days", sharing no term with the key.
   So the query returns the **least** trustworthy member of the coupling set at rank 1 and
   misses the governing policy entirely: 1 of 5, and the plural rule did not change that.
   `refund.approval_threshold` reaches 6 of 6 under `tokens-3` — up from 5 — but only
   because the words "refund" and "approval" happen to appear in its spans. This is the
   empirical case for resolving coupling through the closed registry and never through text
   similarity, and for Class A recall not depending on retrieval at all.
2. **A paraphrase can be unreachable for either of two reasons, and they were confused.**
   Under `tokens-2`, "who has to sign off on a large refund" returned neither of the 2 spans
   that answer it, and this document called it the clearest case for vector retrieval. It
   was a **morphological** gap, not a semantic one: the spans write "Refunds", the query
   writes "refund", and the plural rule recovered both. What is left is one query —
   `lexical_miss_token_expiry_wording`, "the link is valid for 24 hours" against "token
   expiry" — with no shared word form under any inflection. That is the real vector case,
   and it is one query rather than three. Correcting this downward is the more useful
   result: two of the three "semantic" failures were spelling.
3. ~~**A singular query term does not reach its plural.**~~ **Fixed** in `tokens-3` and
   measured: `known_lexical_miss` went from 20% flat across every *k* to 60 / 80 / 80% at
   @3 / @5 / @10, with no cost in negative-control noise. See "The plural rule, measured".
   The residual 20% is `lexical_miss_token_expiry_wording`, which is finding 2 rather than
   a morphology problem.
4. **Coverage is the ceiling, and it is not retrieval's to raise.** Four labelled spans are
   unreachable because only binding-bearing spans are admitted (`docs/STATUS.md` 25). One is
   a material policy exception — "a refund arising from a confirmed billing defect may be
   released without manager approval" — which is exactly what an approval question most
   needs. Another is blocked by registry coverage rather than by extraction: the EMEA
   escalation ceiling proposes an unregistered key, so it was quarantined. Registry
   coverage is a retrieval ceiling, which is not obvious until it is measured.

### Vector retrieval, as measured

The second strategy, scored against the **same 62 expectations** by
`app/evaluation/retrieval_benchmark.py:run_vector_suite`. Reproduce with
`make retrieval-benchmark-vector`, or
`python -m scripts.retrieval_benchmark --strategy vector`.

**The comparison is legitimate because the labels are strategy-independent.** An expectation
is a relevance judgment about the *question*, not a prediction about any retriever, so
nothing in the suite was changed for this run. Coverage, the unreachable set and the withheld
set come out identical under both strategies, and that is asserted rather than assumed — if
a change of strategy had moved any of them, the recall difference below would be partly a
difference of denominators and would mean nothing.

**Read every number here with the embedder in mind.** They were produced by
`HashingEmbedder`: feature hashing over a bag of words, offline, deterministic, and carrying
**no semantic knowledge at all**. Two texts are close only if they share literal tokens. So
these figures measure the vector *machinery* — the index, the filters, the ordering, the
provenance — plus a term-overlap signal expressed as cosine. **They are not a measurement of
semantic retrieval.** A real Gemini recording has now been completed separately; its results
are described below, but the benchmark scorer still identifies the vector run as
`hashing-1:d=256`, so those benchmark numbers are not presented as Gemini semantic-retrieval
results.

#### Results

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **vector recall** (of the 56 reachable) | **30.4%** | **60.7%** | **76.8%** | **80.4%** |
| vector end-to-end recall (of all 62) | 27.4% | 54.8% | 69.4% | 72.6% |
| vector mean per query | 46.1% | 72.8% | 83.1% | 86.1% |
| vector authorization-filtered recall | 50.0% | 100% | 100% | 100% |

| | |
|---|---|
| unreachable expectations (never admitted) | **4** |
| withheld expectations (authorization) | **2** |
| withheld evidence returned anyway | **0** |
| coverage | **90.3%** |
| expectations retrieved / missed | 46 / 10 |
| negative controls: silent / results returned | 1 of 4 / **28** |

#### Lexical vs vector

The lexical column is the frozen `tokens-3` baseline. Nothing about lexical retrieval was
touched to produce this table, and nothing was tuned in response to it.

| | lexical (`tokens-3`) | vector (`hashing-1`) | |
|---|---|---|---|
| recall@1 | **35.7%** | 30.4% | lexical +5.3 |
| recall@3 | **75.0%** | 60.7% | lexical +14.3 |
| recall@5 | **89.3%** | 76.8% | lexical +12.5 |
| recall@10 | **91.1%** | 80.4% | lexical +10.7 |
| authorization-filtered recall (all *k*) | 50 / 100 / 100 / 100% | 50 / 100 / 100 / 100% | identical |
| coverage · unreachable · withheld · leaks | 90.3% / 4 / 2 / 0 | 90.3% / 4 / 2 / 0 | identical |
| expectations retrieved | **51** | 46 | |
| negative-control results | **5** | 28 | lexical far quieter |

**Lexical wins at every *k*, and the interesting result is sharper than that: the vector
strategy finds a strict subset of what lexical finds.**

| | count |
|---|---|
| reachable expectations | 56 |
| found by both | 46 |
| found by lexical only | 5 |
| **found by vector only** | **0** |
| found by neither | 5 |

A second strategy earns its place by reaching evidence the first cannot. By that test this
embedder earns nothing, so **there is no case for fusing them yet** — and building hybrid
fusion now would mean building it against a measurement that says not to.

Three things worth separating in that result:

1. **Why lexical wins is preprocessing, not ranking.** All five lexical-only finds are
   singular/plural pairs the `tokens-3` suffix rule merges (`lexical_miss_refund_singular`
   ×3, `paraphrase_who_signs_off_large_refunds`, `key_refund_approval_threshold`). The
   hashing embedder has no suffix rule — deliberately, since it must not share preprocessing
   with the lexical strategy — so `refund` and `Refunds` land in different coordinates.
   `known_lexical_miss` recall@10 is 20% against lexical's 80% for exactly that reason. This
   is not evidence that cosine is a worse ranking function.
2. **Paraphrase recall got worse, which contradicts the reason the slice was scheduled.**
   Paraphrase was the motivation for a vector strategy, and it measures 71.4% @10 against
   lexical's 100%. An embedder with no semantics cannot help a paraphrase. Measuring that is
   worth more than having assumed the opposite.
3. **The strategies agree wherever a filter or literal wording decides the answer.**
   `exact_terminology`, `numeric_value`, `cross_medium`, `scoped_concept`,
   `version_currency`, `source_type_filter` and `permission_restricted` all reach 100% @10
   under both. That agreement is the evidence that the vector machinery is wired correctly
   rather than merely returning something.

**The five expectations neither strategy reaches are the same five.** Four are the coupling
set of `retention.window` and one is `lexical_miss_token_expiry_wording`. A second,
independent strategy failing on the same coupling set is stronger evidence for the closed
registry than one strategy failing was: text similarity of any kind does not reconstruct a
coupling set.

#### Cosine has no natural cutoff, and it costs precision

The sharpest cost of the strategy: **28 results across the four negative controls against
lexical's 5, and one silent run against three.** Lexical retrieval is quiet because a span
sharing no term scores exactly nothing; here a span sharing one hashed coordinate scores
something, and something is enough to be returned.

A similarity floor would fix it and would be an untuned knob, so it is measured rather than
pre-empted by a threshold nobody justified. Note also that the *quietness this embedder does
have* is an artefact: feature hashing is sparse, so an unrelated text lands in no shared
coordinate and scores exactly 0.0. A real dense embedder has no zeros and would return
`top_k` results for every question, including ones the corpus cannot answer. **The measured
control figures here must not be read as evidence about how a real embedder behaves.**

#### What the vector benchmark deliberately does not claim

- **It is not a semantic measurement.** This subsection is the offline `HashingEmbedder`
  baseline. A real Gemini recording has since been completed, but the current scorer still
  labels the vector benchmark as `hashing-1:d=256`, so the benchmark figures below are not
  presented as a measured Gemini semantic-retrieval result.
- **No hybrid fusion, no reranking, no query expansion.** The two strategies are scored
  separately and never merged in this baseline. Each is pinned to its own scorer version.
- **Nothing was tuned on either side.** No similarity floor, no dimension search, no
  re-tuning of `k1`/`b`, no change to the tokenizer or the labels.
- **The index is rebuilt and re-embedded per query.** Honest for 21 spans and badly wrong
  for a real corpus, where embedding the whole tenant per query would dominate cost and
  latency. It is a correctness-first implementation, recorded as the largest open issue in
  the slice.

#### Real Gemini embedding recording

A real Gemini embedding run was completed after the offline vector baseline. The recording
script sent the same 51 distinct texts used by the retrieval benchmark: 21 admitted evidence
spans and 35 benchmark queries. Gemini returned all 51 vectors in 4.6 seconds.

The run used `gemini` with model `gemini-embedding-001` at 768 dimensions. The generated fixture
records the provider, model, dimensions, adapter version, recording date and text digests; the
API key is not written to the fixture.

The recording proves that the provider configuration, credential, request path and replay
fixture path all work. It does **not**, by itself, justify calling the existing vector benchmark
a Gemini semantic-retrieval benchmark. The subsequent benchmark output still reports the scorer
as `vector-cosine-1:hashing-1:d=256`, the same scorer used for the offline vector baseline.
Because of that mismatch, the benchmark recall figures remain attributed to the frozen
HashingEmbedder run rather than being relabelled as Gemini results.

The practical v1 decision is therefore unchanged: keep BM25 as the default retrieval strategy.
Gemini embeddings remain an available provider and a validated recording path, but a clean
provider-specific semantic benchmark should be treated as a separate follow-up once the scorer
and replay path expose the real provider identity consistently.

### Hybrid retrieval, as measured

Reciprocal rank fusion over the two strategies, scored against the **same 62 expectations**
by `app/evaluation/retrieval_benchmark.py:run_hybrid_suite`. Reproduce with
`make retrieval-benchmark-hybrid`, or
`python -m scripts.retrieval_benchmark --strategy hybrid`.

Fusion is `score(d) = Σ 1/(k + rank_s(d))` at `k = 60` — the paper's value, untuned here. It
fuses *ranks*, never scores, because BM25 and cosine are on incomparable scales and any
score-level combination would need a normalization nobody has measured.

**The vector input is still `HashingEmbedder`**, which carries no semantic knowledge. So this
measures fusion over a strong lexical ranking and a weak term-overlap one — not what fusion
would do with a real embedder.

#### Results

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **hybrid recall** (of the 56 reachable) | **33.9%** | **66.1%** | **83.9%** | **91.1%** |
| lexical baseline (`tokens-3`, frozen) | **35.7%** | **75.0%** | **89.3%** | **91.1%** |
| vector baseline (`hashing-1`, frozen) | 30.4% | 60.7% | 76.8% | 80.4% |
| hybrid end-to-end recall (of all 62) | 30.6% | 59.7% | 75.8% | 82.3% |
| hybrid authorization-filtered recall | 50.0% | 100% | 100% | 100% |

| | hybrid | lexical | vector |
|---|---|---|---|
| unreachable (never admitted) | 4 | 4 | 4 |
| withheld by authorization | 2 | 2 | 2 |
| withheld evidence returned anyway | **0** | 0 | 0 |
| coverage | 90.3% | 90.3% | 90.3% |
| expectations retrieved / missed | 51 / 5 | 51 / 5 | 46 / 10 |
| negative controls silent / results | 1 of 4 / **28** | 3 of 4 / 5 | 1 of 4 / 28 |

#### Strategy contribution

Over the 260 results the 35 fused queries returned in total:

| | count | share |
|---|---|---|
| found by **both** | 143 | 55.0% |
| **lexical-only** | 13 | 5.0% |
| **vector-only** | 104 | 40.0% |
| mean per-query agreement | | 57.2% |

#### The result: fusion recovered the lexical set and ranked it worse

**Hybrid finds exactly what lexical finds** — the same 51 hits, the same 5 misses, the same
set. Fusion recovered the strong strategy's coverage and added nothing. But it *orders* that
set worse: recall falls 1.8 points at @1, **8.9 points at @3**, 5.4 at @5, and only ties at
@10.

The mechanism is in the contribution table. The vector input supplies **104 vector-only
results against lexical's 13** — eight times the volume — while contributing no labelled
evidence lexical missed. Those results occupy ranks, and RRF gives every one of them a
positive contribution, so relevant lexical results are pushed down. RRF is behaving
correctly: it beats the weak input at every *k*, which is what it is for. Beating the *strong*
input requires the second input to contribute something, and this one does not.

Three further observations:

- **Negative-control noise is inherited wholesale.** 28 results across the four controls,
  identical to the vector input and against lexical's 5. A union cannot be quieter than its
  noisiest member, and fusion adds no floor of its own.
- **Every guarantee survived.** Coverage, the unreachable count, the withheld count, the
  authorization-filtered recall and the leak count are identical across all three runs.
  Fusion inherits its filtering rather than re-implementing it, and the measurement confirms
  that rather than merely asserting it.
- **The same five expectations are still missed** by all three strategies — four are the
  coupling set of `retention.window`, one is the token-expiry wording. A third strategy built
  from the same two failing on the same five is stronger evidence for the closed registry
  than either failing alone.

#### What the hybrid benchmark deliberately does not claim

- **This is a verdict on the frozen lexical + HashingEmbedder combination, not on a semantic
  hybrid.** A real Gemini embedding recording has been completed, but the hybrid benchmark
  below was run with the existing `HashingEmbedder` vector side.
- **`k = 60` was not tuned.** A larger `k` flattens both lists and would reduce the damage
  the weak input does at the top; searching it now would be fitting a fusion parameter to
  compensate for the current vector baseline.
- **No weighting was tried.** Down-weighting the vector input would improve these numbers
  and would be exactly the wrong lesson to draw from one measurement over 62 labels.
- **Precision is still not scored.** The contribution split and the negative controls are
  the only precision-adjacent signals, and the split is the one that explains the result.

#### What the benchmark deliberately does not claim

- **No parameter is fitted to it.** `k1=1.2` / `b=0.75` remain the untuned defaults. Two
  changes have been made because of it — the stopword list and the plural rule — and each
  was specified by a linguistic principle rather than read off the suite's failures: a
  closed-class function vocabulary, and "guard the endings where a trailing `s` is never a
  plural marker". A word list assembled from the 24 noisy results, or a guard list
  assembled from this corpus's `-as` words, would each have scored better and meant less.
  The plural rule was also validated against the *full* corpus vocabulary (736 tokens),
  not only the 62 labelled expectations, so its safety claim does not rest on the labels.
- **Precision is not scored.** The labels are recall labels; a span that came back
  unlabelled is not thereby wrong. The negative controls are the only precision-adjacent
  signal here, and they are reported as raw counts. This understated the stopword change —
  most of its value was 19 fewer results on questions the corpus does not answer — and it
  is why the per-query result counts are pinned by name. For the plural rule the same gap
  cuts the other way: a rule that widened matching *could* have bought recall with noise,
  and the only reason to believe it did not is that the control counts are unchanged.
- **A ranking change is not a recall change, and the suite mostly cannot see one.** The
  plural rule reordered several result lists without moving any recall figure, including
  one case where an unauthorized wiki override rose above the policy that governs it. Those
  are pinned individually because no aggregate here would report them.
- **The suite is authored by the same people who built the retriever**, over a synthetic
  corpus, and it will flatter the design in the ways `docs/EVALUATION.md` already warns
  about. 62 expectations cannot separate two retrieval strategies; they can show that one
  of them fails on a shape of query, which is what these do.
- **Authorization recall rests on a one-document overlay**, not on a realistic label
  distribution. `docs/STATUS.md` 30 stands.

### Confidence inputs, as measured

`app/judgment/confidence_inputs.py` measures the eight checkpoint 9.1 factors from pipeline
data. The formula and weights in `app/domain/services/confidence.py` are unchanged by this
measurement; only the sources are fixed.

| factor | measured from |
|---|---|
| `source_authority` | mean normalized `AUTHORITY_RANK` of the supporting spans' holder documents |
| `source_freshness` | mean of `1 - age/5y` over distinct supporting `SourceVersion.effective_from` |
| `explicitness` | mean normalized `EXPLICITNESS_RANK` of the candidate's own bindings |
| `corroborating_source_version_ids` | distinct `source_version_id` among the supporting spans |
| `corroborating_strategies` | strategies whose outcome actually returned a supporting span |
| `retrieval_agreement` | share of supporting spans **every** executed strategy returned |
| `contradiction_strength` | strongest *other* divergence on the same key, by class weight |
| `inference_depth` | 1 for a Class A derivation, 2 for a Class B hypothesis |
| `alias_mediated` | caller-supplied; nothing records it yet |

#### Measured on the sample corpus

Both strategies executed (`lexical`, `vector` over `HashingEmbedder`), `as_of` 2026-01-01,
`top_k` 10, unrestricted scope, lexical result set taken as the support.

| query | spans | distinct versions | authority | freshness | agreement | confidence |
|---|---|---|---|---|---|---|
| `refunds require manager approval` | 7 | 6 | 0.540 | 0.804 | 1.000 | **0.557** |
| `personal data is retained for 90 days after the account is closed` | 8 | 7 | 0.667 | 0.824 | 0.875 | **0.576** |
| `token expiry` | 1 | 1 | 0.556 | 0.883 | 1.000 | **0.401** |

Every source version in the manifest states `effective_from`, so freshness is genuinely
measured here rather than defaulted - a 0.0 on this corpus would mean *old*, not *unknown*.
The one factor a retrieval-only caller cannot reach is `explicitness`: retrieval returns spans,
not bindings, so `FactorCoverage.unmeasured` reports it rather than letting the zero read as
"implied".

#### A prediction this falsified

`docs/STATUS.md` recorded, at the end of the judge slice, that corroboration on this corpus
"should be low, because lexical and vector reach the same source versions and the strategy gate
is designed to say so."

**That was wrong about which factor carries the signal.** Corroboration counts *distinct source
versions*, and it saturates at three - so a broad query returning spans from six or seven
documents scores 1.0 and the strategy gate never binds. The dependence between the two
strategies is real, and it lands on `retrieval_agreement` at **0.875-1.000** instead.

That relocation matters, because the two factors point opposite ways. The current formula
scores high agreement as *positive* (retrieval reliability, +). On this corpus high agreement
means the opposite: `HashingEmbedder` is feature hashing over a bag of words, so the vector
strategy reaches nearly the same spans as lexical, and their agreement is redundancy rather
than independent confirmation. Recorded as an open issue rather than fixed, because the weights
are frozen and re-weighting against one uncalibrated embedder would be tuning to an artefact.

The genuinely low-corroboration case is visible in the same table: `token expiry` reaches one
span in one document, scores corroboration 1/3, and lands at 0.401 confidence - which is the
factor discriminating correctly when there is something to discriminate.

#### Authorization effect, measured

Same query, same corpus, the retention policy restricted to a `privacy-office` grant:

| scope | distinct versions | authority | confidence |
|---|---|---|---|
| unrestricted requester | 6 | 0.593 | **0.489** |
| `privacy-office` holder | 7 | 0.667 | **0.502** |

A withheld span never reaches the outcome, so it cannot enter the measurement: the narrower
scope sees one fewer source version and lower mean authority. **A narrower scope can never
score higher**, which is a property of the assembler reading only what the retrievers returned
rather than a rule it applies.

#### What is deliberately unmeasurable

`ConfidenceInputs` has no field for an agent, an agent count, a model, or a level of agreement
between agents, and `assemble` has no parameter for one. Checkpoint 9.3 is therefore enforced by
the data not existing at that boundary: the function *cannot* count agents, so it cannot be
talked into treating their agreement as corroboration. Several specialists reading one span
remain one source.

### End-to-end pipeline, as measured

`tests/integration/test_analysis_pipeline.py`, over the real sample corpus with the real
`LlmGateway` and a fake provider. No credentials, no network, no prompt or threshold tuned to
make any of it pass.

The fake agent (`tests/fixtures/fake_agents.py`) **reads its own prompt** and answers at the
strongest position the schema allows for every handle and every evidence id it was offered. It
is therefore the *adversarial* case for every gate: it cites everything it is given and claims
`likely` for everything. Anything that comes out weaker than that was weakened by the gates,
and the difference between a positive and a negative control comes from what the pipeline put
in the prompt rather than from a script.

#### Positive control - a change to `retention.window`

Twelve stages, with the blind-spot hunter, the evidence arbiter and the report composer wired.

| | |
|---|---|
| stages | 13 ran, 0 failed, 0 skipped (`evidence_gaps` empty) |
| Class A findings | **9** (3 `delta_invalidation`, 4 `same_scope_contradiction`, 1 `stale_consumer`, 1 `unauthorized_override`) |
| retrieved spans | 6 |
| Class B hypotheses | 1 from the main hunt, **1 from the blind-spot partition** |
| consequences | 11 (9 Class A `likely`, 2 Class B `speculative`) |
| **material** | **9** |
| confidence | 0.199 - 0.384 |
| graph | 24 nodes, 27 edges, 5 `retention.window` holders |
| report | 9 foregrounded, 2 reachable, 7 evidence references, 2 unresolved questions |

The two Class B consequences cap at `speculative` although the fake judge asked for `likely`:
neither has a historical analogue, and the frozen gate lowered both. Every Class A consequence
carries `likely` because the evidence is real and admitted.

The fake hunter also *claims* a well-formed analogue id here, and it appears in
`fabricated_analogue_ids` rather than in the finding. That is the interesting half of this
control: the corpus records no measured outcome for `retention.window`, so nothing can anchor,
and a model asserting otherwise changes nothing.

**Blind-spot partition.** 5 spans, in the `single_holder` and `unwritten_rule` slices; **10
spans the partition queries returned were dropped because the main hunter had already been
shown them.** `unqueried_source_type` and `low_ranked` came out empty and are reported that way
- the measurement that forced the third partition query, since those two slices are reachable
only through the change's own words. Yield: **1 finding, credited to `single_holder`**, resting
on 2 evidence ids disjoint from the main analysis, plus **1 corpus-silence finding** (a
question and a recommended source, never a consequence).

**Arbitration.** 11 rulings, all `supported`, **0 clamped** - and **all 11 had a corroboration
claim trimmed**, because the cooperative fake reports the measured source versions plus one
that no corpus contains. Nothing was bought by the overclaim: the ceiling reads the measured
count, which was 2 for every candidate.

**Report composer.** The fake proposes the eligible handles in reverse plus one handle it was
never offered; the unoffered handle is dropped (`refused_handles = ('r99',)`) and the reversal
is honoured. Three sections are empty and named: `historical_warnings`, `evidence_gaps`,
`reviewer_signals`.

#### Anchored control - a change with a historical precedent

A change to `refund.approval_threshold` (10,000 → 5,000 USD). The previous reduction of the
same threshold is recorded in INC-4471 **with measured figures**, which makes this the only
change this corpus can anchor.

| | |
|---|---|
| stages | 13 ran, 0 failed, 0 skipped (`evidence_gaps` empty) |
| Class A findings | **10** (6 `conflicts_with`, 4 `invalidates_assumption`) |
| retrieved spans | 9 |
| specialists | **5 of 7** invoked (policy 3, process 4, people 1, risk 1, historical 1) |
| historical events | 3 proposed → **1 admitted, and it qualifies as an anchor** |
| Class B hypotheses | 1 main hunt, 1 blind-spot partition |
| consequences | 12 (11 `likely`, 1 `speculative`) |
| **material** | **11** |
| graph | 25 nodes, 27 edges |

**The main-hunt Class B consequence is judged `likely` carrying
`historical_analogue_id = he_8cce2b46…`** - the first non-speculative Class B finding the
project has produced. It reached that status through the existing frozen gates and nothing
else: checkpoint 9.2 rule 2 stopped firing because the precedent exists, with no new
confidence input, no new scoring and no new status.

The blind-spot Class B consequence in the **same run** stays `speculative` with no analogue.
That contrast is what makes the first number worth anything: a system in which everything rose
once the feature landed would not have demonstrated a gate.

What this control does *not* establish: one anchored finding on one change is a demonstration,
not a measurement. The corpus records a measured outcome for exactly one incident, so the
anchor rate cannot be estimated from it, and `class_b_anchor_rate` is not a metric worth
quoting until a corpus has more than one precedent in it.

**Ablation on the precedent.** Deny the requester the incident (labelled `support-operations`,
run with an empty scope) and the same change produces no event, no anchor, and a Class B
finding back at `speculative` - the ablation bar applied to the analogue rather than to the
evidence. Break the historical analyst instead and the result is the same status with the stage
reported `failed`: degradation is downward either way.

#### Negative control - a change the corpus does not document

An espresso-machine maintenance change, with no assumption delta.

| stage | status |
|---|---|
| coupling | completed (21 bindings - the corpus is still there) |
| class_a | **empty** |
| retrieval | **empty** |
| class_b | **empty** |
| evidence_gaps | completed (**1**) |
| contradictions | **empty** |
| challenge | **empty** |
| blind_spots | completed (**2**: 1 hypothesis + 1 corpus silence) |
| evidence_arbitration | completed (1) |
| judgment | completed (1) |
| graph | completed (22 edges - the bindings still project) |
| report | **empty** (nothing foregrounded) |

**1 consequence, 0 material findings, nothing on the default surface, `is_quiet = True`,
`succeeded = True`.** Quiet because nothing actionable was found, not because a stage broke -
and the statuses are what make those two distinguishable.

**The blind-spot stage is deliberately not empty here, and that is the honest result.** On a
change the main query found nothing for, *every* retrievable span is unseen, so the partition is
at its **largest** (9 spans, 3 slices covered, 0 dropped as already-seen) and the hunter does
propose one hypothesis. What matters is what happens to it: `speculative`, not foregrounded,
`material_consequences` empty. Demanding that the stage fall silent exactly where it has the
most to look at would be demanding a false negative from the one stage whose job is coverage.

Note what the run is *not*: silent. One `EvidenceGap`, one corpus-silence finding and two
unresolved questions are recorded, because a retrieval that returned nothing over a readable
corpus is exactly the negative provenance type E needs. The honest quiet result is **nothing to
act on plus a recorded account of what could not be established**, not an empty page.

#### Ablation honesty - the retention policy removed

The corpus is re-ingested without `policy/data-retention-policy.md` and the whole pipeline
re-run. Nothing is patched afterwards, so the degradation has to come from the resolver no
longer being able to derive the finding.

| | intact | ablated |
|---|---|---|
| Class A findings | 9 | **6** |
| retrieved spans | 6 | 4 |
| material consequences | 9 | **6** |
| foregrounded in the report | 9 | **6** |
| `retention.window` holders in the graph | 5 | **4** |
| graph edges | 27 | 23 |
| arbitration rulings clamped | 0 of 11 | **3 of 8** |
| confidence range | 0.199 - 0.384 | 0.132 - 0.396 |

The precise result:

- **4 findings rested on the removed document's spans. All 4 disappeared**, identified by
  `finding_id` (derived from relation + affected version, so it is the same finding, not a
  similar one).
- **0 surviving findings, consequences or hypotheses cite a removed evidence id.** This is the
  sharpest available anti-priors check: the cooperative fake cites everything it is offered, so
  a removed id reaching a citation could only have come from somewhere other than the corpus.
- **3 `retention.window` findings correctly survive.** Five artifacts bind the key, and the EU
  policy and the team notes still genuinely disagree about it. Demanding that the key vanish
  would be demanding a false negative - the opposite failure, and just as bad.
- **One finding reattributed.** A `stale_consumer` on the metric definition names the Data
  Retention Policy as its counterpart intact and the Vendor Integration Notes ablated, because
  `resolve_authority` now returns a different governing declarer. That is the resolver
  recomputing on a smaller corpus, not the pipeline inventing: it is a different finding with a
  different id, resting on spans that both still exist.
- **The evidence arbiter degrades independently, which was not designed for and is the most
  useful thing here.** Intact, every candidate measures 2 distinct corroborating source
  versions and nothing is clamped. Ablated, **3 of 8 rulings drop to `weakly_supported`** with
  reason `insufficient_independent_corroboration`, because removing one declarer leaves those
  findings on a single source version. So ablation honesty is now visible on two axes that were
  built separately - status via the judge's gates, and evidence position via a ceiling computed
  from a count the model cannot influence.

#### Failure propagation

With every agent returning unparseable output:

| stage | status |
|---|---|
| coupling / class_a / retrieval / graph / report | completed - **unaffected** |
| class_b, contradictions, challenge, blind_spots, evidence_arbitration, judgment | **failed**, each with its detail recorded |
| evidence_gaps | **skipped** |

9 Class A findings and 21 graph edges survive, and a report is still composed (empty surface,
deterministic summary); `succeeded = False`. A provider outage costs the agentic stages, not the
analysis - and the same stage reads `empty` on the quiet change and `failed` here, which is
requirement 5 stated as a comparison.

Per-stage, with only one agent broken:

| broken agent | stage status | what survives |
|---|---|---|
| `blind-spot-hunter` | `failed`, detail recorded | judgment completes; 0 blind-spot hypotheses, and the partition is still reported |
| `evidence-arbiter` | `failed` | every candidate is `unruled` - never defaulted to a position - and the judge still arbitrates |
| `report-composer` | REPORT stays `completed` | the report exists with the deterministic surface and summary; `layout.succeeded` is `False` |

#### A challenge that succeeds

With the fake challenger citing evidence against every candidate, all 11 consequences come out
`rejected` and **0** are material - and no `derived` graph edge references a rejected
consequence. The review stage can empty the report, which is what a challenger is for.

The arbiter agrees on a separate axis, which is the point of having two: **all 11 rulings are
clamped from `supported` to `disputed`**, with reason `refuted_by_challenge` or
`contradicting_evidence`. Nothing on the default surface, and the 11 findings stay reachable.

It was 10 of 11 before `blind_spots` moved ahead of `challenge`: the eleventh candidate was the
blind-spot finding, which nothing had challenged, so nothing could clamp it. That one number is
the measurement the reordering was worth making for.

### Unsupported claim rate
Final material claims lacking adequate evidence.

### Contradiction detection
Precision and recall of meaningful contradictions.

### Evidence citation correctness
Whether cited evidence actually supports the claim.

### Challenge value
Fraction of rejected/downgraded hypotheses that would otherwise have been false positives.

### Blind-spot yield

Novel correct consequences surfaced only by the blind-spot stage.

**Measurable now, and attributable rather than asserted.** The stage reports
`yield_by_slice`, derived from the spans each finding actually cited, so a figure can be traced
to the evidence basis that produced it. Correctness still needs a human pass; what the
implementation settles is that the yield is not a restatement:

| | positive control | quiet control |
|---|---|---|
| partition spans | 5 | 9 |
| slices covered | `single_holder`, `unwritten_rule` | + `unqueried_source_type` |
| spans dropped as already seen by the main hunt | 10 | 0 |
| findings | 1 (`single_holder`) | 1 (`single_holder`) |
| corpus-silence findings | 1 | 1 |
| findings citing any main-analysis evidence | **0** (structurally impossible) | **0** |
| findings reviewed by the challenger | **1 of 1** | **1 of 1** |

The last row is the one that makes the metric mean anything: the partition type refuses to hold
a span the main hunter was offered, so a blind-spot finding rests on evidence the main analysis
did not see or on nothing at all.

**Measured limitation.** Two of the four slices are empty on this corpus. `low_ranked` needs
retrieval to return more spans than the hunter's budget (24), which 6 results cannot do;
`unqueried_source_type` needs the change's own wording to hook a different source type, which
it does not. Both are honest zeroes rather than defects, and both would fill on a larger corpus
- but until then the stage's real yield comes from one slice, and reporting otherwise would
overstate it.

### Specialist fan-out, as measured

Positive control, `retention.window`:

| | |
|---|---|
| invoked | **3 of 7** - policy (3 spans), process (2), data (1) |
| not invoked | technology, people, risk, historical - no authorized evidence of their source types, each reported with that reason |
| observations retained | **4** |
| suppressed | **2**, both the same uncited claim from a second analyst |
| fabricated citations dropped | 1 |
| bindings reported for review, none admitted | 3 |
| unresolved questions contributed | 3 of the analysis's 5 |

Quiet control: **0 of 7 invoked**, all seven reported un-selected, stage `empty`. Seven
provider calls not made, which is the cheap gate doing exactly what it is for.

**The measurement that did not go the way the design assumed.** On this corpus the three
slices *partition* the six spans the main retrieval already returned - `spans only a
specialist retrieved: 0`. The fan-out re-reads the authorized set by domain rather than
widening it, because the main query's `top_k` of 10 already recovered every lexically matching
span. So the value delivered here is per-domain *reading*, not extra recall, and the
slice-union mechanism that would add recall is built and unexercised. On a corpus where `top_k`
truncates, or with a semantic embedder, it would matter; saying so now is cheaper than
discovering the claim was hollow later.

Failure isolation:

| broken | result |
|---|---|
| one analyst (`policy-analyst`) | stage `failed` naming it, **the other two analysts' findings retained**, Class B and everything downstream complete |
| all three invoked analysts | stage `failed`, 0 observations, the hunter runs alone and still produces a hypothesis and 11 consequences |
| none configured | stage `not_configured`, and the Class A finding ids are identical to the agentic run's |

### Evidence arbitration, as measured

| control | rulings | clamped | positions |
|---|---|---|---|
| positive | 11 | 0 | `supported` |
| ablated | 8 | **3** | `supported`, `weakly_supported` |
| challenger refutes | 11 | **11** | `disputed` |
| arbiter broken | 0 | - | every candidate `unruled` |

**Corroboration overclaim is caught in every run.** The cooperative fake reports the measured
source versions plus `sv_000...0`, which no corpus contains. In all four controls the invented
version is trimmed into `overclaimed_source_version_ids` and the ceiling reads the measured
count - so the overclaim never moves a position. That is the shape of the attack this stage
exists to survive, exercised by the pipeline rather than by a unit fixture.

### Reviewer suppression across analyses, as measured

Two runs of the same change with one `wrong` and one `already_known` verdict recorded between
them:

| | first run | second run |
|---|---|---|
| foregrounded | 9 | **7** |
| findings matched by a standing verdict | 0 | **3** |
| suppressed from the surface | 0 | **2** |
| already-known rate | not measurable | **0.33** |
| active standing suppressions | 0 | 1 |

**One verdict suppressed two findings**, and that is correct: suppression is scoped to
`(assumption_key, source version)`, and two findings in the second run share that scope. The
unit of a reviewer's rejection is the artifact they were looking at, not the consequence record
- a reviewer who says "this deck is wrong about retention" has said it about every finding that
rests on that deck.

Both suppressed findings stay in `AnalysisReport.consequence_ids` and are counted in
`suppressed_count`; both `already_known` findings stay on the default surface.

### Alert load
Findings surfaced per analysis at default view settings, and the ratio of `useful` to
`already_known` + `not_material` + `wrong` dispositions. A system with excellent precision
that surfaces 18 findings per change will still be abandoned.

### Cost
Approximate model cost per analysis, broken down into amortized ingestion-time extraction
and per-analysis reasoning. Reporting only the second understates the true cost.

### Latency
End-to-end analysis latency.

## Provider-diversity study

Because per-agent provider binding is configuration (`ARCHITECTURE.md` 11.2), the harness
must be able to run:

- F1: all agents on one provider and model
- F2: specialists on one provider, judge and challenger on another
- F3: adversarial agents on a different model family from the generators
- F4: local model vs remote model at equal tier
- F5: same provider, different model families

The question is whether provider/model diversity reduces correlated reasoning error beyond
what input diversity already achieves. Report it as an experimental result, and hold the
retrieval configuration fixed across conditions or the comparison is confounded.

## Reviewer disposition, as implemented

The primary feedback signal now has a producer (`IDEA.md` 21.2). What it measures, and what it
cannot yet:

| metric | source | status |
|---|---|---|
| `already_known_rate` | share of reviewed findings marked `already_known` | **measurable** |
| `useful_rate` | share marked `useful` | **measurable** |
| `wrong_rate` | share marked `wrong` | **measurable** |
| `coverage` | share of an analysis a reviewer has looked at | **measurable** |
| alert load after suppression | `ForegroundView.suppressed_count` | **measurable** |

All four rates are computed over the **reviewed** set rather than every finding. An analysis
nobody has looked at reports `None`, not zero: a 0% already-known rate on an unreviewed
analysis would read as a success and is the easiest number in this system to mislead with.

**`already_known_rate` is the one the product thesis rests on.** It is the failure no internal
metric can detect: precision, recall, coupling recall and confidence can all look excellent
while the system reports only what a diligent reviewer would have found anyway. The pipeline
currently produces 9 material findings on the positive control (see "End-to-end pipeline, as
measured") and this is the first slice that makes "were any of them worth reading?" a question
with an answer.

**No number is reported yet, and that is honest.** There are no reviewers and no stored
analyses, so the rates are exercised against synthetic verdicts in
`tests/unit/judgment/test_dispositions.py` and nothing else. The machinery is measured; the
signal is not, and it cannot be until a real reviewer looks at a real run.

### What suppression is measured to do

- `wrong` and `not_material` remove a finding from the foreground; the consequence, its
  evidence, its provenance, its score and its graph edges are **byte-identical** before and
  after - asserted by serializing them.
- suppression is **reversible**: a corrected verdict restores the finding, and the transition
  records both values, the reviewer, the timestamp and the reason.
- `already_known` **does not** suppress, and the frozen entity refuses to let it request
  persistent suppression at all.
- a persistent suppression is scoped to `(assumption_key, source_version_id)` and expires when
  the version changes, so a revised document is re-reported rather than permanently silenced.

## Human judging

Use blinded evaluation where possible.

Each finding should be judged for:
- correctness
- materiality
- evidence sufficiency
- severity
- novelty relative to direct dependencies

## Important caution

Do not report confidence as probability unless calibration has been separately tested.

Do not report an improvement over baselines from the development scenario suite as a
validated result. The suite is authored alongside the system and will flatter it.
