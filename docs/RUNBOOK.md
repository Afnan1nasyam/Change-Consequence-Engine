# Operational runbook

Short, and only for things that have to be done in a specific order. Everything else lives in
`README.md` or in the module that does it.

## Database

### Configure

There is **no default database URL**. `CCE_DATABASE_URL` must be set, and
`app.infrastructure.database.session.database_url()` raises `DatabaseNotConfigured` naming the
variable when it is not. A default carrying `cce:cce` would be a credential in source, and a
default pointing at a host that is not there turns a configuration mistake into a connection
error.

```bash
# .env (gitignored) or the shell. .env.example carries a placeholder only.
export CCE_DATABASE_URL='postgresql+psycopg://cce:<password>@localhost:5432/cce'
```

The Postgres driver is an optional extra, because the test suite installs and runs without it:

```bash
pip install -e '.[postgres,dev]'
```

### Start the local server

```bash
docker compose up -d postgres
docker compose exec postgres pg_isready -U cce -d cce
```

`scripts/init_db.sql` runs once on first container start. It installs extensions
(`vector`, `uuid-ossp`, `pg_trgm`) and creates the `cce_test` database the persistence suites
use. **Schema is owned by Alembic**, so a schema change is a reviewable revision rather than
something baked into an image.

### Migrate

```bash
alembic current          # where this database is
alembic upgrade head     # apply
alembic downgrade -1     # step back one revision
```

`alembic.ini` deliberately carries **no** `sqlalchemy.url`. `migrations/env.py` reads it from
application settings, so a credential never lives in a committed file. A contract test asserts
that line stays empty.

To review the DDL before it touches a database - and this works with no server running, which
is how the Postgres dialect is verified in an environment that has none:

```bash
alembic upgrade head --sql > /tmp/upgrade.sql
```

### Check the application is ready

```bash
python -m app.composition          # human-readable
python -m app.composition --json   # one JSON object
```

The composition root (`app/composition.py`) constructs the whole object graph from
configuration - database, the seven SQL stores, ingestion, and `AnalyzeChangeUseCase` - and
reports what it got: the dialect in use, the retrieval strategies it can actually serve, the
registry version, the loaded agents, and whether the database is reachable and migrated. Exits
non-zero when it is not.

It is a readiness check, not a server. The point is that a configuration mistake fails here,
naming the setting, rather than on the first analysis somebody runs.

## Running the persistence tests against Postgres

The persistence suites run on SQLite by default so they need no infrastructure. Point them at a
real server with a **separate** variable:

```bash
# `cce_test` is created by scripts/init_db.sql on first container start. If the volume
# predates that, create it once: docker compose exec postgres createdb -U cce cce_test
export CCE_TEST_DATABASE_URL='postgresql+psycopg://cce:<password>@localhost:5432/cce_test'
pytest tests/integration/test_sql_repositories.py tests/integration/test_persistence_restart.py -v
```

`CCE_TEST_DATABASE_URL` is deliberately not `CCE_DATABASE_URL`: the fixtures run
`alembic downgrade base` between tests, and pointing that at a development database by
forgetting to unset one variable would drop its tables.

`tests/integration/test_persistence_restart.py::TestSubstrate` skips when the variable is unset
and passes when it names a Postgres URL, so a green run records which substrate it proved.

**What each substrate proves.** SQLite exercises the SQL, the ORM mapping, the JSON codecs, the
ordering guarantees, the tenant predicates, the transaction boundaries and durability across a
process restart - the same repository code and the same Alembic revision. It proves nothing
about JSONB indexing, Postgres type coercion, or behaviour under Postgres' isolation levels.
Those need the real server.

## Writing a full analysis to a database by hand

Useful for inspecting real rows:

```bash
python -m tests.fixtures.persist_corpus "$CCE_DATABASE_URL"
```

Ingests the sample corpus, resolves the coupling projection, runs the twelve-stage pipeline,
records two reviewer verdicts and one reversal, and prints a JSON manifest of what it wrote. It
is a test fixture, not a seeding tool - it uses the replay extractor and the fake provider, so
nothing it produces came from a live model.
