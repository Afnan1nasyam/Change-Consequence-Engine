"""Disposable databases for the persistence tests, and the rule for which one runs.

Two substrates, one code path
-----------------------------

The repositories under test are Postgres-targeted, and the only Postgres-specific thing in
them is the ``JSONB`` column type - which carries a ``JSON`` variant. So the *same* repository
code, the same ORM models and the same Alembic revision run on SQLite, and that is the default
substrate here: it ships with Python, needs no daemon, and lets every persistence test run in
CI with no infrastructure.

When ``CCE_TEST_DATABASE_URL`` is set, every fixture below repoints at it instead and the same
tests run against the real target. That is the intended verification path, and
``docker-compose.yml`` plus ``docs/RUNBOOK.md`` give the command.

**What SQLite does and does not prove.** It proves the SQL, the ORM mapping, the codecs, the
ordering guarantees, the tenant predicates, the transaction boundaries and durability across a
process restart. It does not prove anything about JSONB indexing, Postgres type coercion or
concurrent-write behaviour under Postgres' isolation levels. Those need the real thing, which
is why the gate exists rather than a claim that SQLite is equivalent.

Schema is created by **Alembic**, not by ``create_all``, so the migration is on the tested
path. ``tests/contract/test_migrations.py`` separately asserts the two agree.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.config.settings import REPO_ROOT, get_settings
from app.infrastructure.database.session import Database, reset_engine_cache

#: Set this to run every persistence test against a real server, e.g.
#: ``postgresql+psycopg://cce:cce@localhost:5432/cce_test``. Deliberately a *separate*
#: variable from ``CCE_DATABASE_URL``: pointing the test suite at a development database by
#: forgetting to unset one variable would drop its tables.
TEST_URL_ENV = "CCE_TEST_DATABASE_URL"


def configured_url() -> str | None:
    """The real-server URL, when one is configured."""
    return os.environ.get(TEST_URL_ENV) or None


def is_postgres() -> bool:
    url = configured_url()
    return url is not None and url.startswith("postgresql")


def sqlite_url(path: Path) -> str:
    """A file-backed SQLite URL. File-backed, not ``:memory:``, on purpose.

    An in-memory database dies with the connection pool, so it cannot answer the question
    these tests exist to answer: does the state survive a new process opening the same
    database.
    """
    return f"sqlite+pysqlite:///{path.as_posix()}"


@contextmanager
def pointed_at(url: str) -> Iterator[None]:
    """Point ``CCE_DATABASE_URL`` at ``url`` for the duration, caches cleared both ways.

    ``get_settings`` is ``lru_cache``d, which is right for a process and wrong for a test that
    repoints the database: without the ``cache_clear`` the environment variable is set and
    ignored, and every migration in the session lands on whichever URL happened to be first.
    That failure is silent - the migration succeeds, against the wrong database - so the cache
    reset belongs in the helper rather than in each caller.
    """
    previous = os.environ.get("CCE_DATABASE_URL")
    os.environ["CCE_DATABASE_URL"] = url
    get_settings.cache_clear()
    reset_engine_cache()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CCE_DATABASE_URL", None)
        else:
            os.environ["CCE_DATABASE_URL"] = previous
        get_settings.cache_clear()
        reset_engine_cache()


def alembic_config() -> Config:
    return Config(str(REPO_ROOT / "alembic.ini"))


def migrate(url: str) -> None:
    """Run ``alembic upgrade head`` against ``url``.

    The URL is passed through the environment because ``migrations/env.py`` reads it from
    settings - which is the production path, and exercising it here means a broken settings
    wiring fails a test rather than a deployment.
    """
    with pointed_at(url):
        command.upgrade(alembic_config(), "head")


def downgrade(url: str) -> None:
    with pointed_at(url):
        command.downgrade(alembic_config(), "base")


@contextmanager
def disposable_database(tmp_path: Path, *, name: str = "cce_test") -> Iterator[Database]:
    """A migrated, empty database that is torn down afterwards.

    On Postgres the teardown is a ``downgrade base`` so the next test starts clean without
    needing to drop the database; on SQLite the file simply goes away with ``tmp_path``.
    """
    url = configured_url() or sqlite_url(tmp_path / f"{name}.db")
    if is_postgres():
        # A previous failed run may have left tables behind. Start from a known state rather
        # than from whatever the last failure produced.
        downgrade(url)
    migrate(url)
    database = Database.from_url(url)
    try:
        yield database
    finally:
        database.dispose()
        if is_postgres():
            downgrade(url)
