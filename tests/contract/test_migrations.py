"""Architectural contract: the migration and the ORM metadata describe the same schema.

Two schemas exist in this repository - ``Base.metadata`` (what the code expects) and the
Alembic revision history (what a database actually gets) - and nothing in SQLAlchemy makes them
agree. A model field added without a migration produces a column the code reads and the
database does not have, which fails in production and nowhere else.

So the parity is asserted by building the schema **both ways** and diffing the reflected
result:

* ``alembic upgrade head`` against a throwaway database;
* ``Base.metadata.create_all`` against a second one.

Columns, nullability, types, primary keys, indexes, unique constraints and foreign keys must
match table for table. The comparison runs on SQLite because it needs no server; the same
revision is separately compiled to Postgres DDL in
:func:`TestPostgresDdl.test_the_revision_compiles_to_postgres_ddl_with_jsonb_payloads`, so the
production dialect is covered without one.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

import app.infrastructure.database.models  # noqa: F401  (registers the tables)
from app.config.settings import REPO_ROOT
from app.infrastructure.database.base import Base
from app.infrastructure.database.models.tables import ALL_TABLES
from tests.fixtures.databases import alembic_config, migrate, pointed_at, sqlite_url

pytestmark = pytest.mark.contract

Snapshot = dict[str, dict[str, Any]]


def _reflect(url: str) -> Snapshot:
    """Everything about a schema that the application depends on, and nothing else.

    ``written_at`` server defaults are deliberately not compared: the model sets a Python-side
    default and the migration sets none, which is a real difference with no observable effect
    - every write goes through the ORM.
    """
    engine = create_engine(url)
    inspector = inspect(engine)
    snapshot: Snapshot = {}
    for table in sorted(inspector.get_table_names()):
        if table == "alembic_version":
            continue
        snapshot[table] = {
            "columns": {
                column["name"]: (str(column["type"]), bool(column["nullable"]))
                for column in inspector.get_columns(table)
            },
            "primary_key": tuple(inspector.get_pk_constraint(table)["constrained_columns"]),
            "indexes": {
                index["name"]: (tuple(index["column_names"]), bool(index["unique"]))
                for index in inspector.get_indexes(table)
            },
            "unique": {
                constraint["name"]: tuple(constraint["column_names"])
                for constraint in inspector.get_unique_constraints(table)
            },
            "foreign_keys": {
                key["name"]: (
                    tuple(key["constrained_columns"]),
                    key["referred_table"],
                    tuple(key["referred_columns"]),
                )
                for key in inspector.get_foreign_keys(table)
            },
        }
    engine.dispose()
    return snapshot


@pytest.fixture
def migrated(tmp_path: Path) -> Snapshot:
    url = sqlite_url(tmp_path / "migrated.db")
    migrate(url)
    return _reflect(url)


@pytest.fixture
def declared(tmp_path: Path) -> Snapshot:
    url = sqlite_url(tmp_path / "declared.db")
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    engine.dispose()
    return _reflect(url)


class TestMigrationParity:
    def test_the_migration_creates_exactly_the_declared_tables(
        self, migrated: Snapshot, declared: Snapshot
    ) -> None:
        assert set(migrated) == set(declared)
        # And both cover every model, so a table nobody registered is also caught.
        assert set(migrated) == {model.__tablename__ for model in ALL_TABLES}

    @pytest.mark.parametrize(
        "section", ["columns", "primary_key", "indexes", "unique", "foreign_keys"]
    )
    def test_the_migration_matches_the_metadata(
        self, migrated: Snapshot, declared: Snapshot, section: str
    ) -> None:
        """Parametrized per section so a failure names *what* diverged, not just that it did."""
        for table in sorted(declared):
            assert migrated[table][section] == declared[table][section], table


class TestRevisionHistory:
    def test_there_is_exactly_one_head(self) -> None:
        """Two heads mean a merge nobody performed, and `upgrade head` then fails."""
        heads = ScriptDirectory.from_config(alembic_config()).get_heads()
        assert len(heads) == 1, heads

    def test_every_revision_has_a_downgrade(self) -> None:
        """A revision that cannot be undone is a deployment with no way back."""
        script = ScriptDirectory.from_config(alembic_config())
        for revision in script.walk_revisions():
            source = Path(revision.path).read_text(encoding="utf-8")
            assert "def downgrade()" in source, revision.revision
            body = source.split("def downgrade()", 1)[1]
            assert "pass" not in body.split("\n", 2)[-1][:200], (
                f"{revision.revision}: downgrade is a stub"
            )

    def test_the_schema_can_be_torn_down_and_rebuilt(self, tmp_path: Path) -> None:
        """`upgrade -> downgrade -> upgrade` must leave the same schema.

        The cheapest way to catch a downgrade that drops tables in the wrong order, which on
        Postgres is a foreign-key error and on SQLite is silent.
        """
        url = sqlite_url(tmp_path / "cycle.db")
        with pointed_at(url):
            config = alembic_config()
            command.upgrade(config, "head")
            first = _reflect(url)
            command.downgrade(config, "base")
            assert _reflect(url) == {}
            command.upgrade(config, "head")
            assert _reflect(url) == first


class TestPostgresDdl:
    def test_the_revision_compiles_to_postgres_ddl_with_jsonb_payloads(
        self, tmp_path: Path
    ) -> None:
        """Offline SQL generation for the production dialect. No server needed.

        This is what makes the Postgres target verifiable in an environment with no Postgres:
        Alembic renders the revision through the ``postgresql`` dialect, so a construct that
        does not compile there fails here rather than on first deploy.
        """
        captured = io.StringIO()
        # Alembic's offline mode writes to `sys.stdout` unless `env.py` passes an
        # `output_buffer`, so redirecting the stream is what captures it without making the
        # production `env.py` aware of this test.
        with (
            pointed_at("postgresql+psycopg://user:pass@localhost:5432/cce"),
            contextlib.redirect_stdout(captured),
        ):
            command.upgrade(alembic_config(), "head", sql=True)

        sql = captured.getvalue()
        (tmp_path / "upgrade.sql").write_text(sql, encoding="utf-8")
        for model in ALL_TABLES:
            assert f"CREATE TABLE {model.__tablename__} (" in sql
        # Every payload column must be JSONB on the production dialect. A `JSON` here would
        # mean the variant is wired the wrong way round and the binary form was never used.
        assert sql.count("payload JSONB NOT NULL") == len(ALL_TABLES)
        assert "JSONB" in sql and "payload JSON NOT NULL" not in sql

    def test_no_credential_appears_in_a_committed_alembic_file(self) -> None:
        """`alembic.ini` must not carry a URL. env.py reads it from settings."""
        ini = (REPO_ROOT / "alembic.ini").read_text(encoding="utf-8")
        line = next(entry for entry in ini.splitlines() if entry.startswith("sqlalchemy.url"))
        assert line.split("=", 1)[1].strip() == ""
