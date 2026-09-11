"""Engine, session factory, and the handle the repositories hold.

Lazily constructed so importing the application does not require a reachable database: the
test suite and a deterministic-only run must work with no infrastructure.

Why the repositories hold a :class:`Database` rather than a ``Session``
-----------------------------------------------------------------------

Every method on the store ports is a complete operation - ``put_document`` either happened or
did not - and none of them takes a transaction. The in-memory implementations commit
immediately, so the faithful SQL mapping is **one transaction per call**, which is what
:meth:`Database.session` gives by default.

That is the right default and it is not always what a caller wants.
``ResolveCouplingUseCase`` writes the index, the relations and the findings in three calls, and
a process that died between them would leave a tenant's projection half-replaced. So
:meth:`Database.unit_of_work` binds one session for its duration and the repositories join it
instead of opening their own - which makes those three calls atomic *when the caller asks*,
without putting a transaction in the port.

The alternative - handing each repository a live session and making the caller commit - was
rejected because it moves durability into every call site, and a call site that forgets is
indistinguishable from one that succeeded.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings


class DatabaseNotConfigured(RuntimeError):
    """No ``CCE_DATABASE_URL`` was supplied. Raised at the point of use, not at import."""


def database_url() -> str:
    """The configured URL, or a refusal naming the variable that is missing.

    Deliberately not defaulted to a local Postgres with development credentials. A default
    that happens to work on a laptop is how a credential ends up committed, and a default that
    happens to *not* work produces a connection error instead of a configuration error.
    """
    url = get_settings().database_url
    if not url:
        raise DatabaseNotConfigured(
            "CCE_DATABASE_URL is not set. Persistence needs an explicit database URL; there is "
            "no default, because a default carrying credentials is how credentials get "
            "committed. See .env.example and docker-compose.yml for the local shape."
        )
    return url


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        database_url(),
        pool_pre_ping=True,
        echo=settings.debug,
        future=True,
    )


@functools.lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def reset_engine_cache() -> None:
    """Drop cached engine/session factory. For tests that repoint the database URL."""
    get_engine.cache_clear()
    get_session_factory.cache_clear()


class Database:
    """A session source the repositories share. Owns no domain logic.

    Constructed from a ``sessionmaker`` rather than a URL so a test can point it at any engine
    - including a file-backed SQLite one, which is what lets the repository conformance suite
    run with no server while executing the same code.
    """

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions
        self._bound: Session | None = None

    @classmethod
    def from_url(cls, url: str, *, echo: bool = False) -> Database:
        """Build an independent engine for ``url``. Does not touch the module-level cache."""
        engine = create_engine(url, pool_pre_ping=True, echo=echo, future=True)
        return cls(sessionmaker(bind=engine, expire_on_commit=False, future=True))

    @classmethod
    def configured(cls) -> Database:
        """The application's database, from ``CCE_DATABASE_URL``."""
        return cls(get_session_factory())

    @property
    def engine(self) -> Engine:
        bind = self._sessions.kw.get("bind")
        if not isinstance(bind, Engine):  # pragma: no cover - sessionmaker always binds one
            raise DatabaseNotConfigured("this session factory is not bound to an engine")
        return bind

    @contextmanager
    def session(self) -> Iterator[Session]:
        """One committed transaction, or the caller's if a unit of work is open.

        When a unit of work is open this yields its session **without committing**, so the
        outer block decides. That is the whole mechanism: a repository does not know or care
        which of the two it is in.
        """
        if self._bound is not None:
            yield self._bound
            return
        with self._sessions.begin() as session:
            yield session

    @contextmanager
    def unit_of_work(self) -> Iterator[Session]:
        """Group several store calls into one transaction.

        Nesting is refused rather than silently flattened: a caller who thinks they opened an
        inner transaction that can roll back independently is wrong, and finding that out from
        a corrupted tenant projection is expensive.
        """
        if self._bound is not None:
            raise RuntimeError(
                "a unit of work is already open on this Database. Nested units of work are "
                "refused because the inner one would not be able to roll back independently, "
                "and a caller relying on that would be wrong in a way nothing reports."
            )
        with self._sessions.begin() as session:
            self._bound = session
            try:
                yield session
            finally:
                self._bound = None

    def create_all(self) -> None:
        """Create every table from the ORM metadata.

        For tests and for a throwaway database only. Production schema comes from Alembic, and
        a contract test asserts the migration and the metadata agree - so this is a shortcut,
        never a second source of truth.
        """
        import app.infrastructure.database.models  # noqa: F401  (registers the tables)
        from app.infrastructure.database.base import Base

        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        import app.infrastructure.database.models  # noqa: F401
        from app.infrastructure.database.base import Base

        Base.metadata.drop_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
