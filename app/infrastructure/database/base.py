"""Declarative base and metadata conventions.

An explicit naming convention is set up front because Alembic autogenerate needs stable
constraint names to emit correct downgrades. Retrofitting one later means renaming every
constraint in a live database.

No ORM models exist yet - they arrive with the Phase 1 domain model, and only after the
Pydantic schemas are frozen.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    metadata = metadata
