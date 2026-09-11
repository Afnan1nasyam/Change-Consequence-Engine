-- Executed once, on first Postgres container start.
--
-- Only extensions belong here. Schema is owned by Alembic migrations, so that schema
-- changes are versioned and reviewable rather than baked into an image.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trigram indexes back the lexical half of hybrid retrieval (docs/ARCHITECTURE.md 6).
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- A separate database for the persistence test suites, created here so the Postgres
-- verification run is one `docker compose up` and one environment variable. It exists
-- because those fixtures run `alembic downgrade base` between tests: pointing that at the
-- development database by forgetting to unset a variable would drop its tables, so the two
-- are kept apart by construction rather than by care.
SELECT 'CREATE DATABASE cce_test OWNER cce'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'cce_test')\gexec
