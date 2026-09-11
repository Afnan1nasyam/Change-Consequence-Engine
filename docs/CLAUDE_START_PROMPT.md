# First Claude Code Prompt

You are entering an existing project specification for Change Consequence Engine (CCE).

Before writing code, read these files in this exact order:

1. `CLAUDE.md`
2. `docs/IDEA.md`
3. `docs/ARCHITECTURE.md`
4. `docs/FOLDER_STRUCTURE.md`
5. `docs/DATA_MODEL.md`
6. `docs/DEVELOPMENT_PLAN.md`
7. `docs/EVALUATION.md`
8. `docs/SECURITY.md`
9. `docs/STATUS.md`
10. all files under `.claude/agents/`

Do not ask me to re-explain the project. Treat those files as the authoritative project specification.

Your first task is ONLY Phase 0 from `docs/DEVELOPMENT_PLAN.md`.

Requirements:
- inspect the repository before changing anything
- preserve the architecture boundaries in the specification
- use Python 3.12+
- create the minimal production-quality project foundation
- create FastAPI health endpoint
- configure pyproject.toml
- configure database migration scaffolding
- create Docker Compose for local infrastructure
- create test infrastructure
- create `.env.example`
- create/update README only where necessary
- update `docs/STATUS.md`
- do not implement later phases yet
- do not create Python agent classes; agent definitions belong in `.claude/agents/*.md`
- do not introduce LangGraph/LangChain unless a concrete need is demonstrated
- do not invent missing requirements
- run tests and validation commands before declaring Phase 0 complete

When you finish, report:
- files created/changed
- commands executed
- test results
- unresolved issues
- what Phase 1 will require, without implementing Phase 1
