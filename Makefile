# Top-level developer-experience targets.
#
# All targets are .PHONY; this Makefile is a typing shortcut, not a
# build system. The canonical build paths are still ``pytest``,
# ``uvicorn``, and ``npm`` invoked directly. Targets here just spell
# out the conventional invocation so a new contributor can boot the
# webapp with one command.
#
# Convention:
#   make webapp-dev      → boot backend on :8000 and frontend on :5173
#   make webapp-backend  → backend only
#   make webapp-frontend → frontend only
#   make webapp-test     → backend pytest + frontend lint + frontend build
#   make webapp-build    → frontend production build only
#
# Override ports with `UVICORN_PORT=8001 make webapp-backend`.

.PHONY: webapp-dev webapp-backend webapp-frontend webapp-test webapp-build

UVICORN_PORT ?= 8000
VITE_PORT ?= 5173

# Boot both servers in one command. `trap 'kill 0'` propagates Ctrl+C
# to every backgrounded child so the cleanup story stays sane on
# macOS GNU make 3.81 (Apple's bundled version) without `.ONESHELL`.
webapp-dev:
	@echo ">> backend  → http://localhost:$(UVICORN_PORT)"
	@echo ">> frontend → http://localhost:$(VITE_PORT)"
	@trap 'kill 0' INT TERM EXIT; \
	  uvicorn webapp.backend.main:app --reload --port $(UVICORN_PORT) & \
	  (cd webapp/frontend && npm run dev -- --port $(VITE_PORT)) & \
	  wait

webapp-backend:
	uvicorn webapp.backend.main:app --reload --port $(UVICORN_PORT)

webapp-frontend:
	cd webapp/frontend && npm run dev -- --port $(VITE_PORT)

webapp-test:
	pytest webapp/backend/tests -q
	cd webapp/frontend && npm run lint && npm run build

webapp-build:
	cd webapp/frontend && npm run build
