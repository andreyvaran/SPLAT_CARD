.PHONY: run-server find-processclean install migrate run-server-nh stop-server backup backup_data check ruff_fix black_fix

clean:
	@echo "Removing lock..."
	@rm poetry.lock

migrate:
	@echo "Applying database migrations..."
	@poetry run alembic upgrade head
	@echo "Migrations applied successfully."


run-server:
	@echo "Starting server..."
	@poetry run python3 run.py

run-server-nh:
	@echo "Starting server..."
	@nohup poetry run python3 run.py & echo $$! > start_pid

find-process:
	@if [ -z "$(p)" ]; then \
		echo "Usage: make find-process PORT=<port_number>"; \
	else \
		echo "Searching for process using port $(p)"; \
		sudo ss -tulnp | grep :$(p); \
	fi

stop-server:
	@sudo netstat -tulnp | grep ':8080' | awk '{print $7}' | cut -d'/' -f1 | xargs -r sudo kill -2

ruff_fix:
	ruff check --fix .

black_fix:
	black .

# Комбинированная проверка: сначала Ruff и Black с исправлениями, затем линтер Ruff
check: ruff_fix black_fix
	@echo "All checks passed!"


backup:
	@poetry run python3 tools/backup/backup.py export_all --output_dir backup



backup_data:
	@echo "Exporting to csv"
	output_dir=backup_$$(date +%Y-%m-%d) && \
	mkdir -p $$output_dir && \
	echo $$output_dir && \
	poetry run python3 tools/backup/backup.py export_all --output_dir $$output_dir

