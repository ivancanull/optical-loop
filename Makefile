.PHONY: image doctor test smoke full multislice-smoke multislice-full

export UID := $(shell id -u)
export GID := $(shell id -g)

image:
	docker compose build

doctor: image
	docker compose run --rm opticalloop python3 optical_loop.py reproduce doctor

test: image
	docker compose run --rm opticalloop python3 -m pytest -q

smoke: image
	docker compose run --rm opticalloop python3 optical_loop.py reproduce smoke --workers $${WORKERS:-4}

full: image
	docker compose run --rm opticalloop python3 optical_loop.py reproduce full --workers $${WORKERS:-4} --max-jobs $${MAX_JOBS:-256}

multislice-smoke: image
	docker compose run --rm opticalloop python3 optical_loop.py multislice smoke --workers $${WORKERS:-4}

multislice-full: image
	docker compose run --rm opticalloop python3 optical_loop.py multislice full --workers $${WORKERS:-4} --max-jobs $${MAX_JOBS:-256}
