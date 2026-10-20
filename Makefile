export LOCAL_UID := $(shell id -u)
export LOCAL_GID := $(shell id -g)
PYTHON ?= python3
IMAGE ?= /input/zoom_views/zoom_bottle_a020_near.png
OUTPUT ?= /output/bottle-near

.PHONY: help build doctor inspect detect bundle
help:
	@printf '%s\n' 'make build   - build CPU-only official YOLO image' 'make doctor  - verify model, CPU packages and source provenance' 'make inspect - inventory dataset (no labels changed)' 'make detect IMAGE=/input/table_000.png OUTPUT=/output/run-001' 'make bundle  - package code, images and portable scene/assets for another PC'
build:
	docker compose build vision
doctor:
	mkdir -p outputs
	docker compose run --rm vision doctor
inspect:
	mkdir -p outputs
	docker compose run --rm vision inspect /input
detect:
	mkdir -p outputs
	docker compose run --rm vision detect "$(IMAGE)" --output "$(OUTPUT)"
bundle:
	$(PYTHON) scripts/export_portable_bundle.py