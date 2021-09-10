LINT_DIRS = . sklearn-iris

install-dev:
	pip install -r requirements.txt

format-all:
	black $(LINT_DIRS)
	isort --recursive $(LINT_DIRS)

lint-all: format-all
	black --check $(LINT_DIRS)
	flake8 $(LINT_DIRS)
	isort --check --recursive $(LINT_DIRS)


.PHONY: install-dev format-all lint-all