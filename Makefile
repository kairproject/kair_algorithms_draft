test:
	pytest --flake8  # --cov=algorithms

format:
	black .
	isort -y

dev:
	pip install -r scripts/requirements-dev.txt
	pre-commit install

dep:
	pip install -r scripts/requirements.txt
