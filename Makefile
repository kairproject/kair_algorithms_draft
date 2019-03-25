test:
	env PYTHONPATH=./scripts pytest --flake8  # --cov=algorithms

format:
	isort -y
	black -t py27 .

dev:
	pip install -r scripts/requirements-dev.txt
	# python3 -m pip install black
	pre-commit install

dep:
	pip install -r scripts/requirements.txt
