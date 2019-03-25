test:
	env PYTHONPATH=./scripts pytest --flake8  # --cov=algorithms

format:
	isort -y
	python3.6 -m black -t py27 .

dev:
	pip install -r scripts/requirements-dev.txt
	sudo add-apt-repository ppa:deadsnakes/ppa -y
	sudo apt-get update
	sudo apt-get install python3.6 -y
	sudo apt-get install python3-pip -y
	python3 -m pip install black
	pre-commit install

dep:
	pip install -r scripts/requirements.txt
