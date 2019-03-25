test:
	env PYTHONPATH=./scripts pytest --flake8  # --cov=algorithms

format:
	isort -y
	python3.6 -m black -t py27 .

dev:
	pip install -r scripts/requirements-dev.txt
	sudo add-apt-repository -y ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install -y python3.6
	sudo apt-get install -y python3-pip
	python3.6 -m pip install black
	pre-commit install

dep:
	pip install -r scripts/requirements.txt
