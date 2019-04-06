test:
	env PYTHONPATH=./scripts pytest --flake8 --ignore=./scripts/envs # --cov=algorithms

format:
	isort -y
	python3.6 -m black -t py27 . --fast

dev:
	pip install -r scripts/requirements-dev.txt
	python3.6 -m pip install black  # required python3.6 & python3-pip
	pre-commit install

dep:
	pip install -r scripts/requirements.txt

py36:
	# install python3.6 for black (only ubuntu)
	pip install -r scripts/requirements-dev.txt
	sudo add-apt-repository -y ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install -y python3.6
	sudo apt-get install -y python3-pip