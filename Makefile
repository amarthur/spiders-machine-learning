PYTHON := python3
PIP := pip3

SRC := src
CONFIG := src/configs
VENV := venv
VENV_ACTIVATE = source $(VENV)/bin/activate

all: fit

fit:
	(clear && \
	$(VENV_ACTIVATE) && \
	$(PYTHON) $(SRC)/main.py fit --config $(CONFIG)/config.yaml)

setup: venv_create venv_install

venv_create:
	$(PYTHON) -m venv $(VENV)

venv_install:
	($(VENV_ACTIVATE) && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements.txt)