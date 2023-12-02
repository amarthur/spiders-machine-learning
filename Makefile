PYTHON := python3
PIP := pip3

SRC := src
CONFIG := src/configs
VENV := venv
VENV_ACTIVATE = source $(VENV)/bin/activate

all: run

run:
	(clear && \
	$(VENV_ACTIVATE) && \
	$(PYTHON) $(SRC)/main.py)

setup: venv_create venv_install

venv_create:
	$(PYTHON) -m venv $(VENV)

venv_install:
	($(VENV_ACTIVATE) && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements.txt)

clean:
	rm -r $(VENV)
