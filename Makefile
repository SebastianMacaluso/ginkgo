# Makefile for ToyJetsShower
SHELL := /bin/bash

# You can set these variables from the commandline.
VERSION=$(shell python setup.py --version)

./dist/ToyJetsShower-${VERSION}-py3-none-any.whl:
	python ./setup.py sdist bdist_wheel

install: ./dist/ToyJetsShower-${VERSION}-py3-none-any.whl # pip install
	pip install --upgrade ./dist/ToyJetsShower-${VERSION}-py3-none-any.whl

uninstall:
    pip uninstall ToyJetsShower

%: Makefile