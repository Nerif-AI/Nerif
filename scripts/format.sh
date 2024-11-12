#! /bin/bash

isort .
ruff check . --select I --fix
ruff format .