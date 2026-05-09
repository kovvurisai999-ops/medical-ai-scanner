#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Create lightweight dummy models to prevent deployment crashes
python create_dummy_models.py

python manage.py collectstatic --no-input
python manage.py migrate
