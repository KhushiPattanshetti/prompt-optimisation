"""
wsgi.py
WSGI entrypoint — use this when running with gunicorn or waitress.

Gunicorn:
	gunicorn user_instruction_parser_svc.wsgi:app --bind 0.0.0.0:5000

Waitress:
	waitress-serve --host=0.0.0.0 --port=5000 user_instruction_parser_svc.wsgi:app

Direct:
	python3 -m user_instruction_parser_svc.main
"""
from .main import app

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)
