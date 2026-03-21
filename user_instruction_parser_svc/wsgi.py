"""
wsgi.py
WSGI entrypoint — use this when running with gunicorn or waitress.

Gunicorn:
	gunicorn wsgi:app --bind 0.0.0.0:5000

Waitress:
	waitress-serve --host=0.0.0.0 --port=5000 wsgi:app

Direct:
	python3 main.py
"""
import os
import sys

# Force the backend directory onto sys.path so imports resolve
# regardless of where gunicorn is invoked from
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
	sys.path.insert(0, _HERE)

# Also set the working directory so relative paths inside the app resolve
os.chdir(_HERE)

from main import app  # noqa: E402

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)
