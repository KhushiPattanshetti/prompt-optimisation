# ICD-10 User Instruction Parser Microservice

## Overview
This project is a microservice for parsing and validating user instructions and clinical notes, forwarding them to the next stage in the pipeline (e.g., a Prompt Rewriter service). It is designed for easy integration with other services, supporting both Windows and Unix-like systems.

## Architecture & Flow
1. **User submits instruction and clinical note** via the `/api/validate` endpoint (supports JSON and multipart form-data).
2. **Validation** is performed on the input using custom logic in `app/validator.py`.
3. **Storage**: The validated instruction is stored locally (see `app/storage.py`).
4. **Forwarding**: The clinical note and metadata are forwarded to the next service (e.g., a Prompt Rewriter) using a POST request. The destination URL is set by the `PROMPT_REWRITER_URL` environment variable. If not set, forwarding is simulated for development/testing.
5. **Response**: The API returns validation results, storage info, and forwarding status.

## API Endpoints
- `POST /api/validate` — Validate and forward a user instruction and clinical note.
- `GET /api/instructions` — List all stored instructions.
- `GET /api/health` — Health check endpoint.

## Running the Service
### 1. Clone the repository
```
git clone https://github.com/KhushiPattanshetti/prompt-optimisation.git
cd prompt-optimisation
```

### 2. Set up a virtual environment
```
python3 -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the server
- **Windows (Waitress):**
  ```
  waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
  ```
- **Mac/Linux (Gunicorn or Waitress):**
  ```
  gunicorn wsgi:app --bind 0.0.0.0:5000
  # or
  waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
  ```

## Environment Variables
- `PROMPT_REWRITER_URL`: URL of the next service to forward clinical notes (e.g., a FastAPI endpoint). If not set, forwarding is simulated.

## Extending/Integrating
- To connect with a FastAPI service, set `PROMPT_REWRITER_URL` to the FastAPI endpoint URL.
- The POST payload sent to the next service includes:
  - `instruction_id`
  - `filename`
  - `clinical_note`

## Contributing
1. Create a new branch for your feature or bugfix.
2. Commit and push your changes.
3. Open a pull request.

## License
MIT

## Forwarding Logic Explained

The forwarding logic in `app/forwarder.py` works as follows:
- After validation and storage, the clinical note, filename, and instruction ID are packaged into a JSON payload.
- If the environment variable `PROMPT_REWRITER_URL` is set, the service sends this payload as a POST request to the specified URL (which can be a FastAPI endpoint or any other HTTP server).
- If `PROMPT_REWRITER_URL` is not set, the forwarding is simulated: the payload is logged and no real HTTP request is made. This is useful for development and testing.
- The forwarding function returns a dictionary indicating whether the forward was successful, the destination, and any errors encountered.

This design allows easy integration with downstream services and ensures reliability by logging errors and supporting simulation mode for development.
