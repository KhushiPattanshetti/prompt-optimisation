"""
main.py
Flask application — ICD-10 User Instruction Parser Microservice
"""

import logging
import os
import sys

# ── Absolute base dir — works regardless of CWD or how Flask is launched ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")

from flask import Flask, jsonify, request, send_from_directory
from app.validator import validate_request
from app.storage import store_instruction, get_all_instructions
from app.forwarder import forward_to_prompt_rewriter

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info("BASE_DIR  : %s", BASE_DIR)
logger.info("STATIC_DIR: %s (exists=%s)", STATIC_DIR, os.path.exists(STATIC_DIR))
logger.info("index.html: exists=%s", os.path.exists(os.path.join(STATIC_DIR, "index.html")))


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)  # disable Flask default /static
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/api/validate", methods=["OPTIONS"])
    def validate_preflight():
        return "", 204

    @app.route("/api/validate", methods=["POST"])
    def validate():
        logger.info("Incoming /api/validate request")

        user_instruction = ""
        filename = ""
        file_content = ""

        try:
            if request.content_type and "multipart/form-data" in request.content_type:
                user_instruction = request.form.get("user_instruction", "").strip()
                uploaded_file = request.files.get("clinical_document")

                if uploaded_file:
                    filename = uploaded_file.filename or ""
                    try:
                        file_content = uploaded_file.read().decode("utf-8", errors="replace")
                    except Exception:
                        file_content = ""

            else:
                body = request.get_json(silent=True) or {}
                user_instruction = (body.get("user_instruction") or "").strip()
                filename = (body.get("filename") or "").strip()
                file_content = body.get("clinical_note") or ""

            logger.info(
                "Validating — instruction: %d chars, file: %s, note: %d chars",
                len(user_instruction or ""),
                filename,
                len(file_content or ""),
            )

            result = validate_request(user_instruction, filename, file_content)

            if not result.ok:
                logger.info("Validation FAILED: %s", result.error)
                return jsonify(result.to_dict()), 422

            logger.info("Validation PASSED")

            stored_record = store_instruction(user_instruction)
            instruction_id = stored_record["instruction_id"]

            logger.info("Forwarding valid request to rewriter service")
            forward_result = forward_to_prompt_rewriter(
                filename=filename,
                clinical_note=file_content,
                instruction_id=instruction_id,
            )

            return jsonify({
                **result.to_dict(),
                "instruction_id": instruction_id,
                "stored_record": stored_record,
                "forward_result": forward_result,
            }), 200

        except Exception as e:
            logger.exception("Unhandled error during /api/validate")
            return jsonify({
                "ok": False,
                "error": f"Internal server error: {str(e)}"
            }), 500

    @app.route("/api/instructions", methods=["GET"])
    def list_instructions():
        instructions = get_all_instructions()
        return jsonify({"count": len(instructions), "instructions": instructions}), 200

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "service": "icd10-user-instruction-parser",
            "rewriter_url": os.environ.get("PROMPT_REWRITER_URL", "http://127.0.0.1:8000/rewrite"),
        }), 200

    @app.route("/", methods=["GET"])
    def ui():
        logger.info("Serving UI from: %s", STATIC_DIR)
        return send_from_directory(STATIC_DIR, "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
