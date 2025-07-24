"""
Flask API Application for Vietnamese History RAG System
======================================================

This Flask application provides REST API endpoints for a Vietnamese History 
Retrieval-Augmented Generation (RAG) system with two main functionalities:

1. History Chat: Query Vietnamese history database using RAG pipeline
2. PDF Chat: Upload and query PDF documents using conversational RAG

The application handles file uploads, processes user queries, and returns
structured responses with answers and source URLs.
"""

import os
import sys
import tempfile
import logging
from typing import List
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import services
from services.rag_service import RAGService
from pdf_services.pdf_rag_service import PDFRAGService

# Configure logging with UTF-8 encoding support to handle Vietnamese characters
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)


# Function to safely log Vietnamese text
def safe_log_info(message):
    """Safely log messages that may contain Vietnamese characters."""
    try:
        logger.info(message)
    except UnicodeEncodeError:
        # Fallback: encode Vietnamese characters safely
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        logger.info(f"[Vietnamese text] {safe_message}")


def safe_log_error(message):
    """Safely log error messages that may contain Vietnamese characters."""
    try:
        logger.error(message)
    except UnicodeEncodeError:
        # Fallback: encode Vietnamese characters safely
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        logger.error(f"[Vietnamese text] {safe_message}")


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global service instances
rag_service = None
pdf_rag_service = None


def initialize_services():
    """Initialize RAG services on application startup."""
    global rag_service, pdf_rag_service

    try:
        logger.info("Initializing RAG services...")
        rag_service = RAGService()
        pdf_rag_service = PDFRAGService()
        logger.info("All services initialized successfully")
    except Exception as e:
        safe_log_error(f"Failed to initialize services: {str(e)}")
        raise


def allowed_file(filename: str) -> bool:
    """Check if uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def process_urls(urls) -> List[str]:
    """
    Process URLs from RAG service response.
    
    Args:
        urls: Can be None, string, or list of strings
        
    Returns:
        List of unique URLs or empty list
    """
    if urls is None:
        return []

    if isinstance(urls, str):
        return [urls]

    if isinstance(urls, list):
        # Convert to set to remove duplicates, then back to list
        unique_urls = list(set(urls))
        return unique_urls

    return []


@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/api/rag-chat', methods=['POST'])
def rag_chat():
    """
    Handle history chat requests using RAG service.
    
    Expected JSON payload:
    {
        "message": "User question about Vietnamese history",
        "timestamp": "ISO timestamp"
    }
    
    Returns:
    {
        "success": true/false,
        "answer": "AI response",
        "source_urls": ["url1", "url2", ...] or [],
        "error": "Error message if failed"
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request must be JSON"
            }), 400

        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'message' field"
            }), 400

        user_message = data['message'].strip()

        if not user_message:
            return jsonify({
                "success": False,
                "error": "Message cannot be empty"
            }), 400

        # Safely log Vietnamese text
        safe_log_info(f"Processing history chat request: {user_message[:100]}...")

        # Process query using RAG service
        result = rag_service.handling_query(user_message)

        # Extract answer and URLs
        answer = result.get("Answer", "")
        urls = result.get("URL")

        # Process URLs according to requirements
        processed_urls = process_urls(urls)

        safe_log_info(f"History chat response generated successfully. URLs: {len(processed_urls)}")

        return jsonify({
            "success": True,
            "answer": answer,
            "source_urls": processed_urls
        })

    except Exception as e:
        safe_log_error(f"Error in rag_chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error occurred"
        }), 500


@app.route('/api/pdf-rag-chat', methods=['POST'])
def pdf_rag_chat():
    """
    Handle PDF chat requests with file upload support.
    
    Expected form data:
    - message: User question about PDF content
    - timestamp: ISO timestamp
    - pdf_0, pdf_1, ...: PDF files
    
    Returns:
    {
        "success": true/false,
        "answer": "AI response",
        "source_urls": [] (always empty for PDF chat),
        "error": "Error message if failed"
    }
    """
    try:
        # Validate request
        if 'message' not in request.form:
            return jsonify({
                "success": False,
                "error": "Missing 'message' field"
            }), 400

        user_message = request.form['message'].strip()

        if not user_message:
            return jsonify({
                "success": False,
                "error": "Message cannot be empty"
            }), 400

        # Get uploaded PDF files
        pdf_files = []
        uploaded_files = []

        for key in request.files:
            if key.startswith('pdf_'):
                file = request.files[key]
                if file and file.filename and allowed_file(file.filename):
                    pdf_files.append(file)

        if not pdf_files:
            return jsonify({
                "success": False,
                "error": "No valid PDF files uploaded"
            }), 400

        # Safely log Vietnamese text
        safe_log_info(f"Processing PDF chat request with {len(pdf_files)} files: {user_message[:100]}...")

        # Save uploaded files temporarily
        temp_paths = []

        try:
            for file in pdf_files:
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                temp_paths.append(temp_path)
                uploaded_files.append(filename)

            logger.info(f"Saved {len(temp_paths)} PDF files: {uploaded_files}")

            # Process PDFs and generate answer
            pdf_rag_service.process_pdf(temp_paths)
            answer = pdf_rag_service.generate_answer(user_message)

            logger.info("PDF chat response generated successfully")

            return jsonify({
                "success": True,
                "answer": answer,
                "source_urls": []  # PDF chat doesn't return URLs
            })

        finally:
            # Clean up temporary files
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup {temp_path}: {cleanup_error}")

    except Exception as e:
        safe_log_error(f"Error in pdf_rag_chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error occurred"
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "services": {
            "rag_service": rag_service is not None,
            "pdf_rag_service": pdf_rag_service is not None
        }
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 50MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    safe_log_error(f"Internal server error: {str(e)}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    try:
        # Initialize services
        initialize_services()

        # Start the Flask application
        logger.info("Starting Flask application...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )

    except Exception as e:
        safe_log_error(f"Failed to start application: {str(e)}")
        raise
