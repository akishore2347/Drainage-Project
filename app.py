"""
DrainSight AI – Intelligent Drainage Monitoring System
Flask Backend Application
"""

import os
import uuid
import json
import logging
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_from_directory, Response, jsonify
)
from werkzeug.utils import secure_filename

from config import Config
from detector import DrainageDetector
import database as db

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Ensure required folders exist
for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.INSTANCE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize Database
db.init_db()

# Initialize AI Detector
detector = DrainageDetector()

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# -------------------------------------------------------------
# Context Processor for Global Template Variables
# -------------------------------------------------------------
@app.context_processor
def inject_global_vars():
    return {
        'current_year': datetime.now().year,
        'app_name': 'DrainSight AI',
        'is_placeholder_model': detector.is_placeholder,
        'placeholder_message': detector.status_message if detector.is_placeholder else None
    }

# -------------------------------------------------------------
# Web Routes
# -------------------------------------------------------------

@app.route('/')
def index():
    """Home Page with Hero, Overview, and Quick Actions."""
    stats = db.get_dashboard_statistics()
    return render_template('index.html', stats=stats)

@app.route('/upload-image')
def upload_image():
    """Image Upload and Analysis Page."""
    return render_template('upload_image.html')

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Handles Image Upload, OpenCV Preprocessing, YOLO Inference, and DB Storage."""
    if 'image' not in request.files:
        flash('No image file selected. Please choose an image to upload.', 'danger')
        return redirect(url_for('upload_image'))

    file = request.files['image']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('upload_image'))

    if not allowed_file(file.filename, Config.ALLOWED_IMAGE_EXTENSIONS):
        flash(f'Invalid file type. Allowed formats: {", ".join(Config.ALLOWED_IMAGE_EXTENSIONS).upper()}', 'danger')
        return redirect(url_for('upload_image'))

    try:
        # Secure unique filename
        original_name = secure_filename(file.filename) or 'drain_image.jpg'
        ext = original_name.rsplit('.', 1)[1].lower()
        unique_id = uuid.uuid4().hex
        stored_name = f"img_{unique_id}.{ext}"
        output_name = f"annotated_{unique_id}.jpg"

        input_path = os.path.join(Config.UPLOAD_FOLDER, stored_name)
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_name)

        # Save uploaded file
        file.save(input_path)
        
        # Check file size (10 MB limit)
        if os.path.getsize(input_path) > Config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            os.remove(input_path)
            flash(f'File size exceeds {Config.MAX_IMAGE_SIZE_MB}MB limit.', 'danger')
            return redirect(url_for('upload_image'))

        # Run AI detection & OpenCV pipeline
        result = detector.process_image_file(input_path, output_path)
        assessment = result['assessment']
        detections = result['detections']

        # Extract waste types and confidence scores
        waste_objs = [d['class_name'] for d in detections if d['class_id'] > 0]
        conf_scores = {d['class_name']: d['confidence'] for d in detections}

        # Save to SQLite Database
        analysis_id = db.save_analysis(
            file_type='image',
            original_filename=original_name,
            stored_filename=stored_name,
            output_filename=output_name,
            detected_objects=waste_objs,
            confidence_scores=conf_scores,
            blockage_percentage=assessment['blockage_percentage'],
            blockage_status=assessment['status'],
            warning_message=assessment['warning']
        )

        flash('Image analyzed successfully!', 'success')
        return redirect(url_for('result_page', analysis_id=analysis_id))

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('upload_image'))

@app.route('/analyze-sample/<sample_name>', methods=['POST', 'GET'])
def analyze_sample(sample_name):
    """Allows one-click demo testing using pre-bundled drainage samples."""
    valid_samples = {
        'clear': ('sample_clear_drain.jpg', 'Sample Clear Drain'),
        'partial': ('sample_partial_blockage.jpg', 'Sample Partial Blockage'),
        'heavy': ('sample_heavy_blockage.jpg', 'Sample Heavy Blockage')
    }

    if sample_name not in valid_samples:
        flash('Invalid sample requested.', 'danger')
        return redirect(url_for('upload_image'))

    filename, display_name = valid_samples[sample_name]
    sample_src = os.path.join(Config.BASE_DIR, 'static', 'images', 'samples', filename)

    if not os.path.exists(sample_src):
        flash('Sample image file not found on server.', 'danger')
        return redirect(url_for('upload_image'))

    try:
        unique_id = uuid.uuid4().hex
        stored_name = f"sample_{sample_name}_{unique_id}.jpg"
        output_name = f"annotated_sample_{unique_id}.jpg"

        input_path = os.path.join(Config.UPLOAD_FOLDER, stored_name)
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_name)

        # Copy sample to uploads
        import shutil
        shutil.copy(sample_src, input_path)

        # Process image
        result = detector.process_image_file(input_path, output_path)
        assessment = result['assessment']
        detections = result['detections']

        waste_objs = [d['class_name'] for d in detections if d['class_id'] > 0]
        conf_scores = {d['class_name']: d['confidence'] for d in detections}

        analysis_id = db.save_analysis(
            file_type='image',
            original_filename=display_name,
            stored_filename=stored_name,
            output_filename=output_name,
            detected_objects=waste_objs,
            confidence_scores=conf_scores,
            blockage_percentage=assessment['blockage_percentage'],
            blockage_status=assessment['status'],
            warning_message=assessment['warning']
        )

        flash(f'Analyzed {display_name} successfully!', 'success')
        return redirect(url_for('result_page', analysis_id=analysis_id))

    except Exception as e:
        logger.error(f"Error processing sample: {str(e)}", exc_info=True)
        flash(f'Error analyzing sample: {str(e)}', 'danger')
        return redirect(url_for('upload_image'))

@app.route('/upload-video')
def upload_video():
    """Video Upload and Analysis Page."""
    return render_template('upload_video.html')

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    """Handles Video Upload, Frame Extraction, YOLO Inference, and Video Generation."""
    if 'video' not in request.files:
        flash('No video file selected. Please choose a video to upload.', 'danger')
        return redirect(url_for('upload_video'))

    file = request.files['video']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('upload_video'))

    if not allowed_file(file.filename, Config.ALLOWED_VIDEO_EXTENSIONS):
        flash(f'Invalid video format. Allowed formats: {", ".join(Config.ALLOWED_VIDEO_EXTENSIONS).upper()}', 'danger')
        return redirect(url_for('upload_video'))

    try:
        original_name = secure_filename(file.filename) or 'drain_video.mp4'
        ext = original_name.rsplit('.', 1)[1].lower()
        unique_id = uuid.uuid4().hex
        stored_name = f"vid_{unique_id}.{ext}"
        output_name = f"annotated_{unique_id}.mp4"

        input_path = os.path.join(Config.UPLOAD_FOLDER, stored_name)
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_name)

        file.save(input_path)

        # Check video size (50 MB limit)
        if os.path.getsize(input_path) > Config.MAX_VIDEO_SIZE_MB * 1024 * 1024:
            os.remove(input_path)
            flash(f'Video exceeds {Config.MAX_VIDEO_SIZE_MB}MB size limit.', 'danger')
            return redirect(url_for('upload_video'))

        # Sample rate from form or default to 4 (processes every 4th frame for high performance)
        sample_rate = int(request.form.get('sample_rate', 4))
        result = detector.process_video_file(input_path, output_path, sample_rate=sample_rate)

        # Save analysis to database
        analysis_id = db.save_analysis(
            file_type='video',
            original_filename=original_name,
            stored_filename=stored_name,
            output_filename=output_name,
            detected_objects=result['detected_waste_types'],
            confidence_scores={'peak_blockage': result['peak_blockage']},
            blockage_percentage=result['peak_blockage'],
            blockage_status=result['final_status'],
            warning_message=result['warning']
        )

        flash('Video processed and annotated successfully!', 'success')
        return redirect(url_for('result_page', analysis_id=analysis_id))

    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
        flash(f'Error processing video: {str(e)}', 'danger')
        return redirect(url_for('upload_video'))

@app.route('/result/<int:analysis_id>')
def result_page(analysis_id):
    """Detection Result Presentation Page."""
    record = db.get_analysis_by_id(analysis_id)
    if not record:
        flash('Analysis record not found.', 'warning')
        return redirect(url_for('dashboard'))

    return render_template('result.html', record=record)

@app.route('/dashboard')
def dashboard():
    """Dashboard Page with Analytics, Charts, and History Table."""
    stats = db.get_dashboard_statistics()
    recent_history = db.get_recent_analyses(limit=50)
    return render_template('dashboard.html', stats=stats, history=recent_history)

@app.route('/how-it-works')
def how_it_works():
    """Detailed Explanation of Architecture, Computer Vision & YOLO Workflow."""
    return render_template('how_it_works.html')

@app.route('/about')
def about():
    """About Page covering Problem Statement, SIMATS Literature, and Future Scope."""
    return render_template('about.html')

@app.route('/download-report')
def download_report():
    """Exports Inspection History as a CSV Report."""
    csv_data = db.generate_csv_report()
    filename = f"drainsight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

@app.route('/downloads/<filename>')
def download_file(filename):
    """Provides secure download of processed annotated output files."""
    return send_from_directory(Config.OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serves original uploaded files securely for preview."""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serves processed output files securely for preview."""
    return send_from_directory(Config.OUTPUT_FOLDER, filename)

@app.route('/api/stats')
def api_stats():
    """JSON API endpoint for dashboard chart updates."""
    stats = db.get_dashboard_statistics()
    return jsonify(stats)

# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(error):
    return render_template('base.html', error_title="404 - Page Not Found",
                           error_msg="The page you requested could not be found."), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File exceeds the maximum allowed upload size (50MB for video, 10MB for image).', 'danger')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_error(error):
    return render_template('base.html', error_title="500 - Server Error",
                           error_msg="An internal error occurred during processing. Please try again."), 500

# -------------------------------------------------------------
# Main Application Runner
# -------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1']
    logger.info(f"Starting DrainSight AI web application on port {port} (debug={debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
