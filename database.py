"""
DrainSight AI - SQLite Database Interface
Manages database initialization, storing inspection results,
fetching statistical metrics, and exporting CSV reports.
"""

import os
import sqlite3
import json
import csv
import io
from datetime import datetime
from config import Config

def get_db_connection():
    """Returns a connection to the SQLite database with row factory enabled."""
    os.makedirs(Config.INSTANCE_FOLDER, exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the analyses table in drainsight.db."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_type TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            output_filename TEXT NOT NULL,
            detected_objects TEXT,
            confidence_scores TEXT,
            blockage_percentage REAL NOT NULL,
            blockage_status TEXT NOT NULL,
            warning_message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(file_type, original_filename, stored_filename, output_filename,
                  detected_objects, confidence_scores, blockage_percentage,
                  blockage_status, warning_message):
    """Saves a single analysis record to the database and returns its new ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Store detected_objects and confidence_scores as JSON strings if they are dict/list
    if not isinstance(detected_objects, str):
        detected_objects = json.dumps(detected_objects)
    if not isinstance(confidence_scores, str):
        confidence_scores = json.dumps(confidence_scores)
        
    cursor.execute("""
        INSERT INTO analyses (
            file_type, original_filename, stored_filename, output_filename,
            detected_objects, confidence_scores, blockage_percentage,
            blockage_status, warning_message, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_type, original_filename, stored_filename, output_filename,
        detected_objects, confidence_scores, blockage_percentage,
        blockage_status, warning_message, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_analysis_by_id(analysis_id):
    """Retrieves an analysis by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        item = dict(row)
        try:
            item['detected_objects_parsed'] = json.loads(item['detected_objects'])
        except Exception:
            item['detected_objects_parsed'] = []
        try:
            item['confidence_scores_parsed'] = json.loads(item['confidence_scores'])
        except Exception:
            item['confidence_scores_parsed'] = {}
        return item
    return None

def get_recent_analyses(limit=50):
    """Retrieves recent analyses ordered by created_at descending."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        item = dict(r)
        try:
            item['detected_objects_parsed'] = json.loads(item['detected_objects'])
        except Exception:
            item['detected_objects_parsed'] = []
        results.append(item)
    return results

def get_dashboard_statistics():
    """Computes high-level aggregated statistics for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total counts by file type
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE file_type = 'image'")
    total_images = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE file_type = 'video'")
    total_videos = cursor.fetchone()[0]
    
    # Total counts by blockage status
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE blockage_status = 'Clear'")
    clear_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE blockage_status = 'Partially Blocked'")
    partial_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE blockage_status = 'Heavily Blocked'")
    heavy_count = cursor.fetchone()[0]
    
    # Aggregate waste types
    cursor.execute("SELECT detected_objects FROM analyses")
    rows = cursor.fetchall()
    
    waste_distribution = {
        'plastic_waste': 0,
        'leaves': 0,
        'paper_waste': 0,
        'mud_sediment': 0
    }
    
    for row in rows:
        raw_objs = row[0]
        if raw_objs:
            try:
                parsed = json.loads(raw_objs)
                if isinstance(parsed, list):
                    for obj in parsed:
                        if isinstance(obj, dict) and 'class_name' in obj:
                            cname = obj['class_name']
                        elif isinstance(obj, str):
                            cname = obj
                        else:
                            continue
                        if cname in waste_distribution:
                            waste_distribution[cname] += 1
                elif isinstance(parsed, dict):
                    for cname, cnt in parsed.items():
                        if cname in waste_distribution:
                            waste_distribution[cname] += cnt
            except Exception:
                pass

    conn.close()
    
    return {
        'total_inspections': total_images + total_videos,
        'total_images': total_images,
        'total_videos': total_videos,
        'clear_count': clear_count,
        'partial_count': partial_count,
        'heavy_count': heavy_count,
        'waste_distribution': waste_distribution
    }

def generate_csv_report():
    """Generates a CSV string containing all recent inspection records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, created_at, file_type, original_filename, blockage_percentage,
               blockage_status, detected_objects, warning_message
        FROM analyses ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Analysis ID", "Timestamp", "Media Type", "Original Filename",
        "Blockage (%)", "Blockage Status", "Detected Waste Types", "Warning Message"
    ])
    
    for r in rows:
        raw_objs = r["detected_objects"]
        waste_summary = []
        if raw_objs:
            try:
                parsed = json.loads(raw_objs)
                if isinstance(parsed, list):
                    for o in parsed:
                        cname = o.get('class_name', str(o)) if isinstance(o, dict) else str(o)
                        if cname != 'drain_opening':
                            waste_summary.append(cname)
                elif isinstance(parsed, dict):
                    waste_summary = [k for k, v in parsed.items() if k != 'drain_opening']
            except Exception:
                waste_summary = [str(raw_objs)]
                
        writer.writerow([
            r["id"],
            r["created_at"],
            r["file_type"].capitalize(),
            r["original_filename"],
            f"{r['blockage_percentage']:.1f}%",
            r["blockage_status"],
            "; ".join(set(waste_summary)) if waste_summary else "None",
            r["warning_message"]
        ])
        
    return output.getvalue()
