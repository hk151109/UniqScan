from flask import Flask, request, jsonify
import os
import json
import logging
from datetime import datetime
import hashlib
from pathlib import Path
import traceback
from matcher import Text, Matcher

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'submissions'
REPORTS_FOLDER = 'reports'
DATABASE_FILE = 'similarity_database.json'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

class SimilarityDatabase:
    """Manages the database of submitted files and their metadata"""
    
    def __init__(self, db_file=DATABASE_FILE):
        self.db_file = db_file
        self.data = self.load_database()
    
    def load_database(self):
        """Load the database from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                return {"students": {}, "submissions": [], "metadata": {}}
        return {"students": {}, "submissions": [], "metadata": {}}
    
    def save_database(self):
        """Save the database to JSON file"""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_submission(self, student_id, student_name, file_path, assignment_id, checksum):
        """Add a new submission to the database"""
        submission = {
            "id": len(self.data["submissions"]),
            "student_id": student_id,
            "student_name": student_name,
            "file_path": file_path,
            "assignment_id": assignment_id,
            "checksum": checksum,
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }
        
        # Add to submissions list
        self.data["submissions"].append(submission)
        
        # Update student records
        if student_id not in self.data["students"]:
            self.data["students"][student_id] = {
                "name": student_name,
                "submissions": []
            }
        
        self.data["students"][student_id]["submissions"].append(submission["id"])
        self.save_database()
        return submission["id"]
    
    def get_all_submissions(self, exclude_student_id=None, assignment_id=None):
        """Get all submissions for comparison, optionally excluding a specific student"""
        submissions = []
        for sub in self.data["submissions"]:
            if exclude_student_id and sub["student_id"] == exclude_student_id:
                continue
            if assignment_id and sub["assignment_id"] != assignment_id:
                continue
            submissions.append(sub)
        return submissions
    
    def mark_processed(self, submission_id):
        """Mark a submission as processed"""
        if submission_id < len(self.data["submissions"]):
            self.data["submissions"][submission_id]["processed"] = True
            self.save_database()


class SimilarityChecker:
    """Handles similarity checking logic"""
    
    def __init__(self):
        self.db = SimilarityDatabase()
    
    def calculate_checksum(self, text_content):
        """Calculate checksum for text content"""
        return hashlib.md5(text_content.encode()).hexdigest()
    
    def read_file_content(self, file_path):
        """Read content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def compare_texts(self, text1, text2, label1, label2):
        """Compare two texts using the matcher algorithm"""
        try:
            text_obj1 = Text(text1, label1)
            text_obj2 = Text(text2, label2)
            
            matcher = Matcher(text_obj1, text_obj2, threshold=3, cutoff=5)
            
            # Calculate similarity percentage
            total_matches = sum(min(match.sizeA, match.sizeB) for match in matcher.extended_matches)
            total_tokens = min(len(text_obj1.tokens), len(text_obj2.tokens))
            
            if total_tokens == 0:
                return 0.0, []
            
            similarity_percentage = (total_matches / total_tokens) * 100
            
            # Extract match details for report
            match_details = []
            for match in matcher.extended_matches:
                match_details.append({
                    "start_a": match.a,
                    "start_b": match.b,
                    "size_a": match.sizeA,
                    "size_b": match.sizeB,
                    "similarity": min(match.sizeA, match.sizeB)
                })
            
            return similarity_percentage, match_details
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return 0.0, []
    
    def generate_similarity_report(self, student_name, assignment_id, similarity_results, file_content):
        """Generate HTML report for similarity analysis"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            max_similarity = max([result["similarity"] for result in similarity_results]) if similarity_results else 0
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Similarity Report - {student_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .score {{ font-size: 24px; font-weight: bold; }}
                    .high {{ color: #e74c3c; }}
                    .medium {{ color: #f39c12; }}
                    .low {{ color: #27ae60; }}
                    .match {{ margin: 10px 0; padding: 10px; border-left: 3px solid #3498db; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Similarity Analysis Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Student:</strong> {student_name}</p>
                    <p><strong>Assignment ID:</strong> {assignment_id}</p>
                    <p><strong>Analysis Date:</strong> {timestamp}</p>
                    <p><strong>Maximum Similarity Score:</strong> 
                        <span class="score {self.get_score_class(max_similarity)}">{max_similarity:.2f}%</span>
                    </p>
                </div>
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Compared With</th>
                        <th>Similarity Score (%)</th>
                        <th>Matches Found</th>
                    </tr>
            """
            
            for result in similarity_results:
                html_content += f"""
                    <tr>
                        <td>{result['compared_with']}</td>
                        <td class="{self.get_score_class(result['similarity'])}">{result['similarity']:.2f}%</td>
                        <td>{len(result['matches'])}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            # Save report
            report_filename = f"{student_name}_{assignment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_similarity_report.html"
            report_path = os.path.join(REPORTS_FOLDER, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def get_score_class(self, score):
        """Get CSS class based on similarity score"""
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"

# Initialize the similarity checker
similarity_checker = SimilarityChecker()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "similarity-detection"})

@app.route('/similarity/analyze', methods=['POST'])
def analyze_similarity():
    """
    Main endpoint for similarity analysis
    Expected JSON payload:
    {
        "student_id": "unique_student_id",
        "student_name": "Student Name",
        "file_path": "/path/to/student/file",
        "assignment_id": "assignment_identifier"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["student_id", "student_name", "file_path", "assignment_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        student_id = data["student_id"]
        student_name = data["student_name"]
        file_path = data["file_path"]
        assignment_id = data["assignment_id"]
        
        logger.info(f"Processing similarity analysis for {student_name} (ID: {student_id})")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Read file content
        file_content = similarity_checker.read_file_content(file_path)
        if file_content is None:
            return jsonify({"error": "Unable to read file content"}), 500
        
        # Calculate checksum
        checksum = similarity_checker.calculate_checksum(file_content)
        
        # Add to database
        submission_id = similarity_checker.db.add_submission(
            student_id, student_name, file_path, assignment_id, checksum
        )
        
        # Get all other submissions for comparison (excluding this student's submissions)
        other_submissions = similarity_checker.db.get_all_submissions(
            exclude_student_id=student_id, 
            assignment_id=assignment_id
        )
        
        # Perform similarity analysis
        similarity_results = []
        max_similarity = 0.0
        
        for other_sub in other_submissions:
            if not os.path.exists(other_sub["file_path"]):
                continue
                
            other_content = similarity_checker.read_file_content(other_sub["file_path"])
            if other_content is None:
                continue
            
            similarity_score, matches = similarity_checker.compare_texts(
                file_content, 
                other_content, 
                f"{student_name}_submission", 
                f"{other_sub['student_name']}_submission"
            )
            
            if similarity_score > max_similarity:
                max_similarity = similarity_score
            
            similarity_results.append({
                "compared_with": other_sub["student_name"],
                "student_id": other_sub["student_id"],
                "similarity": similarity_score,
                "matches": matches
            })
        
        # Generate report
        report_path = similarity_checker.generate_similarity_report(
            student_name, assignment_id, similarity_results, file_content
        )
        
        # Mark as processed
        similarity_checker.db.mark_processed(submission_id)
        
        # Return results
        response = {
            "student_id": student_id,
            "student_name": student_name,
            "assignment_id": assignment_id,
            "similarity_score": max_similarity,
            "total_comparisons": len(similarity_results),
            "detailed_results": similarity_results,
            "report_path": report_path,
            "submission_id": submission_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Similarity analysis completed for {student_name}: {max_similarity:.2f}%")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in similarity analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/similarity/database/stats', methods=['GET'])
def get_database_stats():
    """Get statistics about the similarity database"""
    try:
        stats = {
            "total_students": len(similarity_checker.db.data["students"]),
            "total_submissions": len(similarity_checker.db.data["submissions"]),
            "processed_submissions": len([s for s in similarity_checker.db.data["submissions"] if s.get("processed", False)]),
            "assignments": list(set([s["assignment_id"] for s in similarity_checker.db.data["submissions"]]))
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/similarity/reset', methods=['POST'])
def reset_database():
    """Reset the similarity database (use with caution!)"""
    try:
        # Backup current database
        if os.path.exists(DATABASE_FILE):
            backup_name = f"{DATABASE_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(DATABASE_FILE, backup_name)
        
        # Reset database
        similarity_checker.db.data = {"students": {}, "submissions": [], "metadata": {}}
        similarity_checker.db.save_database()
        
        return jsonify({"message": "Database reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
