from flask import Flask, request, jsonify
import os
import json
import logging
from datetime import datetime
import traceback
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our custom modules
from similarity_api import SimilarityChecker
from ai_detection_api import AIContentDetector

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
similarity_checker = SimilarityChecker()
ai_detector = AIContentDetector()

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=3)

class UnifiedGradingService:
    """Unified service that combines similarity detection and AI detection"""
    
    def __init__(self):
        self.similarity_checker = similarity_checker
        self.ai_detector = ai_detector
    
    def process_submission(self, student_id, student_name, file_path, assignment_id, classroom_name=None):
        """
        Process a submission for both similarity and AI detection
        Returns unified results
        """
        try:
            results = {
                "student_id": student_id,
                "student_name": student_name,
                "file_path": file_path,
                "assignment_id": assignment_id,
                "classroom_name": classroom_name,
                "timestamp": datetime.now().isoformat(),
                "similarity_analysis": None,
                "ai_analysis": None,
                "unified_report_path": None,
                "status": "processing",
                "errors": []
            }
            
            # Check if file exists
            if not os.path.exists(file_path):
                results["status"] = "error"
                results["errors"].append("File not found")
                return results
            
            # Process similarity detection
            try:
                logger.info(f"Processing similarity analysis for {student_name}")
                
                # Read file content
                file_content = self.similarity_checker.read_file_content(file_path)
                if file_content is None:
                    raise Exception("Unable to read file content")
                
                # Calculate checksum
                checksum = self.similarity_checker.calculate_checksum(file_content)
                
                # Add to similarity database
                submission_id = self.similarity_checker.db.add_submission(
                    student_id, student_name, file_path, assignment_id, checksum
                )
                
                # Get other submissions for comparison
                other_submissions = self.similarity_checker.db.get_all_submissions(
                    exclude_student_id=student_id, 
                    assignment_id=assignment_id
                )
                
                # Perform similarity analysis
                similarity_results = []
                max_similarity = 0.0
                
                for other_sub in other_submissions:
                    if not os.path.exists(other_sub["file_path"]):
                        continue
                        
                    other_content = self.similarity_checker.read_file_content(other_sub["file_path"])
                    if other_content is None:
                        continue
                    
                    similarity_score, matches = self.similarity_checker.compare_texts(
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
                
                # Generate similarity report
                similarity_report_path = self.similarity_checker.generate_similarity_report(
                    student_name, assignment_id, similarity_results, file_content
                )
                
                results["similarity_analysis"] = {
                    "similarity_score": max_similarity,
                    "total_comparisons": len(similarity_results),
                    "detailed_results": similarity_results,
                    "report_path": similarity_report_path,
                    "submission_id": submission_id
                }
                
                self.similarity_checker.db.mark_processed(submission_id)
                logger.info(f"Similarity analysis completed: {max_similarity:.2f}%")
                
            except Exception as e:
                logger.error(f"Error in similarity analysis: {e}")
                results["errors"].append(f"Similarity analysis error: {str(e)}")
            
            # Process AI detection
            try:
                logger.info(f"Processing AI detection for {student_name}")
                
                # Ensure AI model is loaded
                if not self.ai_detector.model_loaded:
                    self.ai_detector.load_model()
                
                if not self.ai_detector.model_loaded:
                    raise Exception("AI detection model not available")
                
                # Read file content (reuse if already read)
                if 'file_content' not in locals():
                    file_content = self.ai_detector.read_file_content(file_path)
                    if file_content is None:
                        raise Exception("Unable to read file content")
                
                # Perform AI content detection
                ai_score, detailed_results = self.ai_detector.detect_ai_content(file_content)
                
                if ai_score is None:
                    raise Exception("AI detection failed")
                
                # Generate AI detection report
                ai_report_path = self.ai_detector.generate_ai_report(
                    student_name, assignment_id, ai_score, detailed_results
                )
                
                # Save to AI database
                result_id = self.ai_detector.db.add_result(
                    student_id, student_name, file_path, assignment_id, ai_score, detailed_results
                )
                
                results["ai_analysis"] = {
                    "ai_score": ai_score,
                    "ai_percentage": ai_score * 100,
                    "chunks_analyzed": len(detailed_results),
                    "detailed_results": detailed_results,
                    "report_path": ai_report_path,
                    "result_id": result_id,
                    "interpretation": self.ai_detector.get_interpretation(ai_score * 100)
                }
                
                logger.info(f"AI detection completed: {ai_score * 100:.2f}%")
                
            except Exception as e:
                logger.error(f"Error in AI detection: {e}")
                results["errors"].append(f"AI detection error: {str(e)}")
            
            # Generate unified report
            try:
                unified_report_path = self.generate_unified_report(results)
                results["unified_report_path"] = unified_report_path
            except Exception as e:
                logger.error(f"Error generating unified report: {e}")
                results["errors"].append(f"Unified report error: {str(e)}")
            
            # Set final status
            if results["similarity_analysis"] and results["ai_analysis"]:
                results["status"] = "completed"
            elif results["similarity_analysis"] or results["ai_analysis"]:
                results["status"] = "partial"
            else:
                results["status"] = "failed"
            
            return results
            
        except Exception as e:
            logger.error(f"Error in unified processing: {e}")
            logger.error(traceback.format_exc())
            return {
                "student_id": student_id,
                "student_name": student_name,
                "status": "error",
                "errors": [f"Processing failed: {str(e)}"]
            }
    
    def generate_unified_report(self, results):
        """Generate a unified HTML report combining similarity and AI detection results"""
        try:
            student_name = results["student_name"]
            assignment_id = results["assignment_id"]
            timestamp = results["timestamp"]
            
            # Create reports directory
            os.makedirs('reports', exist_ok=True)
            
            # Extract data
            similarity_score = results.get("similarity_analysis", {}).get("similarity_score", 0)
            ai_score = results.get("ai_analysis", {}).get("ai_percentage", 0)
            
            # Calculate overall risk score
            plagiarism_weight = 0.6
            ai_weight = 0.4
            overall_risk = (similarity_score * plagiarism_weight) + (ai_score * ai_weight)
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Academic Integrity Report - {student_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
                    .score-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }}
                    .score-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #e9ecef; }}
                    .score {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
                    .high {{ color: #e74c3c; }}
                    .medium {{ color: #f39c12; }}
                    .low {{ color: #27ae60; }}
                    .risk-indicator {{ 
                        width: 100%; 
                        height: 30px; 
                        background: linear-gradient(to right, #27ae60 0%, #f39c12 50%, #e74c3c 100%); 
                        border-radius: 15px; 
                        position: relative; 
                        margin: 20px 0;
                    }}
                    .risk-marker {{ 
                        position: absolute; 
                        top: -5px; 
                        width: 20px; 
                        height: 40px; 
                        background-color: #2c3e50; 
                        border-radius: 10px; 
                        transform: translateX(-50%);
                    }}
                    .details {{ margin: 20px 0; }}
                    .alert {{ padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    .alert-danger {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
                    .alert-warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
                    .alert-success {{ background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 2px solid #e9ecef; color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Academic Integrity Analysis Report</h1>
                    
                    <div class="summary">
                        <h2 style="color: white; border-bottom: 2px solid white;">📋 Summary</h2>
                        <p><strong>Student:</strong> {student_name}</p>
                        <p><strong>Assignment:</strong> {assignment_id}</p>
                        <p><strong>Analysis Date:</strong> {datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Status:</strong> {results.get('status', 'Unknown').title()}</p>
                    </div>
                    
                    <div class="score-grid">
                        <div class="score-card">
                            <h3>🔍 Similarity Score</h3>
                            <div class="score {self.get_risk_class(similarity_score)}">{similarity_score:.1f}%</div>
                            <p>Compared with {results.get('similarity_analysis', {}).get('total_comparisons', 0)} submissions</p>
                        </div>
                        <div class="score-card">
                            <h3>🤖 AI Detection</h3>
                            <div class="score {self.get_risk_class(ai_score)}">{ai_score:.1f}%</div>
                            <p>{results.get('ai_analysis', {}).get('interpretation', 'N/A')}</p>
                        </div>
                        <div class="score-card">
                            <h3>⚠️ Overall Risk</h3>
                            <div class="score {self.get_risk_class(overall_risk)}">{overall_risk:.1f}%</div>
                            <p>{self.get_risk_interpretation(overall_risk)}</p>
                        </div>
                    </div>
                    
                    <div class="risk-indicator">
                        <div class="risk-marker" style="left: {overall_risk}%;"></div>
                    </div>
                    <p style="text-align: center;"><strong>Overall Academic Integrity Risk Level</strong></p>
            """
            
            # Add similarity details if available
            if results.get("similarity_analysis"):
                similarity_data = results["similarity_analysis"]
                html_content += f"""
                    <div class="details">
                        <h2>🔍 Similarity Analysis Details</h2>
                        {self.get_alert_html(similarity_score, "similarity")}
                        
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Maximum Similarity Score</td><td>{similarity_score:.2f}%</td></tr>
                            <tr><td>Total Comparisons</td><td>{similarity_data.get('total_comparisons', 0)}</td></tr>
                            <tr><td>Matches Found</td><td>{len([r for r in similarity_data.get('detailed_results', []) if r.get('similarity', 0) > 0])}</td></tr>
                        </table>
                        
                        <h3>Top Similarities</h3>
                        <table>
                            <tr><th>Student</th><th>Similarity %</th><th>Matches</th></tr>
                """
                
                # Show top 5 similarities
                top_similarities = sorted(
                    similarity_data.get('detailed_results', []), 
                    key=lambda x: x.get('similarity', 0), 
                    reverse=True
                )[:5]
                
                for sim in top_similarities:
                    if sim.get('similarity', 0) > 0:
                        html_content += f"""
                            <tr>
                                <td>{sim.get('compared_with', 'Unknown')}</td>
                                <td class="{self.get_risk_class(sim.get('similarity', 0))}">{sim.get('similarity', 0):.2f}%</td>
                                <td>{len(sim.get('matches', []))}</td>
                            </tr>
                        """
                
                html_content += "</table></div>"
            
            # Add AI detection details if available
            if results.get("ai_analysis"):
                ai_data = results["ai_analysis"]
                html_content += f"""
                    <div class="details">
                        <h2>🤖 AI Detection Analysis</h2>
                        {self.get_alert_html(ai_score, "ai")}
                        
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>AI Content Probability</td><td>{ai_score:.2f}%</td></tr>
                            <tr><td>Text Chunks Analyzed</td><td>{ai_data.get('chunks_analyzed', 0)}</td></tr>
                            <tr><td>Interpretation</td><td>{ai_data.get('interpretation', 'N/A')}</td></tr>
                        </table>
                    </div>
                """
            
            # Add recommendations
            html_content += f"""
                <div class="details">
                    <h2>📝 Recommendations</h2>
                    {self.get_recommendations_html(similarity_score, ai_score, overall_risk)}
                </div>
                
                <div class="footer">
                    <p><strong>Disclaimer:</strong> This report is generated by automated analysis tools and should be used as a guide for further investigation. Manual review is recommended for definitive academic integrity decisions.</p>
                    <p><em>Generated by UniqScan Academic Integrity System</em></p>
                </div>
            </div>
            </body>
            </html>
            """
            
            # Save unified report
            report_filename = f"{student_name}_{assignment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_unified_report.html"
            report_path = os.path.join('reports', report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Unified report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating unified report: {e}")
            return None
    
    def get_risk_class(self, score):
        """Get CSS class based on risk score"""
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"
    
    def get_risk_interpretation(self, score):
        """Get risk interpretation"""
        if score >= 80:
            return "Very High Risk"
        elif score >= 60:
            return "High Risk"
        elif score >= 40:
            return "Moderate Risk"
        elif score >= 20:
            return "Low Risk"
        else:
            return "Very Low Risk"
    
    def get_alert_html(self, score, type_name):
        """Get alert HTML based on score and type"""
        if score >= 70:
            return f'<div class="alert alert-danger"><strong>High {type_name.title()} Detected:</strong> This submission requires immediate attention and manual review.</div>'
        elif score >= 40:
            return f'<div class="alert alert-warning"><strong>Moderate {type_name.title()} Detected:</strong> This submission should be reviewed for potential issues.</div>'
        else:
            return f'<div class="alert alert-success"><strong>Low {type_name.title()} Risk:</strong> This submission appears to have minimal {type_name} concerns.</div>'
    
    def get_recommendations_html(self, similarity_score, ai_score, overall_risk):
        """Get recommendations based on scores"""
        recommendations = []
        
        if overall_risk >= 70:
            recommendations.append("🚨 <strong>Immediate Action Required:</strong> Contact the student for discussion and possible disciplinary action.")
        elif overall_risk >= 40:
            recommendations.append("⚠️ <strong>Further Investigation:</strong> Review the submission manually and consider student interview.")
        else:
            recommendations.append("✅ <strong>Low Risk:</strong> Submission appears acceptable, routine monitoring sufficient.")
        
        if similarity_score >= 60:
            recommendations.append("📄 Review the specific text matches highlighted in the similarity report.")
            recommendations.append("👥 Compare with the identified similar submissions manually.")
        
        if ai_score >= 60:
            recommendations.append("🤖 Examine the text sections flagged as potentially AI-generated.")
            recommendations.append("💬 Consider asking the student to explain their writing process.")
        
        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html

# Initialize unified service
unified_service = UnifiedGradingService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "unified-grading-service",
        "similarity_service": "available",
        "ai_detection_service": "available",
        "ai_model_loaded": ai_detector.model_loaded
    })

@app.route('/grade/analyze', methods=['POST'])
def analyze_submission():
    """
    Unified endpoint for complete submission analysis
    Expected JSON payload:
    {
        "student_id": "unique_student_id",
        "student_name": "Student Name", 
        "file_path": "/path/to/student/file",
        "assignment_id": "assignment_identifier",
        "classroom_name": "Optional classroom name"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["student_id", "student_name", "file_path", "assignment_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        logger.info(f"Processing unified analysis for {data['student_name']}")
        
        # Process the submission
        results = unified_service.process_submission(
            student_id=data["student_id"],
            student_name=data["student_name"],
            file_path=data["file_path"],
            assignment_id=data["assignment_id"],
            classroom_name=data.get("classroom_name")
        )
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in unified analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/grade/batch', methods=['POST'])
def analyze_batch_submissions():
    """
    Analyze multiple submissions at once
    Expected JSON payload:
    {
        "submissions": [
            {
                "student_id": "id1",
                "student_name": "Name1",
                "file_path": "/path1",
                "assignment_id": "assignment1"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if "submissions" not in data or not isinstance(data["submissions"], list):
            return jsonify({"error": "Invalid payload: submissions list required"}), 400
        
        submissions = data["submissions"]
        logger.info(f"Processing batch analysis for {len(submissions)} submissions")
        
        results = []
        
        # Process submissions concurrently
        futures = []
        for submission in submissions:
            future = executor.submit(
                unified_service.process_submission,
                submission["student_id"],
                submission["student_name"],
                submission["file_path"],
                submission["assignment_id"],
                submission.get("classroom_name")
            )
            futures.append(future)
        
        # Collect results
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=300)  # 5 minute timeout per submission
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing submission {i}: {e}")
                results.append({
                    "status": "error",
                    "error": str(e),
                    "submission_index": i
                })
        
        return jsonify({
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r.get("status") == "completed"]),
            "partial": len([r for r in results if r.get("status") == "partial"]),
            "failed": len([r for r in results if r.get("status") in ["error", "failed"]])
        })
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stats', methods=['GET'])
def get_unified_stats():
    """Get comprehensive statistics"""
    try:
        similarity_stats = similarity_checker.db.data
        ai_stats = ai_detector.db.data
        
        return jsonify({
            "similarity_stats": {
                "total_students": len(similarity_stats.get("students", {})),
                "total_submissions": len(similarity_stats.get("submissions", [])),
                "processed_submissions": len([s for s in similarity_stats.get("submissions", []) if s.get("processed", False)])
            },
            "ai_detection_stats": {
                "total_analyses": len(ai_stats.get("results", [])),
                "high_ai_content": len([r for r in ai_stats.get("results", []) if r.get("ai_score", 0) > 0.7])
            },
            "model_status": {
                "ai_model_loaded": ai_detector.model_loaded
            }
        })
    except Exception as e:
        logger.error(f"Error getting unified stats: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Load AI model on startup
    logger.info("Starting unified grading service...")
    try:
        ai_detector.load_model()
    except Exception as e:
        logger.warning(f"Failed to load AI model on startup: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
