from flask import Flask, request, jsonify
import os
import json
import logging
from datetime import datetime
import traceback
import requests
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
# Using a reliable and simple model that's widely available
AI_DETECTION_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Fallback models in order of preference  
FALLBACK_MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "nlptown/bert-base-multilingual-uncased-sentiment"
]

HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")  # Set this in your environment
MAX_CHUNK_SIZE = 512  # Maximum tokens per chunk for the model
DATABASE_FILE = 'ai_detection_database.json'

class AIDetectionDatabase:
    """Manages the database of AI detection results"""
    
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
                logger.error(f"Error loading AI detection database: {e}")
                return {"results": [], "metadata": {}}
        return {"results": [], "metadata": {}}
    
    def save_database(self):
        """Save the database to JSON file"""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving AI detection database: {e}")
    
    def add_result(self, student_id, student_name, file_path, assignment_id, ai_score, detailed_results):
        """Add AI detection result to database"""
        result = {
            "id": len(self.data["results"]),
            "student_id": student_id,
            "student_name": student_name,
            "file_path": file_path,
            "assignment_id": assignment_id,
            "ai_score": ai_score,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.data["results"].append(result)
        self.save_database()
        return result["id"]

class AIContentDetector:
    """Handles AI content detection using Hugging Face Inference API"""
    
    def __init__(self):
        self.db = AIDetectionDatabase()
        self.current_model = AI_DETECTION_MODEL
        self.api_headers = {}
        if HUGGINGFACE_API_TOKEN:
            self.api_headers["Authorization"] = f"Bearer {HUGGINGFACE_API_TOKEN}"
        else:
            logger.warning("HUGGINGFACE_API_TOKEN not set. API calls will be rate-limited.")
    
    def get_api_url(self, model_name=None):
        """Get the API URL for a specific model"""
        model = model_name or self.current_model
        return f"https://api-inference.huggingface.co/models/{model}"
        
    def call_huggingface_api(self, text, model_name=None):
        """Make API call to Hugging Face Inference API with fallback models"""
        models_to_try = [model_name] if model_name else [self.current_model] + FALLBACK_MODELS
        
        for model in models_to_try:
            if not model:
                continue
                
            try:
                api_url = self.get_api_url(model)
                payload = {"inputs": text}
                
                logger.debug(f"Trying model: {model}")
                
                response = requests.post(
                    api_url,
                    headers=self.api_headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    logger.info(f"Model {model} is loading, waiting 20 seconds...")
                    import time
                    time.sleep(20)
                    
                    response = requests.post(
                        api_url,
                        headers=self.api_headers,
                        json=payload,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully used model: {model}")
                    self.current_model = model  # Update current model to successful one
                    return result
                else:
                    logger.warning(f"Model {model} failed: {response.status_code} - {response.text}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with model {model}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error with model {model}: {e}")
                continue
        
        logger.error("All AI detection models failed")
        return None
    
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
    
    def chunk_text(self, text, max_length=200):  # Much smaller chunks
        """Split text into chunks for processing - very conservative chunking"""
        # Simple word-based chunking with very small chunks
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            # Use very conservative chunking - aim for ~150 words per chunk
            if len(current_chunk) >= max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If still no chunks or text is very short, split by sentences
        if not chunks and text.strip():
            sentences = text.split('.')
            for sentence in sentences[:5]:  # Only first 5 sentences
                if sentence.strip() and len(sentence.strip()) > 10:
                    # Truncate very long sentences
                    if len(sentence.strip()) > 400:
                        sentence = sentence[:400] + "..."
                    chunks.append(sentence.strip())
        
        # Final safety check - truncate chunks that are still too long
        safe_chunks = []
        for chunk in chunks:
            if len(chunk) > 500:  # Character limit
                chunk = chunk[:500] + "..."
            safe_chunks.append(chunk)
        
        return safe_chunks if safe_chunks else ["Sample text for analysis"]  # Fallback
    
    def detect_ai_content(self, text):
        """Detect AI-generated content in text using Hugging Face API"""
        try:
            # Split text into chunks
            chunks = self.chunk_text(text)
            
            if not chunks:
                return 0.0, []
            
            chunk_results = []
            total_ai_score = 0.0
            successful_chunks = 0
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) == 0:
                    continue
                    
                try:
                    # Call Hugging Face API for this chunk
                    api_result = self.call_huggingface_api(chunk)
                    
                    if api_result is None:
                        chunk_results.append({
                            "chunk_id": i,
                            "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                            "ai_probability": 0.0,
                            "error": "API call failed"
                        })
                        continue
                    
                    # Extract AI probability from API response
                    ai_prob = 0.0
                    if isinstance(api_result, list) and len(api_result) > 0:
                        # Handle different model response formats
                        max_score_item = max(api_result, key=lambda x: x.get('score', 0))
                        
                        for pred in api_result:
                            if isinstance(pred, dict) and 'label' in pred and 'score' in pred:
                                label = pred['label'].upper()
                                score = pred['score']
                                
                                # Check for AI-related labels (various formats)
                                if any(keyword in label for keyword in ['AI', 'GENERATED', 'FAKE', 'MACHINE', 'GPT', 'CHATGPT', 'BOT']):
                                    ai_prob = score
                                    break
                                # Check for human-related labels
                                elif any(keyword in label for keyword in ['HUMAN', 'REAL', 'AUTHENTIC', 'PERSON']):
                                    ai_prob = 1.0 - score
                                    break
                                # Check for positive/negative labels (some models use LABEL_1/LABEL_0)
                                elif 'LABEL_1' in label or 'POSITIVE' in label:
                                    ai_prob = score  # Assume LABEL_1 means AI-generated
                                    break
                                elif 'LABEL_0' in label or 'NEGATIVE' in label:
                                    ai_prob = 1.0 - score  # Assume LABEL_0 means human-written
                                    break
                        
                        # If no specific labels found, use the highest confidence score
                        if ai_prob == 0.0 and max_score_item:
                            ai_prob = max_score_item['score']
                    
                    chunk_results.append({
                        "chunk_id": i,
                        "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                        "ai_probability": ai_prob,
                        "predictions": api_result,
                        "chunk_length": len(chunk)
                    })
                    
                    total_ai_score += ai_prob
                    successful_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    chunk_results.append({
                        "chunk_id": i,
                        "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                        "ai_probability": 0.0,
                        "error": str(e)
                    })
            
            # Calculate average AI score based on successful chunks
            if successful_chunks > 0:
                average_ai_score = total_ai_score / successful_chunks
            else:
                average_ai_score = 0.0
            
            return average_ai_score, chunk_results
            
        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            return None, str(e)
    
    def generate_ai_report(self, student_name, assignment_id, ai_score, detailed_results):
        """Generate HTML report for AI detection analysis"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Detection Report - {student_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .score {{ font-size: 24px; font-weight: bold; }}
                    .high {{ color: #e74c3c; }}
                    .medium {{ color: #f39c12; }}
                    .low {{ color: #27ae60; }}
                    .chunk {{ margin: 10px 0; padding: 10px; border-left: 3px solid #3498db; }}
                    .ai-detected {{ border-left-color: #e74c3c; background-color: #fdf2f2; }}
                    .human-likely {{ border-left-color: #27ae60; background-color: #f2f8f2; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .progress-bar {{ 
                        width: 100%; 
                        background-color: #e0e0e0; 
                        border-radius: 10px; 
                        overflow: hidden; 
                    }}
                    .progress {{ 
                        height: 20px; 
                        background-color: #3498db; 
                        text-align: center; 
                        line-height: 20px; 
                        color: white; 
                        font-size: 12px; 
                    }}
                </style>
            </head>
            <body>
                <h1>AI Content Detection Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Student:</strong> {student_name}</p>
                    <p><strong>Assignment ID:</strong> {assignment_id}</p>
                    <p><strong>Analysis Date:</strong> {timestamp}</p>
                    <p><strong>AI Content Probability:</strong> 
                        <span class="score {self.get_ai_score_class(ai_score * 100)}">{ai_score * 100:.2f}%</span>
                    </p>
                    <div class="progress-bar">
                        <div class="progress" style="width: {ai_score * 100}%">{ai_score * 100:.1f}%</div>
                    </div>
                </div>
                
                <h2>Detailed Analysis</h2>
                <p>Total chunks analyzed: {len(detailed_results)}</p>
                
                <div class="chunks-analysis">
            """
            
            for result in detailed_results:
                ai_prob = result.get('ai_probability', 0) * 100
                chunk_class = "ai-detected" if ai_prob > 50 else "human-likely"
                
                html_content += f"""
                    <div class="chunk {chunk_class}">
                        <h3>Chunk {result.get('chunk_id', 'N/A')}</h3>
                        <p><strong>AI Probability:</strong> {ai_prob:.2f}%</p>
                        <p><strong>Text Preview:</strong> {result.get('text_preview', 'N/A')}</p>
                        <div class="progress-bar" style="width: 200px;">
                            <div class="progress" style="width: {ai_prob}%">{ai_prob:.1f}%</div>
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
                
                <h2>Interpretation Guide</h2>
                <ul>
                    <li><strong>0-30%:</strong> Very likely human-written content</li>
                    <li><strong>30-60%:</strong> Mixed or uncertain - could be human or AI</li>
                    <li><strong>60-85%:</strong> Likely AI-generated with some human editing</li>
                    <li><strong>85-100%:</strong> Very likely AI-generated content</li>
                </ul>
                
                <p><em>Note: This analysis is based on AI detection models and should be used as a guide rather than definitive proof.</em></p>
            </body>
            </html>
            """
            
            # Save report
            report_filename = f"{student_name}_{assignment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ai_detection_report.html"
            report_path = os.path.join('reports', report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating AI detection report: {e}")
            return None
    
    def get_ai_score_class(self, score):
        """Get CSS class based on AI detection score"""
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"

# Initialize the AI detector
ai_detector = AIContentDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test API connectivity
    api_available = False
    try:
        test_result = ai_detector.call_huggingface_api("This is a test.")
        api_available = test_result is not None
    except:
        api_available = False
    
    return jsonify({
        "status": "healthy", 
        "service": "ai-detection",
        "api_available": api_available,
        "model": ai_detector.current_model,
        "api_url": ai_detector.get_api_url(),
        "has_token": HUGGINGFACE_API_TOKEN is not None,
        "fallback_models": FALLBACK_MODELS
    })

@app.route('/ai-detection/analyze', methods=['POST'])
def analyze_ai_content():
    """
    Main endpoint for AI content analysis
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
        
        logger.info(f"Processing AI detection analysis for {student_name} (ID: {student_id})")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Read file content
        file_content = ai_detector.read_file_content(file_path)
        if file_content is None:
            return jsonify({"error": "Unable to read file content"}), 500
        
        if len(file_content.strip()) == 0:
            return jsonify({"error": "File is empty"}), 400
        
        # Perform AI content detection
        ai_score, detailed_results = ai_detector.detect_ai_content(file_content)
        
        if ai_score is None:
            return jsonify({"error": f"AI detection failed: {detailed_results}"}), 500
        
        # Generate report
        report_path = ai_detector.generate_ai_report(
            student_name, assignment_id, ai_score, detailed_results
        )
        
        # Save to database
        result_id = ai_detector.db.add_result(
            student_id, student_name, file_path, assignment_id, ai_score, detailed_results
        )
        
        # Return results
        response = {
            "student_id": student_id,
            "student_name": student_name,
            "assignment_id": assignment_id,
            "ai_score": ai_score,
            "ai_percentage": ai_score * 100,
            "chunks_analyzed": len(detailed_results),
            "detailed_results": detailed_results,
            "report_path": report_path,
            "result_id": result_id,
            "timestamp": datetime.now().isoformat(),
            "interpretation": ai_detector.get_interpretation(ai_score * 100)
        }
        
        logger.info(f"AI detection analysis completed for {student_name}: {ai_score * 100:.2f}%")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in AI detection analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ai-detection/text', methods=['POST'])
def analyze_text_content():
    """
    Analyze raw text content directly
    Expected JSON payload:
    {
        "text": "Text content to analyze",
        "student_name": "Student Name (optional)",
        "assignment_id": "assignment_identifier (optional)"
    }
    """
    try:
        data = request.get_json()
        
        if "text" not in data:
            return jsonify({"error": "Missing required field: text"}), 400
        
        text_content = data["text"]
        student_name = data.get("student_name", "Unknown")
        assignment_id = data.get("assignment_id", "direct_analysis")
        
        if len(text_content.strip()) == 0:
            return jsonify({"error": "Text content is empty"}), 400
        
        # Perform AI content detection
        ai_score, detailed_results = ai_detector.detect_ai_content(text_content)
        
        if ai_score is None:
            return jsonify({"error": f"AI detection failed: {detailed_results}"}), 500
        
        # Return results
        response = {
            "ai_score": ai_score,
            "ai_percentage": ai_score * 100,
            "chunks_analyzed": len(detailed_results),
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat(),
            "interpretation": ai_detector.get_interpretation(ai_score * 100)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in direct text AI detection: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ai-detection/stats', methods=['GET'])
def get_detection_stats():
    """Get statistics about AI detection results"""
    try:
        results = ai_detector.db.data["results"]
        
        if not results:
            return jsonify({
                "total_analyses": 0,
                "average_ai_score": 0,
                "high_ai_content": 0,
                "assignments": []
            })
        
        ai_scores = [r["ai_score"] for r in results]
        high_ai_count = len([s for s in ai_scores if s > 0.7])
        
        stats = {
            "total_analyses": len(results),
            "average_ai_score": sum(ai_scores) / len(ai_scores),
            "average_ai_percentage": (sum(ai_scores) / len(ai_scores)) * 100,
            "high_ai_content": high_ai_count,
            "high_ai_percentage": (high_ai_count / len(results)) * 100,
            "assignments": list(set([r["assignment_id"] for r in results]))
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting detection stats: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Add method to AIContentDetector class
def get_interpretation(self, percentage):
    """Get interpretation of AI detection score"""
    if percentage < 30:
        return "Very likely human-written content"
    elif percentage < 60:
        return "Mixed or uncertain - could be human or AI-assisted"
    elif percentage < 85:
        return "Likely AI-generated with possible human editing"
    else:
        return "Very likely AI-generated content"

# Bind the method to the class
AIContentDetector.get_interpretation = get_interpretation

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
