import os
import json
import logging
import nltk
import time
import random
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import re
from difflib import SequenceMatcher
from nltk.metrics.distance import edit_distance as editDistance
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import ngrams
from termcolor import colored
import textwrap
from urllib.parse import urlparse
import PyPDF2
import docx
from io import BytesIO


app = Flask(__name__)
CORS(app)

# Configuration
HOMEWORK_FOLDER = 'homework'
DATABASE_FILE = 'plagiarism_database.json'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'py', 'java', 'cpp', 'c', 'js', 'html', 'css', 'md'}

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary directories
os.makedirs(HOMEWORK_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download NLTK resources if not already downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_resources()


class Text:
    def __init__(self, raw_text, label, filepath, removeStopwords=True):
        if isinstance(raw_text, list):
            self.text = ' \n '.join(raw_text)
        else:
            self.text = raw_text
        self.label = label
        self.filepath = filepath
        self.preprocess(self.text)
        self.tokens = self.getTokens(removeStopwords)
        self.trigrams = self.ngrams(3)
        self.checksum = self.calculate_checksum()

    def calculate_checksum(self):
        return hashlib.md5(self.text.encode()).hexdigest()

    def preprocess(self, text):
        self.text = re.sub(r'([A-Za-z])- ([a-z])', r'\1\2', text)

    def getTokens(self, removeStopwords=True):
        tokenizer = nltk.RegexpTokenizer('[a-zA-Z]\\w+\'?\\w*')
        spans = list(tokenizer.span_tokenize(self.text))
        
        if spans:
            self.length = spans[-1][-1]
        else:
            self.length = 0
            
        tokens = tokenizer.tokenize(self.text)
        tokens = [token.lower() for token in tokens]
        
        stemmer = LancasterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        if not removeStopwords:
            self.spans = spans
            return tokens
            
        tokenSpans = list(zip(tokens, spans))
        stopwords = nltk.corpus.stopwords.words('english')
        tokenSpans = [token for token in tokenSpans if token[0] not in stopwords]
        self.spans = [x[1] for x in tokenSpans]
        return [x[0] for x in tokenSpans]

    def ngrams(self, n):
        return list(ngrams(self.tokens, n))


class ExtendedMatch:
    def __init__(self, a, b, sizeA, sizeB):
        self.a = a
        self.b = b
        self.sizeA = sizeA
        self.sizeB = sizeB
        self.healed = False
        self.extendedBackwards = 0
        self.extendedForwards = 0

    def __repr__(self):
        out = "a: %s, b: %s, size a: %s, size b: %s" % (self.a, self.b, self.sizeA, self.sizeB)
        if self.extendedBackwards:
            out += ", extended backwards x%s" % self.extendedBackwards
        if self.extendedForwards:
            out += ", extended forwards x%s" % self.extendedForwards
        if self.healed:
            out += ", healed"
        return out


class Matcher:
    def __init__(self, textObjA, textObjB, threshold=3, cutoff=5, ngramSize=3, minDistance=8, silent=True):
        self.threshold = threshold
        self.ngramSize = ngramSize
        self.minDistance = minDistance
        self.silent = silent

        self.textA = textObjA
        self.textB = textObjB

        self.textAgrams = self.textA.ngrams(ngramSize)
        self.textBgrams = self.textB.ngrams(ngramSize)

        self.locationsA = []
        self.locationsB = []
        self.match_texts = []

        self.initial_matches = self.get_initial_matches()
        self.healed_matches = self.heal_neighboring_matches()
        self.extended_matches = self.extend_matches()
        self.extended_matches = [match for match in self.extended_matches
                                if min(match.sizeA, match.sizeB) >= cutoff]

        self.numMatches = len(self.extended_matches)
        self.similarity_score = self.calculate_similarity()

    def calculate_similarity(self):
        if not self.extended_matches:
            return 0.0
            
        total_matched_tokens_A = sum(match.sizeA for match in self.extended_matches)
        total_tokens_A = len(self.textA.tokens)
        
        if total_tokens_A == 0:
            return 0.0
            
        similarity = (total_matched_tokens_A / total_tokens_A) * 100
        return round(similarity, 2)

    def get_initial_matches(self):
        sequence = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        matchingBlocks = sequence.get_matching_blocks()
        return [match for match in matchingBlocks if match.size > self.threshold]

    def getTokensText(self, text, start, length):
        if start < 0:
            start = 0
        
        matchTokens = text.tokens[start:start + length]
        
        if start >= len(text.spans) or start + length > len(text.spans):
            return ""
            
        spans = text.spans[start:start + length]
        if len(spans) == 0:
            return ""
        else:
            passage = text.text[spans[0][0]:spans[-1][-1]]
        return passage

    def getLocations(self, text, start, length, asPercentages=False):
        if start >= len(text.spans) or start + length > len(text.spans):
            return None
            
        spans = text.spans[start:start + length]
        if len(spans) == 0:
            return None
            
        if asPercentages:
            locations = (spans[0][0] / text.length, spans[-1][-1] / text.length)
        else:
            try:
                locations = (spans[0][0], spans[-1][-1])
            except IndexError:
                return None
        return locations

    def getMatch(self, match, citation_number, context=5):
        textA, textB = self.textA, self.textB
        lengthA = match.sizeA + self.ngramSize - 1
        lengthB = match.sizeB + self.ngramSize - 1
        
        matched_text = self.getTokensText(textA, match.a, lengthA)
        spansA = self.getLocations(textA, match.a, lengthA)
        spansB = self.getLocations(textB, match.b, lengthB)
        
        if spansA is not None and spansB is not None:
            self.locationsA.append(spansA)
            self.locationsB.append(spansB)
            
            self.match_texts.append({
                "text": matched_text,
                "citation": citation_number,
                "spans": spansA,
                "source_file": textB.label
            })
            
            return {
                "matched_text": matched_text,
                "source_file": textB.label,
                "similarity": self.similarity_score
            }
        return None

    def heal_neighboring_matches(self):
        healedMatches = []
        ignoreNext = False
        matches = self.initial_matches.copy()
        
        if len(matches) == 1:
            match = matches[0]
            sizeA, sizeB = match.size, match.size
            match = ExtendedMatch(match.a, match.b, sizeA, sizeB)
            healedMatches.append(match)
            return healedMatches
            
        for i, match in enumerate(matches):
            if i + 1 > len(matches) - 1:
                break
            nextMatch = matches[i + 1]
            
            if ignoreNext:
                ignoreNext = False
                continue
            else:
                if (nextMatch.a - (match.a + match.size)) < self.minDistance:
                    sizeA = (nextMatch.a + nextMatch.size) - match.a
                    sizeB = (nextMatch.b + nextMatch.size) - match.b
                    healed = ExtendedMatch(match.a, match.b, sizeA, sizeB)
                    healed.healed = True
                    healedMatches.append(healed)
                    ignoreNext = True
                else:
                    sizeA, sizeB = match.size, match.size
                    match = ExtendedMatch(match.a, match.b, sizeA, sizeB)
                    healedMatches.append(match)
        return healedMatches

    def edit_ratio(self, wordA, wordB):
        distance = editDistance(wordA, wordB)
        averageLength = (len(wordA) + len(wordB)) / 2
        return distance / averageLength

    def extend_matches(self, cutoff=0.4):
        extended = False
        for match in self.healed_matches:
            if match.a > 0 and match.b > 0 and len(self.textAgrams) > match.a - 1 and len(self.textBgrams) > match.b - 1:
                wordA = self.textAgrams[(match.a - 1)][0]
                wordB = self.textBgrams[(match.b - 1)][0]
                if self.edit_ratio(wordA, wordB) < cutoff:
                    match.a -= 1
                    match.b -= 1
                    match.sizeA += 1
                    match.sizeB += 1
                    match.extendedBackwards += 1
                    extended = True
                    
            idxA = match.a + match.sizeA + 1
            idxB = match.b + match.sizeB + 1
            if idxA >= len(self.textAgrams) or idxB >= len(self.textBgrams):
                continue
                
            wordA = self.textAgrams[idxA][-1] if idxA < len(self.textAgrams) else ""
            wordB = self.textBgrams[idxB][-1] if idxB < len(self.textBgrams) else ""
            
            if wordA and wordB and self.edit_ratio(wordA, wordB) < cutoff:
                match.sizeA += 1
                match.sizeB += 1
                match.extendedForwards += 1
                extended = True

        if extended:
            self.extend_matches()

        return self.healed_matches

    def match(self):
        matches_info = []
        for num, match in enumerate(self.extended_matches):
            match_info = self.getMatch(match, num + 1)
            if match_info:
                matches_info.append(match_info)

        return self.numMatches, self.locationsA, self.locationsB, matches_info, self.match_texts, self.similarity_score


class PlagiarismDatabase:
    def __init__(self, db_file=DATABASE_FILE):
        self.db_file = db_file
        self.data = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Error loading database: {e}")
                return {"files": []}
        return {"files": []}

    def save_database(self):
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logging.error(f"Error saving database: {e}")

    def add_file(self, file_data):
        self.data["files"].append(file_data)
        self.save_database()

    def get_all_files(self):
        return self.data["files"]

    def get_file_by_id(self, file_id):
        for file_data in self.data["files"]:
            if file_data.get("id") == file_id:
                return file_data
        return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_path):
    """Extract text content from various file formats"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        elif ext in ['.doc', '.docx']:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            # For text files and code files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return ""


def download_file_from_url(file_url, destination_path):
    """Download file from URL to specified destination"""
    try:
        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logging.error(f"Error downloading file from {file_url}: {e}")
        return False


def generate_mock_ai_score():
    """Generate a mock AI detection score"""
    # Generate a random AI percentage between 0-100
    ai_percentage = round(random.uniform(0.0, 100.0), 2)
    
    # Generate interpretation based on percentage
    if ai_percentage >= 80:
        interpretation = "Very High AI probability"
    elif ai_percentage >= 60:
        interpretation = "High AI probability"
    elif ai_percentage >= 40:
        interpretation = "Medium AI probability"
    elif ai_percentage >= 20:
        interpretation = "Low AI probability"
    else:
        interpretation = "Very Low AI probability"
    
    # Generate other mock values
    chunks_analyzed = random.randint(10, 100)
    confidence = round(random.uniform(0.5, 1.0), 2)
    
    return {
        "ai_percentage": ai_percentage,
        "interpretation": interpretation,
        "chunks_analyzed": chunks_analyzed,
        "confidence": confidence
    }


def generate_html_report(student_name, filename, similarity_analysis, ai_analysis, detailed_results):
    """Generate HTML report for plagiarism analysis"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_similarity_class(similarity):
        if similarity >= 30:
            return "high-similarity"
        elif similarity >= 10:
            return "medium-similarity"
        else:
            return "low-similarity"
    
    html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Plagiarism Analysis Report - {filename}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .file-info {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; 
                     margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .similarity-score {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
        .ai-score {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .high-similarity {{ color: #e74c3c; }}
        .medium-similarity {{ color: #f39c12; }}
        .low-similarity {{ color: #27ae60; }}
        .analysis-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .analysis-card {{ background: white; padding: 25px; border-radius: 10px; 
                         box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #3498db; }}
        .metric {{ text-align: center; margin: 15px 0; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; font-weight: 500; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .detailed-matches {{ margin-top: 30px; }}
        .match-item {{ background: #fff; border: 1px solid #e1e8ed; border-radius: 8px; 
                      margin-bottom: 15px; padding: 20px; }}
        .match-header {{ font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .match-text {{ background: #f8f9fa; padding: 15px; border-radius: 5px; 
                      font-family: 'Courier New', monospace; border-left: 4px solid #3498db; }}
        .footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; 
                  border-top: 1px solid #e1e8ed; padding-top: 20px; }}
        @media (max-width: 768px) {{
            .analysis-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analysis Report</h1>
        <p style="margin: 0; font-size: 16px; opacity: 0.9;">Comprehensive analysis of academic integrity</p>
    </div>
    
    <div class="file-info">
        <h2>Document Information</h2>
        <p><strong>Student:</strong> {student_name}</p>
        <p><strong>Filename:</strong> {filename}</p>
        <p><strong>Analysis Date:</strong> {timestamp}</p>
    </div>
    
    <div class="analysis-grid">
        <div class="analysis-card">
            <h3>Similarity Analysis</h3>
            <div class="metric">
                <div class="metric-label">Similarity Score</div>
                <div class="metric-value {get_similarity_class(similarity_analysis['similarity_score'])}">{similarity_analysis['similarity_score']}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Comparisons</div>
                <div class="metric-value">{similarity_analysis['total_comparisons']}</div>
            </div>
        </div>
        
        <div class="analysis-card">
            <h3>AI Content Detection</h3>
            <div class="metric">
                <div class="metric-label">AI Probability</div>
                <div class="metric-value {get_similarity_class(ai_analysis['ai_percentage'])}">{ai_analysis['ai_percentage']}%</div>
            </div>
            <p><strong>Interpretation:</strong> {ai_analysis['interpretation']}</p>
            <p><strong>Confidence:</strong> {ai_analysis['confidence']}</p>
        </div>
    </div>"""
    
    if detailed_results:
        html_report += """
    <div class="detailed-matches">
        <h2>Detailed Match Results</h2>"""
        
        for i, result in enumerate(detailed_results, 1):
            html_report += f"""
        <div class="match-item">
            <div class="match-header">
                Match #{i} - Similar to: {result['source']} 
                <span class="{get_similarity_class(result['similarity'])}">{result['similarity']}%</span>
            </div>
            <div class="match-text">{result['matched_text'][:200]}{'...' if len(result['matched_text']) > 200 else ''}</div>
        </div>"""
        
        html_report += """
    </div>"""
    else:
        html_report += """
    <div class="detailed-matches">
        <div class="analysis-card">
            <h3>Original Content</h3>
            <p>No significant similarities found with existing documents. This appears to be original work.</p>
        </div>
    </div>"""
    
    html_report += """
    <div class="footer">
        <p>Generated by Academic Integrity Analysis System</p>
        <p>This report is confidential and intended for educational purposes only.</p>
    </div>
</body>
</html>"""
    
    return html_report


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "plagiarism_detector",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/grade/analyze', methods=['POST'])
def analyze_document():
    """Main endpoint for document analysis"""
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0},
                "ai_analysis": {"ai_percentage": 0},
                "report_html": "<html><body><h1>Error: No data provided</h1></body></html>",
                "errors": ["No JSON data provided in request"]
            }), 400

        # Extract required fields
        student_id = data.get('student_id')
        student_name = data.get('student_name')
        file_url = data.get('file_url')
        assignment_id = data.get('assignment_id')
        classroom_name = data.get('classroom_name')

        # Validate required fields
        if not all([student_id, student_name, file_url, assignment_id]):
            return jsonify({
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0},
                "ai_analysis": {"ai_percentage": 0},
                "report_html": "<html><body><h1>Error: Missing required fields</h1></body></html>",
                "errors": ["Missing required fields: student_id, student_name, file_url, assignment_id"]
            }), 400

        # Extract filename from URL
        parsed_url = urlparse(file_url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            return jsonify({
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0},
                "ai_analysis": {"ai_percentage": 0},
                "report_html": "<html><body><h1>Error: Invalid file URL</h1></body></html>",
                "errors": ["Could not extract filename from URL"]
            }), 400

        # Check if file extension is allowed
        if not allowed_file(filename):
            return jsonify({
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0},
                "ai_analysis": {"ai_percentage": 0},
                "report_html": "<html><body><h1>Error: File type not supported</h1></body></html>",
                "errors": [f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"]
            }), 400

        # Create secure filename and download path
        secure_name = secure_filename(filename)
        file_path = os.path.join(HOMEWORK_FOLDER, secure_name)

        # Download file from URL
        if not download_file_from_url(file_url, file_path):
            return jsonify({
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0},
                "ai_analysis": {"ai_percentage": 0},
                "report_html": "<html><body><h1>Error: Could not download file</h1></body></html>",
                "errors": ["Failed to download file from provided URL"]
            }), 400

        # Extract text content from downloaded file
        content = extract_text_from_file(file_path)
        if not content.strip():
            return jsonify({
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0},
                "ai_analysis": {"ai_percentage": 0},
                "report_html": "<html><body><h1>Error: No content extracted</h1></body></html>",
                "errors": ["Could not extract text content from file"]
            }), 400

        # Initialize database
        db = PlagiarismDatabase()

        # Create Text object for the new file
        new_file = Text(content, filename, file_path)

        # Perform plagiarism analysis
        similarity_results = []
        total_comparisons = 0
        max_similarity = 0.0

        existing_files = db.get_all_files()
        total_comparisons = len(existing_files)

        for existing_file_data in existing_files:
            # Skip if comparing with the same file
            if existing_file_data.get('student_id') == student_id and existing_file_data.get('assignment_id') == assignment_id:
                continue

            existing_file_path = existing_file_data.get('file_path')
            if existing_file_path and os.path.exists(existing_file_path):
                existing_content = extract_text_from_file(existing_file_path)
                if existing_content.strip():
                    existing_file_obj = Text(existing_content, existing_file_data.get('filename', ''), existing_file_path)

                    # Compare files using Matcher
                    matcher = Matcher(new_file, existing_file_obj, threshold=3, cutoff=5, ngramSize=3, minDistance=8, silent=True)
                    num_matches, _, _, matches_info, match_texts, similarity = matcher.match()

                    if similarity > 0:
                        max_similarity = max(max_similarity, similarity)
                        
                        for match in matches_info:
                            similarity_results.append({
                                "source": existing_file_data.get('filename', 'Unknown'),
                                "similarity": similarity,
                                "matched_text": match.get('matched_text', '')[:200]  # Limit text length
                            })

        # Generate mock AI analysis
        ai_analysis = generate_mock_ai_score()

        # Prepare similarity analysis
        similarity_analysis = {
            "similarity_score": max_similarity,
            "total_comparisons": total_comparisons,
            "detailed_results": similarity_results
        }

        # Generate HTML report
        report_html = generate_html_report(
            student_name, filename, similarity_analysis, ai_analysis, similarity_results
        )

        # Save file data to database
        file_data = {
            "id": f"{student_id}_{assignment_id}_{int(time.time())}",
            "student_id": student_id,
            "student_name": student_name,
            "assignment_id": assignment_id,
            "classroom_name": classroom_name,
            "filename": filename,
            "file_path": file_path,
            "content_checksum": new_file.checksum,
            "upload_timestamp": datetime.now().isoformat(),
            "similarity_score": max_similarity,
            "ai_score": ai_analysis["ai_percentage"]
        }
        
        db.add_file(file_data)

        # Return response
        return jsonify({
            "status": "completed",
            "similarity_analysis": similarity_analysis,
            "ai_analysis": ai_analysis,
            "report_html": report_html,
            "errors": []
        })

    except Exception as e:
        logging.error(f"Error in analyze_document: {str(e)}")
        return jsonify({
            "status": "failed",
            "similarity_analysis": {"similarity_score": 0},
            "ai_analysis": {"ai_percentage": 0},
            "report_html": f"<html><body><h1>Error: {str(e)}</h1></body></html>",
            "errors": [str(e)]
        }), 500


@app.route('/uploads/homeworks/<filename>')
def serve_homework_file(filename):
    """Serve homework files statically"""
    return send_from_directory(HOMEWORK_FOLDER, filename)


@app.route('/database/files', methods=['GET'])
def get_all_files():
    """Get all files in the database (for debugging/admin purposes)"""
    db = PlagiarismDatabase()
    return jsonify(db.get_all_files())


@app.route('/database/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    db = PlagiarismDatabase()
    files = db.get_all_files()
    
    total_files = len(files)
    unique_students = len(set(f.get('student_id') for f in files if f.get('student_id')))
    unique_assignments = len(set(f.get('assignment_id') for f in files if f.get('assignment_id')))
    
    avg_similarity = 0
    avg_ai_score = 0
    if files:
        avg_similarity = sum(f.get('similarity_score', 0) for f in files) / total_files
        avg_ai_score = sum(f.get('ai_score', 0) for f in files) / total_files
    
    return jsonify({
        "total_files": total_files,
        "unique_students": unique_students,
        "unique_assignments": unique_assignments,
        "average_similarity_score": round(avg_similarity, 2),
        "average_ai_score": round(avg_ai_score, 2)
    })


@app.route('/database/clear', methods=['DELETE'])
def clear_database():
    """Clear all files from database (for testing purposes)"""
    db = PlagiarismDatabase()
    db.data = {"files": []}
    db.save_database()
    
    # Also remove all files from homework folder
    try:
        for filename in os.listdir(HOMEWORK_FOLDER):
            file_path = os.path.join(HOMEWORK_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        logging.error(f"Error clearing homework folder: {e}")
    
    return jsonify({
        "message": "Database and files cleared successfully",
        "status": "success"
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "status": "failed",
        "similarity_analysis": {"similarity_score": 0},
        "ai_analysis": {"ai_percentage": 0},
        "report_html": "<html><body><h1>Error: File too large</h1></body></html>",
        "errors": ["File size exceeds maximum allowed size"]
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "failed",
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /grade/analyze",
            "GET /uploads/homeworks/<filename>",
            "GET /database/files",
            "GET /database/stats",
            "DELETE /database/clear"
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        "status": "failed",
        "similarity_analysis": {"similarity_score": 0},
        "ai_analysis": {"ai_percentage": 0},
        "report_html": "<html><body><h1>Internal Server Error</h1></body></html>",
        "errors": ["Internal server error occurred"]
    }), 500


if __name__ == '__main__':
    # Create required directories
    os.makedirs(HOMEWORK_FOLDER, exist_ok=True)
    
    # Initialize empty database if it doesn't exist
    if not os.path.exists(DATABASE_FILE):
        db = PlagiarismDatabase()
        db.save_database()
    
    print("🚀 Starting Plagiarism Detection API...")
    print(f"📁 Homework folder: {os.path.abspath(HOMEWORK_FOLDER)}")
    print(f"🗄️  Database file: {os.path.abspath(DATABASE_FILE)}")
    print(f"📊 Supported file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📏 Max file size: {MAX_CONTENT_LENGTH // (1024*1024)}MB")
    print("🌐 API Endpoints:")
    print("   GET  /health - Health check")
    print("   POST /grade/analyze - Analyze document for plagiarism")
    print("   GET  /uploads/homeworks/<filename> - Serve homework files")
    print("   GET  /database/files - List all files in database")
    print("   GET  /database/stats - Database statistics")
    print("   DELETE /database/clear - Clear database (testing)")
    print("\n" + "="*50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )