from flask import Flask, request, jsonify
import os
import json
import logging
import nltk
import time
import random
import hashlib
import requests
import tempfile
from datetime import datetime
from difflib import SequenceMatcher
from nltk.metrics.distance import edit_distance as editDistance
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import ngrams
from pathlib import Path
import re
from urllib.parse import urlparse
import mimetypes
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global storage for user scores
user_scores_log = {}

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
        """Calculate a checksum for the file content to detect changes"""
        return hashlib.md5(self.text.encode()).hexdigest()

    def preprocess(self, text):
        """ Heals hyphenated words, and maybe other things. """
        self.text = re.sub(r'([A-Za-z])- ([a-z])', r'\1\2', text)

    def getTokens(self, removeStopwords=True):
        """ Tokenizes the text, breaking it up into words, removing punctuation. """
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
        """ Returns ngrams for the text."""
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
    def __init__(self, textObjA, textObjB, threshold=3, cutoff=5, ngramSize=3, removeStopwords=True, minDistance=8, silent=True):
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
        highMatchingBlocks = [match for match in matchingBlocks if match.size > self.threshold]
        return highMatchingBlocks

    def getTokensText(self, text, start, length):
        if start < 0:
            start = 0
        
        matchTokens = text.tokens[start:start + length]
        
        if start >= len(text.spans) or start + length > len(text.spans):
            return ""
            
        spans = text.spans[start:start + length]
        if len(spans) == 0:
            passage = ""
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
            lengthA = match.sizeA + self.ngramSize - 1
            matched_text = self.getTokensText(self.textA, match.a, lengthA)
            
            if matched_text:
                matches_info.append({
                    "source": self.textB.label,
                    "similarity": self.similarity_score,
                    "matched_text": matched_text[:200] + "..." if len(matched_text) > 200 else matched_text
                })

        return self.numMatches, matches_info, self.similarity_score


class AIDetector:
    """Mock AI detection class - generates random but realistic AI scores"""
    
    def __init__(self):
        self.patterns = [
            "repetitive sentence structures",
            "uniform vocabulary complexity",
            "lack of personal voice",
            "perfect grammar consistency",
            "unnatural transitions",
            "generic examples"
        ]
    
    def analyze_text(self, text):
        """Analyze text for AI-generated content"""
        # Mock analysis - in reality, this would use AI detection models
        word_count = len(text.split())
        
        # Generate realistic AI percentage based on text characteristics
        base_score = random.uniform(5.0, 35.0)
        
        # Adjust based on text length (longer texts tend to have lower AI scores)
        if word_count > 1000:
            base_score *= 0.8
        elif word_count < 200:
            base_score *= 1.2
            
        ai_percentage = min(95.0, max(0.0, base_score))
        
        # Generate interpretation
        if ai_percentage < 15:
            interpretation = "Low AI probability"
        elif ai_percentage < 35:
            interpretation = "Moderate AI probability"
        elif ai_percentage < 60:
            interpretation = "High AI probability"
        else:
            interpretation = "Very high AI probability"
            
        # Mock chunks analyzed
        chunks_analyzed = max(1, word_count // 50)
        
        # Mock confidence (higher confidence for more extreme scores)
        if ai_percentage < 20 or ai_percentage > 70:
            confidence = random.uniform(0.8, 0.95)
        else:
            confidence = random.uniform(0.6, 0.85)
            
        return {
            "ai_percentage": round(ai_percentage, 1),
            "interpretation": interpretation,
            "chunks_analyzed": chunks_analyzed,
            "confidence": round(confidence, 2)
        }


class PlagiarismAPI:
    def __init__(self):
        self.ai_detector = AIDetector()
        self.ensure_nltk_data()
        
    def ensure_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def download_file(self, url):
        """Download file from URL"""
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            filename = os.path.basename(urlparse(url).path) or "document.txt"
            temp_path = os.path.join(temp_dir, secure_filename(filename))
            
            # Handle different content types
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type.lower():
                # For PDF files, you'd need to extract text using PyPDF2 or similar
                # For now, we'll treat as text
                content = response.text
            else:
                content = response.text
            
            with open(temp_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
                
            return temp_path, content
            
        except Exception as e:
            logging.error(f"Error downloading file from {url}: {e}")
            raise
    
    def analyze_similarity(self, text_content, filename):
        """Analyze text for similarity against a mock database"""
        # Create sample comparison texts (in reality, this would be your database)
        sample_texts = [
            ("Sample Academic Paper 1.pdf", "This is a sample academic paper discussing various topics in computer science and technology. The methodology involves comprehensive analysis of existing literature and implementation of novel algorithms."),
            ("Research Document 2.txt", "Modern educational systems require innovative approaches to learning and assessment. Digital transformation has fundamentally changed how we approach knowledge sharing."),
            ("Thesis Chapter 3.docx", "The implementation of machine learning algorithms in educational technology presents unique challenges and opportunities for enhancing student learning outcomes."),
        ]
        
        main_text = Text(text_content, filename, "")
        detailed_results = []
        max_similarity = 0.0
        total_comparisons = 0
        
        for source_name, source_content in sample_texts:
            try:
                source_text = Text(source_content, source_name, "")
                matcher = Matcher(main_text, source_text, silent=True)
                num_matches, matches_info, similarity = matcher.match()
                
                if similarity > 0:
                    detailed_results.extend(matches_info)
                    max_similarity = max(max_similarity, similarity)
                    
                total_comparisons += 1
                
            except Exception as e:
                logging.error(f"Error comparing with {source_name}: {e}")
                continue
        
        return {
            "similarity_score": round(max_similarity, 1),
            "total_comparisons": total_comparisons,
            "detailed_results": detailed_results[:10]  # Limit to top 10 results
        }
    
    def calculate_plagiarism_score(self, similarity_score, ai_percentage):
        """Calculate overall plagiarism score"""
        # Weight the scores (you can adjust these weights)
        similarity_weight = 0.6
        ai_weight = 0.4
        
        plagiarism_score = (similarity_score * similarity_weight) + (ai_percentage * ai_weight)
        
        # Determine risk level
        if plagiarism_score < 15:
            risk_level = "low"
        elif plagiarism_score < 35:
            risk_level = "moderate"
        elif plagiarism_score < 60:
            risk_level = "high"
        else:
            risk_level = "critical"
            
        # Determine contributing factors
        contributing_factors = []
        if similarity_score > 25:
            contributing_factors.append("high_similarity_matches")
        if ai_percentage > 25:
            contributing_factors.append("ai_generated_content")
        if similarity_score > 15 and ai_percentage > 15:
            contributing_factors.append("multiple_detection_methods")
        if not contributing_factors:
            contributing_factors.append("low_risk_indicators")
            
        return {
            "plagiarism_score": round(plagiarism_score, 1),
            "risk_level": risk_level,
            "contributing_factors": contributing_factors
        }
    
    def generate_html_report(self, student_data, similarity_analysis, ai_analysis, plagiarism_analysis, content):
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine color classes based on scores
        def get_score_class(score):
            if score < 15:
                return "low-risk"
            elif score < 35:
                return "medium-risk"
            elif score < 60:
                return "high-risk"
            else:
                return "critical-risk"
        
        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Analysis Report - {student_data['student_name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 300;
        }}
        .content {{
            padding: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .info-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .info-card p {{
            margin: 0;
            font-size: 16px;
            font-weight: 500;
        }}
        .scores-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        .score-card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-top: 4px solid;
        }}
        .score-card.similarity {{
            border-top-color: #3498db;
        }}
        .score-card.ai-detection {{
            border-top-color: #e74c3c;
        }}
        .score-card.plagiarism {{
            border-top-color: #f39c12;
        }}
        .score-card h3 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 600;
        }}
        .score-display {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .low-risk {{ color: #27ae60; }}
        .medium-risk {{ color: #f39c12; }}
        .high-risk {{ color: #e74c3c; }}
        .critical-risk {{ color: #c0392b; }}
        .analysis-details {{
            margin-top: 30px;
        }}
        .details-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .details-card h4 {{
            margin: 0 0 15px 0;
            color: #2c3e50;
        }}
        .matches-list {{
            list-style: none;
            padding: 0;
        }}
        .matches-list li {{
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }}
        .risk-factors {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .risk-factor {{
            background: #e74c3c;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 12px;
        }}
        .interpretation {{
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Plagiarism Analysis Report</h1>
            <p>Comprehensive academic integrity assessment</p>
        </div>
        
        <div class="content">
            <div class="info-grid">
                <div class="info-card">
                    <h3>Student</h3>
                    <p>{student_data['student_name']}</p>
                </div>
                <div class="info-card">
                    <h3>Classroom</h3>
                    <p>{student_data['classroom_name']}</p>
                </div>
                <div class="info-card">
                    <h3>Assignment ID</h3>
                    <p>{student_data['assignment_id']}</p>
                </div>
                <div class="info-card">
                    <h3>Analysis Date</h3>
                    <p>{timestamp}</p>
                </div>
            </div>
            
            <div class="scores-section">
                <div class="score-card similarity">
                    <h3>Similarity Analysis</h3>
                    <div class="score-display {get_score_class(similarity_analysis['similarity_score'])}">
                        {similarity_analysis['similarity_score']}%
                    </div>
                    <p>Compared against {similarity_analysis['total_comparisons']} documents</p>
                </div>
                
                <div class="score-card ai-detection">
                    <h3>AI Detection</h3>
                    <div class="score-display {get_score_class(ai_analysis['ai_percentage'])}">
                        {ai_analysis['ai_percentage']}%
                    </div>
                    <p class="interpretation">{ai_analysis['interpretation']}</p>
                    <p>Confidence: {ai_analysis['confidence']}</p>
                </div>
                
                <div class="score-card plagiarism">
                    <h3>Overall Plagiarism Score</h3>
                    <div class="score-display {get_score_class(plagiarism_analysis['plagiarism_score'])}">
                        {plagiarism_analysis['plagiarism_score']}%
                    </div>
                    <p>Risk Level: <strong>{plagiarism_analysis['risk_level'].upper()}</strong></p>
                    <div class="risk-factors">
                        {' '.join([f'<span class="risk-factor">{factor.replace("_", " ").title()}</span>' for factor in plagiarism_analysis['contributing_factors']])}
                    </div>
                </div>
            </div>
            
            <div class="analysis-details">
                {f'''
                <div class="details-card">
                    <h4>Similarity Matches Found</h4>
                    <ul class="matches-list">
                        {"".join([f"<li><strong>{match['source']}</strong><br>Similarity: {match['similarity']}%<br><em>{match['matched_text']}</em></li>" for match in similarity_analysis['detailed_results'][:5]])}
                    </ul>
                </div>
                ''' if similarity_analysis['detailed_results'] else '<div class="details-card"><h4>No Significant Similarity Matches Found</h4><p>The document appears to have original content with no substantial matches in our database.</p></div>'}
                
                <div class="details-card">
                    <h4>AI Detection Analysis</h4>
                    <p>Analyzed {ai_analysis['chunks_analyzed']} text segments with {ai_analysis['confidence']} confidence level.</p>
                    <p><strong>Interpretation:</strong> {ai_analysis['interpretation']}</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Academic Integrity Analysis System | {timestamp}</p>
            <p>This report is confidential and intended solely for academic evaluation purposes.</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_report
    
    def log_user_score(self, student_id, scores):
        """Log user scores for tracking"""
        global user_scores_log
        
        if student_id not in user_scores_log:
            user_scores_log[student_id] = []
            
        user_scores_log[student_id].append({
            "timestamp": datetime.now().isoformat(),
            "scores": scores
        })
        
        # Keep only last 10 entries per user
        user_scores_log[student_id] = user_scores_log[student_id][-10:]
    
    def analyze_document(self, payload):
        """Main analysis function"""
        try:
            # Download and process file
            file_path, content = self.download_file(payload['file_url'])
            
            # Perform similarity analysis
            similarity_analysis = self.analyze_similarity(content, payload['student_name'])
            
            # Perform AI detection analysis
            ai_analysis = self.ai_detector.analyze_text(content)
            
            # Calculate overall plagiarism score
            plagiarism_analysis = self.calculate_plagiarism_score(
                similarity_analysis['similarity_score'],
                ai_analysis['ai_percentage']
            )
            
            # Generate HTML report
            html_report = self.generate_html_report(
                payload, similarity_analysis, ai_analysis, plagiarism_analysis, content
            )
            
            # Log scores
            scores = {
                "similarity_score": similarity_analysis['similarity_score'],
                "ai_percentage": ai_analysis['ai_percentage'],
                "plagiarism_score": plagiarism_analysis['plagiarism_score']
            }
            self.log_user_score(payload['student_id'], scores)
            
            # Clean up temporary file
            try:
                os.unlink(file_path)
                os.rmdir(os.path.dirname(file_path))
            except:
                pass
            
            return {
                "status": "completed",
                "similarity_analysis": similarity_analysis,
                "ai_analysis": ai_analysis,
                "plagiarism_analysis": plagiarism_analysis,
                "report_html": html_report,
                "errors": []
            }
            
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            return {
                "status": "failed",
                "similarity_analysis": {"similarity_score": 0, "total_comparisons": 0, "detailed_results": []},
                "ai_analysis": {"ai_percentage": 0, "interpretation": "Analysis failed", "chunks_analyzed": 0, "confidence": 0},
                "plagiarism_analysis": {"plagiarism_score": 0, "risk_level": "unknown", "contributing_factors": ["analysis_error"]},
                "report_html": f"<html><body><h1>Analysis Failed</h1><p>Error: {str(e)}</p></body></html>",
                "errors": [str(e)]
            }


# Initialize API instance
api = PlagiarismAPI()

@app.route('/grade/analyze', methods=['POST'])
def analyze_submission():
    """Main API endpoint for plagiarism analysis"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        payload = request.get_json()
        
        # Validate required fields
        required_fields = ['student_id', 'student_name', 'file_url', 'assignment_id', 'classroom_name']
        for field in required_fields:
            if field not in payload:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Perform analysis
        result = api.analyze_document(payload)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({
            "status": "failed",
            "similarity_analysis": {"similarity_score": 0, "total_comparisons": 0, "detailed_results": []},
            "ai_analysis": {"ai_percentage": 0, "interpretation": "Analysis failed", "chunks_analyzed": 0, "confidence": 0},
            "plagiarism_analysis": {"plagiarism_score": 0, "risk_level": "unknown", "contributing_factors": ["api_error"]},
            "report_html": f"<html><body><h1>Analysis Failed</h1><p>API Error: {str(e)}</p></body></html>",
            "errors": [str(e)]
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Plagiarism Detection API"
    })


@app.route('/user/<student_id>/scores', methods=['GET'])
def get_user_scores(student_id):
    """Get historical scores for a user"""
    global user_scores_log
    
    if student_id in user_scores_log:
        return jsonify({
            "student_id": student_id,
            "scores_history": user_scores_log[student_id]
        })
    else:
        return jsonify({
            "student_id": student_id,
            "scores_history": [],
            "message": "No scores found for this user"
        })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    global user_scores_log
    
    total_users = len(user_scores_log)
    total_analyses = sum(len(scores) for scores in user_scores_log.values())
    
    return jsonify({
        "total_users_analyzed": total_users,
        "total_analyses_performed": total_analyses,
        "system_status": "operational"
    })


if __name__ == '__main__':
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    print("Starting Plagiarism Detection API Service...")
    print("Server will run on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("POST /grade/analyze - Main plagiarism analysis endpoint")
    print("GET /health - Health check")
    print("GET /user/<student_id>/scores - Get user's score history")
    print("GET /stats - Get system statistics")
    
    app.run(host='0.0.0.0', port=5000, debug=False)