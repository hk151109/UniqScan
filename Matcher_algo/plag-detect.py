import re
import os
import json
import logging
import nltk
import time
from difflib import SequenceMatcher
from nltk.metrics.distance import edit_distance as editDistance
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import ngrams
from termcolor import colored
from datetime import datetime
import argparse
import hashlib
import shutil
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.lib.units import inch
from reportlab.platypus.flowables import HRFlowable
import textwrap
from io import StringIO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Text:
    def __init__(self, raw_text, label, filepath, removeStopwords=True):
        if isinstance(raw_text, list):
            # JSTOR critical works come in lists, where each item represents a page.
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
        tokenizer = nltk.RegexpTokenizer('[a-zA-Z]\\w+\'?\\w*')  # A custom regex tokenizer.
        spans = list(tokenizer.span_tokenize(self.text))
        # Take note of how many spans there are in the text
        if spans:
            self.length = spans[-1][-1]
        else:
            self.length = 0
        tokens = tokenizer.tokenize(self.text)
        tokens = [token.lower() for token in tokens]  # make them lowercase
        stemmer = LancasterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        if not removeStopwords:
            self.spans = spans
            return tokens
        tokenSpans = list(zip(tokens, spans))  # zip it up
        stopwords = nltk.corpus.stopwords.words('english')  # get stopwords
        tokenSpans = [token for token in tokenSpans if token[0] not in stopwords]  # remove stopwords from zip
        self.spans = [x[1] for x in tokenSpans]  # unzip; get spans
        return [x[0] for x in tokenSpans]  # unzip; get tokens

    def ngrams(self, n):
        """ Returns ngrams for the text."""
        return list(ngrams(self.tokens, n))


class ExtendedMatch:
    """
    Data structure container for a fancy version of a difflib-style
    Match object. The difflib Match class won't work for extended
    matches, since it only has the properties `a` (start location in
    text A), `b` (start location in text B), and size. Since our fancy
    new matches have different sizes in our different texts, we'll need
    two size attributes.
    """

    def __init__(self, a, b, sizeA, sizeB):
        self.a = a
        self.b = b
        self.sizeA = sizeA
        self.sizeB = sizeB
        # Whether this is actually two matches that have been fused into one.
        self.healed = False
        # Whether this match has been extended from its original boundaries.
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
    """
    Does the text matching.
    """

    def __init__(self, textObjA, textObjB, threshold=3, cutoff=5, ngramSize=3, removeStopwords=True, minDistance=8, silent=False):
        """
        Takes as input two Text() objects, and matches between them.
        """
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

        # Extend matches
        self.extended_matches = self.extend_matches()

        # Prune matches
        self.extended_matches = [match for match in self.extended_matches
                                if min(match.sizeA, match.sizeB) >= cutoff]

        self.numMatches = len(self.extended_matches)
        
        # Calculate similarity percentages
        self.similarity_score = self.calculate_similarity()

    def calculate_similarity(self):
        """Calculate the similarity percentage between the two texts."""
        if not self.extended_matches:
            return 0.0
            
        # Calculate total matched tokens in text A
        total_matched_tokens_A = sum(match.sizeA for match in self.extended_matches)
        
        # Calculate total tokens in text A
        total_tokens_A = len(self.textA.tokens)
        
        if total_tokens_A == 0:
            return 0.0
            
        # Calculate similarity percentage
        similarity = (total_matched_tokens_A / total_tokens_A) * 100
        
        return round(similarity, 2)

    def get_initial_matches(self):
        """
        This does the main work of finding matching n-gram sequences between
        the texts.
        """
        sequence = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        matchingBlocks = sequence.get_matching_blocks()

        # Only return the matching sequences that are higher than the
        # threshold given by the user.
        highMatchingBlocks = [match for match in matchingBlocks if match.size > self.threshold]

        numBlocks = len(highMatchingBlocks)

        if numBlocks > 0 and self.silent is not True:
            logging.info('%s total matches found.', numBlocks)

        return highMatchingBlocks

    def getContext(self, text, start, length, context):
        match = self.getTokensText(text, start, length)
        before = self.getTokensText(text, start - context, context)
        after = self.getTokensText(text, start + length, context)
        match = colored(match, 'red')
        out = " ".join([before, match, after])
        out = out.replace('\n', ' ')  # Replace newlines with spaces.
        out = re.sub('\\s+', ' ', out)
        return out

    def getTokensText(self, text, start, length):
        """ Looks up the passage in the original text, using its spans. """
        if start < 0:
            start = 0
        
        matchTokens = text.tokens[start:start + length]
        
        if start >= len(text.spans) or start + length > len(text.spans):
            # Handle out-of-range indices
            return ""
            
        spans = text.spans[start:start + length]
        if len(spans) == 0:
            # Don't try to get text or context beyond the end of a text.
            passage = ""
        else:
            passage = text.text[spans[0][0]:spans[-1][-1]]
        return passage

    def getLocations(self, text, start, length, asPercentages=False):
        """ Gets the numeric locations of the match. """
        if start >= len(text.spans) or start + length > len(text.spans):
            # Handle out-of-range indices
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
        lengthA = match.sizeA + self.ngramSize - 1  # offset according to nGram size
        lengthB = match.sizeB + self.ngramSize - 1  # offset according to nGram size
        
        # Get the matched text without context for citation
        matched_text = self.getTokensText(textA, match.a, lengthA)
        
        wordsA = self.getContext(textA, match.a, lengthA, context)
        wordsB = self.getContext(textB, match.b, lengthB, context)
        spansA = self.getLocations(textA, match.a, lengthA)
        spansB = self.getLocations(textB, match.b, lengthB)
        
        if spansA is not None and spansB is not None:
            self.locationsA.append(spansA)
            self.locationsB.append(spansB)
            
            # Store matched text with citation
            self.match_texts.append({
                "text": matched_text,
                "citation": citation_number,
                "spans": spansA,
                "source_file": textB.label
            })
            
            line1 = ('%s: %s %s [%d]' % (colored(textA.label, 'green'), spansA, wordsA, citation_number))
            line2 = ('%s: %s %s' % (colored(textB.label, 'green'), spansB, wordsB))
            out = line1 + '\n' + line2
            return out
        return None

    def heal_neighboring_matches(self):
        healedMatches = []
        ignoreNext = False
        matches = self.initial_matches.copy()
        # Handle only one match.
        if len(matches) == 1:
            match = matches[0]
            sizeA, sizeB = match.size, match.size
            match = ExtendedMatch(match.a, match.b, sizeA, sizeB)
            healedMatches.append(match)
            return healedMatches
        # For multiple match
        for i, match in enumerate(matches):
            # If last match
            if i + 1 > len(matches) - 1:
                break
            nextMatch = matches[i + 1]
            # If math already treated
            if ignoreNext:
                ignoreNext = False
                continue
            else:
                # Look at the number of different character between two raw match
                if (nextMatch.a - (match.a + match.size)) < self.minDistance:
                    # logging.debug('Potential healing candidate found: ' % (match, nextMatch))
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
        """ Computes the number of edits required to transform one
        (stemmed already, probably) word into another word, and
        adjusts for the average number of letters in each.
        Examples:
        color, colour: 0.1818181818
        theater, theatre: 0.2857
        day, today: 0.5
        foobar, foo56bar: 0.2857
        """
        distance = editDistance(wordA, wordB)
        averageLength = (len(wordA) + len(wordB)) / 2
        return distance / averageLength

    def extend_matches(self, cutoff=0.4):
        extended = False
        for match in self.healed_matches:
            # Check if indices are valid before looking one word before
            if match.a > 0 and match.b > 0 and len(self.textAgrams) > match.a - 1 and len(self.textBgrams) > match.b - 1:
                # Look one word before
                wordA = self.textAgrams[(match.a - 1)][0]
                wordB = self.textBgrams[(match.b - 1)][0]
                if self.edit_ratio(wordA, wordB) < cutoff:
                    if self.silent is not True:
                        logging.debug('Extending match backwards with words: %s %s', wordA, wordB)
                    match.a -= 1
                    match.b -= 1
                    match.sizeA += 1
                    match.sizeB += 1
                    match.extendedBackwards += 1
                    extended = True
                    
            # Look one word after
            idxA = match.a + match.sizeA + 1
            idxB = match.b + match.sizeB + 1
            if idxA >= len(self.textAgrams) or idxB >= len(self.textBgrams):
                # We've gone too far, and we're actually at the end of the text
                continue
                
            wordA = self.textAgrams[idxA][-1] if idxA < len(self.textAgrams) else ""
            wordB = self.textBgrams[idxB][-1] if idxB < len(self.textBgrams) else ""
            
            if wordA and wordB and self.edit_ratio(wordA, wordB) < cutoff:
                if self.silent is not True:
                    logging.debug('Extending match forwards with words: %s %s', wordA, wordB)
                match.sizeA += 1
                match.sizeB += 1
                match.extendedForwards += 1
                extended = True

        if extended:
            # If we've gone through the whole list and there's nothing
            # left to extend, then stop. Otherwise do this again.
            self.extend_matches()

        return self.healed_matches

    def match(self):
        """ Gets all matches and returns match data. """
        matches_info = []

        for num, match in enumerate(self.extended_matches):
            out = self.getMatch(match, num + 1)
            if out and self.silent is not True:
                logging.info('\nMatch %s:\n%s', num + 1, out)
            
            if out:
                matches_info.append({
                    "match_number": num + 1,
                    "match_text": out
                })

        return self.numMatches, self.locationsA, self.locationsB, matches_info, self.match_texts, self.similarity_score


class PlagiarismDetector:
    

    def __init__(self, folder_path, output_dir, threshold=3, cutoff=5, ngramSize=3, minDistance=8, silent=False):
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.cutoff = cutoff
        self.ngramSize = ngramSize
        self.minDistance = minDistance
        self.silent = silent
        self.known_files = {}  # Dictionary to store Text objects for existing files
        self.file_checksums = {}  # Store file checksums to detect changes
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(output_dir, 'plagiarism_detector.log'),
            filemode='a'
        )
        
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize by scanning existing files
        self.scan_existing_files()
        
    def scan_existing_files(self):
        """Scan existing files in the folder and add them to known_files."""
        logging.info("Scanning existing files in %s", self.folder_path)
        
        for filename in os.listdir(self.folder_path):
            filepath = os.path.join(self.folder_path, filename)
            
            # Skip directories and non-text files
            if os.path.isdir(filepath) or not self._is_text_file(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Create Text object for the file
                text_obj = Text(content, filename, filepath)
                self.known_files[filepath] = text_obj
                self.file_checksums[filepath] = text_obj.checksum
                
                logging.info(f"Added existing file: {filename}")
            except Exception as e:
                logging.error(f"Error reading file {filename}: {e}")
    
    def _is_text_file(self, filepath):
        """Check if the file is likely a text file."""
        text_extensions = ['.txt', '.md', '.py', '.java', '.c', '.cpp', '.h', '.js', '.html', '.css', '.json', '.xml']
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() in text_extensions:
            return True
            
        # Try to open and read a few bytes to check if it's text
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                f.read(1024)
                return True
        except:
            return False
    
    def process_new_file(self, filepath):
        """Process a new file to check for plagiarism against known files."""
        filename = os.path.basename(filepath)
        
        try:
            # Read the new file
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Create Text object for the new file
            new_file = Text(content, filename, filepath)
            
            # If this is the first file, mark it as original
            if not self.known_files:
                self.known_files[filepath] = new_file
                self.file_checksums[filepath] = new_file.checksum
                self.generate_report(filepath, [], 0, [], [])
                logging.info(f"File {filename} is original (first file in directory)")
                return
            
            # If the file is already known and hasn't changed, skip it
            if filepath in self.file_checksums and self.file_checksums[filepath] == new_file.checksum:
                logging.info(f"File {filename} unchanged, skipping")
                return
                
            # Compare with all known files
            results = []
            total_similarity = 0.0
            citations = []
            match_texts = []
            citation_number = 1
            
            for known_path, known_file in self.known_files.items():
                # Skip comparing the file with itself
                if filepath == known_path:
                    continue
                    
                # Compare the new file with the known file
                matcher = Matcher(
                    new_file, known_file,
                    threshold=self.threshold,
                    cutoff=self.cutoff,
                    ngramSize=self.ngramSize,
                    minDistance=self.minDistance,
                    silent=self.silent
                )
                
                num_matches, _, _, matches_info, file_match_texts, similarity = matcher.match()
                
                if num_matches > 0:
                    known_filename = os.path.basename(known_path)
                    
                    # Add citation
                    citations.append({
                        "number": citation_number,
                        "file": known_filename,
                        "similarity": similarity
                    })
                    
                    # Update match texts with the citation number
                    for match in file_match_texts:
                        match["citation"] = citation_number
                    
                    match_texts.extend(file_match_texts)
                    
                    results.append({
                        "comparison_file": known_filename,
                        "num_matches": num_matches,
                        "similarity": similarity,
                        "matches": matches_info,
                        "citation_number": citation_number
                    })
                    
                    # For overall similarity, take the highest similarity score
                    total_similarity = max(total_similarity, similarity)
                    
                    citation_number += 1
            
            # Add the new file to known files
            self.known_files[filepath] = new_file
            self.file_checksums[filepath] = new_file.checksum
            
            # Generate report
            self.generate_report(filepath, results, total_similarity, citations, match_texts)
            
            logging.info(f"Processed file {filename} - Similarity: {total_similarity}%")
            
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

    
    def generate_report(self, filepath, results, total_similarity, citations, match_texts):
        """Generate a detailed plagiarism report for the file."""
        filename = os.path.basename(filepath)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create report directory if it doesn't exist
        report_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create report filename based on the original filename
        report_filename = f"{os.path.splitext(filename)[0]}_report.html"
        report_path = os.path.join(report_dir, report_filename)
        
        # Read the original file content
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Apply highlighting to plagiarized sections
        highlighted_content = self._highlight_plagiarized_text(content, match_texts)
        
        # Generate HTML report
        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Plagiarism Report - {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .file-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .similarity-score {{ font-size: 24px; font-weight: bold; }}
        .high-similarity {{ color: #e74c3c; }}
        .medium-similarity {{ color: #f39c12; }}
        .low-similarity {{ color: #27ae60; }}
        .highlighted {{ background-color: #ffecb3; padding: 2px; border-radius: 3px; position: relative; }}
        .citation {{ font-size: 10px; position: relative; top: -5px; color: #e74c3c; font-weight: bold; }}
        .citation-list {{ margin-top: 30px; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Plagiarism Report</h1>
    
    <div class="file-info">
        <h2>File Information</h2>
        <p><strong>Filename:</strong> {filename}</p>
        <p><strong>Analysis Date:</strong> {timestamp}</p>
        <p><strong>Overall Similarity Score:</strong> 
            <span class="similarity-score {self._get_similarity_class(total_similarity)}">{total_similarity}%</span>
        </p>
    </div>
    
    <h2>File Content with Highlighted Similarities</h2>
    <pre>{highlighted_content}</pre>
    
    <div class="citation-list">
        <h2>Citations</h2>
"""
        
        if citations:
            html_report += """
        <table>
            <tr>
                <th>#</th>
                <th>Source File</th>
                <th>Similarity</th>
            </tr>
"""
            
            for citation in citations:
                html_report += f"""
            <tr>
                <td>{citation['number']}</td>
                <td>{citation['file']}</td>
                <td class="{self._get_similarity_class(citation['similarity'])}">{citation['similarity']}%</td>
            </tr>"""
            
            html_report += """
        </table>
"""
        else:
            html_report += """
        <p>No similarities found. This content appears to be original.</p>
"""
        
        html_report += """
    </div>
    
    <div>
        <h2>Detailed Match Information</h2>
"""
        
        if results:
            for result in results:
                html_report += f"""
        <h3>Comparison with {result['comparison_file']} (Citation #{result['citation_number']})</h3>
        <p>Similarity Score: <span class="{self._get_similarity_class(result['similarity'])}">{result['similarity']}%</span></p>
        <p>Number of Matches: {result['num_matches']}</p>
"""
        else:
            html_report += """
        <p>No detailed match information available.</p>
"""
        
        html_report += """
    </div>
    
    <footer style="margin-top: 50px; color: #7f8c8d; font-size: 12px; text-align: center;">
        <p>Generated by Plagiarism Detector</p>
    </footer>
</body>
</html>
"""
        
        # Write the report to a file
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(html_report)
            
        logging.info(f"Report generated: {report_path}")
        
        # Create a plain text version of the report
        text_report_path = os.path.join(report_dir, f"{os.path.splitext(filename)[0]}_report.txt")
        self._generate_text_report(text_report_path, filename, timestamp, total_similarity, citations, results, match_texts)
    
    def _highlight_plagiarized_text(self, content, match_texts):
        """Highlight plagiarized sections of text with citation numbers."""
        if not match_texts:
            return content
            
        # Sort match_texts by span start position in descending order
        # (Process from end to beginning to avoid index shifting issues)
        sorted_matches = sorted(match_texts, key=lambda x: x["spans"][0], reverse=True)
        
        # Apply highlighting and citations
        for match in sorted_matches:
            start, end = match["spans"]
            citation = match["citation"]
            
            # Insert citation and highlighting
            content = (
                content[:end] + 
                "</span><sup class='citation'>[" + str(citation) + "]</sup>" + 
                content[end:]
            )
            content = (
                content[:start] + 
                "<span class='highlighted'>" + 
                content[start:]
            )
            
        return content
    
    def _generate_text_report(self, report_path, filename, timestamp, total_similarity, citations, results, match_texts):
        """Generate a plain text version of the plagiarism report."""
        with open(report_path, 'w', encoding='utf-8') as file:
            file.write(f"PLAGIARISM REPORT\n")
            file.write(f"=================\n\n")
            
            file.write(f"File: {filename}\n")
            file.write(f"Analysis Date: {timestamp}\n")
            file.write(f"Overall Similarity Score: {total_similarity}%\n\n")
            
            if citations:
                file.write(f"CITATIONS\n")
                file.write(f"---------\n")
                for citation in citations:
                    file.write(f"[{citation['number']}] {citation['file']} - {citation['similarity']}% similarity\n")
                file.write("\n")
            
            if results:
                file.write(f"DETAILED MATCH INFORMATION\n")
                file.write(f"-------------------------\n\n")
                for result in results:
                    file.write(f"Comparison with {result['comparison_file']} (Citation #{result['citation_number']})\n")
                    file.write(f"Similarity Score: {result['similarity']}%\n")
                    file.write(f"Number of Matches: {result['num_matches']}\n\n")
                    
                    for idx, match_info in enumerate(result['matches']):
                        file.write(f"Match {idx+1}:\n")
                        file.write(f"{match_info['match_text']}\n\n")
            else:
                file.write("No similarities found. This content appears to be original.\n")
    
    def _get_similarity_class(self, similarity):
        """Return CSS class based on similarity percentage."""
        if similarity >= 30:
            return "high-similarity"
        elif similarity >= 10:
            return "medium-similarity"
        else:
            return "low-similarity"
    
    def monitor_folder(self):
        """Monitor the folder for new files and check them for plagiarism."""
        event_handler = FileChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.folder_path, recursive=False)
        observer.start()
        
        try:
            logging.info(f"Monitoring folder: {self.folder_path}")
            print(f"Monitoring folder: {self.folder_path}")
            print(f"Reports will be saved to: {self.output_dir}")
            print("Press Ctrl+C to stop monitoring")
            
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events."""
    
    def __init__(self, detector):
        self.detector = detector
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        logging.info(f"New file detected: {event.src_path}")
        self.detector.process_new_file(event.src_path)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        logging.info(f"File modified: {event.src_path}")
        self.detector.process_new_file(event.src_path)


def create_html_report(report_path, filename, timestamp, total_similarity, content, match_texts, citations, results):
    """Create a standalone HTML report with embedded styled content."""
    # Create HTML report with content, styling and citations
    # Implementation moved to the generate_report method in PlagiarismDetector class
    pass


def main():
    parser = argparse.ArgumentParser(description='Plagiarism detector for text files')
    parser.add_argument('folder', help='Folder to monitor for text files')
    parser.add_argument('--output', '-o', default='plagiarism_reports', help='Output directory for reports')
    parser.add_argument('--threshold', '-t', type=int, default=3, help='Threshold for initial matching')
    parser.add_argument('--cutoff', '-c', type=int, default=5, help='Cutoff for match size')
    parser.add_argument('--ngram', '-n', type=int, default=3, help='N-gram size')
    parser.add_argument('--distance', '-d', type=int, default=8, help='Min distance for healing matches')
    parser.add_argument('--silent', '-s', action='store_true', help='Suppress detailed output')
    parser.add_argument('--scan-only', action='store_true', help='Only scan existing files, don\'t monitor')
    
    args = parser.parse_args()
    
    # Create the detector
    detector = PlagiarismDetector(
        args.folder, 
        args.output,
        threshold=args.threshold,
        cutoff=args.cutoff,
        ngramSize=args.ngram,
        minDistance=args.distance,
        silent=args.silent
    )
    
    # Scan all existing files
    print(f"Scanning existing files in {args.folder}...")
    
    # Process each existing file to compare against others
    for filepath in list(detector.known_files.keys()):
        detector.process_new_file(filepath)
    
    # If scan-only mode is not enabled, start monitoring
    if not args.scan_only:
        detector.monitor_folder()
    else:
        print("Scan complete. Reports saved to", args.output)


if __name__ == "__main__":
    main()