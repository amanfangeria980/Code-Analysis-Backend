import os
import re
from collections import defaultdict
from git import Repo
import ast
import radon.metrics
import radon.complexity
from typing import Dict, List, Any
from openai import AsyncOpenAI
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

class CodeAnalyzer:
    def __init__(self, repo_url: str, openai_api_key: str, model: str = "gpt-4"):
        self.repo_url = repo_url
        self.openai_api_key = openai_api_key
        self.repo_path = "temp_repo"
        self.model = model
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.supported_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.hpp', '.cc', '.h'],
            'ruby': ['.rb'],
            'php': ['.php'],
            'go': ['.go'],
            'rust': ['.rs'],
            'swift': ['.swift'],
            'kotlin': ['.kt'],
            'csharp': ['.cs']
        }
        self.metrics = {
            'total_files': 0,
            'total_lines': 0,
            'complexity_metrics': {},
            'commit_analysis': {},
            'developer_metrics': defaultdict(lambda: {
                'total_commits': 0,
                'total_lines_changed': 0,
                'files_modified': set()
            })
        }

    def _get_file_type(self, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        for lang, extensions in self.supported_extensions.items():
            if ext in extensions:
                return lang
        return 'unknown'

    async def initialize(self):
        # Clone repository
        if os.path.exists(self.repo_path):
            os.system(f"rm -rf {self.repo_path}")
        os.makedirs(self.repo_path)
        repo = Repo.clone_from(self.repo_url, self.repo_path)
        
        # Analyze repository
        await self._analyze_code_structure()
        await self._analyze_git_commits(repo)
        
        return self.metrics

    async def _analyze_code_structure(self):
        """Analyze code structure and complexity"""
        all_chunks = []
        
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                file_type = self._get_file_type(file)
                if file_type != 'unknown':
                    filepath = os.path.join(root, file)
                    
                    # Count total files and lines
                    self.metrics['total_files'] += 1
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.splitlines()
                            self.metrics['total_lines'] += len(lines)
                            
                            # Process code chunks
                            chunks = await self._process_code_chunks(filepath, content)
                            all_chunks.extend(chunks)
                            
                            # Existing complexity analysis code...
                            if file_type == 'python':
                                complexity = radon.complexity.cc_visit(content)
                                tree = ast.parse(content)
                                
                                file_metrics = {
                                    'cyclomatic_complexity': sum(block.complexity for block in complexity),
                                    'function_count': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                                    'class_count': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                                    'language': file_type,
                                    'chunks': len(chunks),
                                    'avg_chunk_size': round(len(content) / len(chunks) if chunks else 0, 2)
                                }
                            else:
                                file_metrics = {
                                    'cyclomatic_complexity': 0,
                                    'function_count': 0,
                                    'class_count': 0,
                                    'language': file_type,
                                    'chunks': len(chunks),
                                    'avg_chunk_size': round(len(content) / len(chunks) if chunks else 0, 2)
                                }
                            
                            self.metrics['complexity_metrics'][file] = file_metrics
                            
                    except Exception as e:
                        print(f"Warning: Could not analyze {file} completely. Error: {str(e)}")
        
        # Store the chunks in the metrics for later use
        self.metrics['code_chunks'] = all_chunks

    async def _analyze_git_commits(self, repo):
        """Analyze git commit history"""
        # Get all supported file extensions
        all_extensions = [ext for extensions in self.supported_extensions.values() for ext in extensions]
        
        for commit in repo.iter_commits():
            author = commit.author.name
            
            # Update developer metrics
            self.metrics['developer_metrics'][author]['total_commits'] += 1
            
            # Analyze files changed in this commit
            for diff in commit.diff(commit.parents[0] if commit.parents else None):
                if diff.a_path:
                    file_path = diff.a_path
                    # Check if file has a supported extension
                    if any(file_path.endswith(ext) for ext in all_extensions):
                        # Count lines changed
                        diff_content = diff.diff
                        if isinstance(diff_content, bytes):
                            diff_content = diff_content.decode('utf-8')
                        lines_changed = len(diff_content.splitlines())
                        
                        self.metrics['developer_metrics'][author]['total_lines_changed'] += lines_changed
                        self.metrics['developer_metrics'][author]['files_modified'].add(file_path)

        # Convert files_modified to list for JSON serialization
        for dev_metrics in self.metrics['developer_metrics'].values():
            dev_metrics['files_modified'] = list(dev_metrics['files_modified'])

    async def analyze_code(self, question: str):
        """
        Generate a comprehensive analysis report based on the metrics
        """
        # Filter out files with actual complexity and metrics
        complex_files = {
            file: metrics for file, metrics in self.metrics['complexity_metrics'].items()
            if metrics['cyclomatic_complexity'] > 0 or 
               metrics['function_count'] > 0 or 
               metrics['class_count'] > 0
        }
        
        # Calculate language distribution
        language_distribution = {}
        for metrics in self.metrics['complexity_metrics'].values():
            lang = metrics['language']
            language_distribution[lang] = language_distribution.get(lang, 0) + 1

        # Prepare the analysis response in a structured JSON format
        analysis_response = {
            "repository_analysis": {
                "code_metrics": {
                    "total_files": self.metrics['total_files'],
                    "total_lines": self.metrics['total_lines'],
                    "average_lines_per_file": round(self.metrics['total_lines'] / self.metrics['total_files'] if self.metrics['total_files'] else 0, 2),
                    "language_distribution": language_distribution
                },
                "complexity_analysis": {
                    "complex_files": [
                        {
                            "file": file,
                            "complexity": metrics['cyclomatic_complexity'],
                            "functions": metrics['function_count'],
                            "classes": metrics['class_count'],
                            "language": metrics['language']
                        }
                        for file, metrics in complex_files.items()
                    ],
                    "average_complexity": round(
                        sum(m['cyclomatic_complexity'] for m in complex_files.values()) / 
                        len(complex_files) if complex_files else 0, 
                        2
                    )
                },
                "developer_activity": {
                    "total_contributors": len(self.metrics['developer_metrics']),
                    "active_contributors": [
                        {
                            "name": author,
                            "commits": stats['total_commits'],
                            "lines_changed": stats['total_lines_changed'],
                            "files_modified": len(stats['files_modified']),
                            "avg_changes_per_commit": round(
                                stats['total_lines_changed'] / stats['total_commits'] 
                                if stats['total_commits'] else 0, 
                                2
                            )
                        }
                        for author, stats in self.metrics['developer_metrics'].items()
                        if stats['total_commits'] > 0
                    ]
                },
                "code_analysis": {
                    "chunk_metrics": {
                        "total_chunks": len(self.metrics['code_chunks']),
                        "avg_chunk_size": round(sum(len(chunk.page_content) for chunk in self.metrics['code_chunks']) / len(self.metrics['code_chunks']) if self.metrics['code_chunks'] else 0, 2),
                        "chunks_per_file": round(len(self.metrics['code_chunks']) / self.metrics['total_files'] if self.metrics['total_files'] else 0, 2)
                    }
                }
            },
            "analysis_summary": {
                "code_health": {
                    "complexity_rating": self._calculate_complexity_rating(complex_files),
                    "maintainability_score": self._calculate_maintainability_score(complex_files)
                },
                "key_findings": self._generate_key_findings(complex_files, language_distribution)
            }
        }

        try:
            # Get AI insights about the specific question
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a code analysis expert. Analyze the provided metrics and code chunks to provide detailed insights. 
    Consider: code structure, complexity patterns, development patterns, and potential improvements."""},
                    {"role": "user", "content": f"""Based on these metrics and code chunks:
    
    Metrics: {json.dumps(analysis_response, indent=2)}
    
    Number of Code Chunks: {len(self.metrics['code_chunks'])}
    Average Chunk Size: {round(sum(len(chunk.page_content) for chunk in self.metrics['code_chunks']) / len(self.metrics['code_chunks']) if self.metrics['code_chunks'] else 0, 2)} characters
    
    Question: {question}"""}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Add AI insights to the response
            analysis_response["ai_insights"] = {
                "question": question,
                "analysis": response.choices[0].message.content
            }
            
            return analysis_response
            
        except Exception as e:
            return {"error": f"Error generating analysis: {str(e)}"}

    def _calculate_complexity_rating(self, complex_files):
        if not complex_files:
            return "Simple"
        avg_complexity = sum(m['cyclomatic_complexity'] for m in complex_files.values()) / len(complex_files)
        if avg_complexity < 5:
            return "Simple"
        elif avg_complexity < 10:
            return "Moderate"
        elif avg_complexity < 20:
            return "Complex"
        return "Very Complex"

    def _calculate_maintainability_score(self, complex_files):
        if not complex_files:
            return 100
        
        # Calculate based on complexity, function count, and class count
        avg_complexity = sum(m['cyclomatic_complexity'] for m in complex_files.values()) / len(complex_files)
        avg_functions = sum(m['function_count'] for m in complex_files.values()) / len(complex_files)
        avg_classes = sum(m['class_count'] for m in complex_files.values()) / len(complex_files)
        
        # Score from 0-100, higher is better
        score = 100 - (avg_complexity * 2) - (avg_functions * 0.5) - (avg_classes * 1)
        return max(0, min(100, round(score, 2)))

    def _generate_key_findings(self, complex_files, language_distribution):
        findings = []
        
        # Analyze language distribution
        if language_distribution:
            primary_language = max(language_distribution.items(), key=lambda x: x[1])[0]
            findings.append(f"Primary language: {primary_language}")
        
        # Analyze complexity
        if complex_files:
            most_complex = max(complex_files.items(), key=lambda x: x[1]['cyclomatic_complexity'])
            findings.append(f"Most complex file: {most_complex[0]}")
            
            avg_functions = sum(m['function_count'] for m in complex_files.values()) / len(complex_files)
            findings.append(f"Average functions per file: {round(avg_functions, 2)}")
        
        return findings

    async def _process_code_chunks(self, file_path: str, content: str) -> List[Document]:
        """Process code into chunks for better analysis"""
        # Initialize text splitter with code-specific parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the content into chunks
        chunks = text_splitter.create_documents(
            texts=[content],
            metadatas=[{
                "file_path": file_path,
                "language": self._get_file_type(file_path),
                "source": "code"
            }]
        )
        
        return chunks