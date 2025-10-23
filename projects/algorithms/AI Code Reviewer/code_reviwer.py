#!/usr/bin/env python3
"""
AI Code Reviewer - A Python script for automated code analysis and improvement suggestions.

This script analyzes Python code for common issues, style violations, and potential
improvements, providing detailed feedback to help developers write better code.
"""

import ast
import sys
import pycodestyle
from typing import List, Set, Dict, Optional, Any
from dataclasses import dataclass
import re
from pathlib import Path

@dataclass
class CodeIssue:
    """Data class to store information about code issues."""
    line_number: int
    issue_type: str
    message: str
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'

class AICodeReviewer:
    """
    A comprehensive code review tool that analyzes Python code for various issues
    and provides improvement suggestions.
    """

    def __init__(self):
        """Initialize the AICodeReviewer with empty issue lists and configuration."""
        self.issues: List[CodeIssue] = []
        self.source_code: str = ""
        self.ast_tree: Optional[ast.AST] = None
        
        # Configure severity levels for different types of issues
        self.severity_levels: Dict[str, str] = {
            'syntax_error': 'HIGH',
            'undefined_variable': 'HIGH',
            'style_violation': 'MEDIUM',
            'missing_docstring': 'MEDIUM',
            'comment_issue': 'LOW',
            'complexity_issue': 'MEDIUM'
        }

    def load_file(self, file_path: str) -> bool:
        """
        Load Python code from a file.

        Args:
            file_path (str): Path to the Python file to analyze

        Returns:
            bool: True if file was successfully loaded, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.source_code = file.read()
            return True
        except Exception as e:
            self.issues.append(CodeIssue(
                0,
                'file_error',
                f"Error loading file: {str(e)}",
                'HIGH'
            ))
            return False

    def load_code(self, code: str) -> None:
        """
        Load Python code from a string.

        Args:
            code (str): Python code to analyze
        """
        self.source_code = code

    def analyze(self) -> None:
        """
        Perform comprehensive code analysis by running all available checks.
        """
        self.issues = []  # Reset issues list before new analysis
        
        # Parse AST
        try:
            self.ast_tree = ast.parse(self.source_code)
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                e.lineno or 0,
                'syntax_error',
                f"Syntax Error: {str(e)}",
                'HIGH'
            ))
            return

        # Run all analysis checks
        self._check_syntax()
        self._check_style()
        self._check_docstrings()
        self._check_complexity()
        self._check_variables()
        self._check_comments()
        self._check_best_practices()

    def _check_syntax(self) -> None:
        """Check for syntax errors and basic structural issues."""
        for node in ast.walk(self.ast_tree):
            # Check for empty code blocks
            if isinstance(node, (ast.For, ast.While, ast.If, ast.With)):
                if not node.body:
                    self.issues.append(CodeIssue(
                        getattr(node, 'lineno', 0),
                        'syntax_error',
                        f"Empty {node.__class__.__name__} block found",
                        'HIGH'
                    ))

    def _check_style(self) -> None:
        """Check code style using pycodestyle."""
        style_guide = pycodestyle.StyleGuide(quiet=True)
        
        # Create a temporary file for pycodestyle to analyze
        temp_file = Path('temp_code_review.py')
        try:
            temp_file.write_text(self.source_code)
            result = style_guide.check_files([temp_file])
            
            for line_number, offset, code, text, doc in result._deferred_print:
                self.issues.append(CodeIssue(
                    line_number,
                    'style_violation',
                    f"{code}: {text}",
                    'MEDIUM'
                ))
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _check_docstrings(self) -> None:
        """Check for missing or inadequate docstrings."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                has_docstring = False
                if node.body and isinstance(node.body[0], ast.Expr):
                    if isinstance(node.body[0].value, ast.Str):
                        has_docstring = True
                        # Check docstring quality
                        docstring = node.body[0].value.s
                        if len(docstring.strip()) < 10:
                            self.issues.append(CodeIssue(
                                node.lineno,
                                'docstring_quality',
                                f"Short or uninformative docstring in {node.__class__.__name__}",
                                'LOW'
                            ))
                
                if not has_docstring:
                    self.issues.append(CodeIssue(
                        node.lineno,
                        'missing_docstring',
                        f"Missing docstring in {node.__class__.__name__}",
                        'MEDIUM'
                    ))

    def _check_complexity(self) -> None:
        """Check for code complexity issues."""
        for node in ast.walk(self.ast_tree):
            # Check function complexity
            if isinstance(node, ast.FunctionDef):
                num_statements = len(list(ast.walk(node)))
                if num_statements > 50:
                    self.issues.append(CodeIssue(
                        node.lineno,
                        'complexity_issue',
                        f"Function '{node.name}' is too complex ({num_statements} statements)",
                        'MEDIUM'
                    ))

    def _check_variables(self) -> None:
        """Check for undefined and unused variables."""
        defined_vars: Set[str] = set()
        used_vars: Set[str] = set()
        builtins = set(dir(__builtins__))

        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in builtins:
                        used_vars.add(node.id)

        # Check for undefined variables
        undefined = used_vars - defined_vars
        for var in undefined:
            self.issues.append(CodeIssue(
                0,  # We don't have line numbers for this check
                'undefined_variable',
                f"Variable '{var}' is used but not defined",
                'HIGH'
            ))

    def _check_comments(self) -> None:
        """Analyze code comments for quality and formatting."""
        lines = self.source_code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                # Check for empty comments
                if len(stripped) == 1:
                    self.issues.append(CodeIssue(
                        i,
                        'comment_issue',
                        "Empty comment found",
                        'LOW'
                    ))
                # Check for space after #
                elif stripped[1] != ' ':
                    self.issues.append(CodeIssue(
                        i,
                        'comment_issue',
                        "Comments should have a space after '#'",
                        'LOW'
                    ))
                # Check for TODO comments
                elif 'TODO' in stripped.upper():
                    self.issues.append(CodeIssue(
                        i,
                        'comment_issue',
                        "TODO comment found - Consider addressing it",
                        'LOW'
                    ))

    def _check_best_practices(self) -> None:
        """Check for violations of Python best practices."""
        for node in ast.walk(self.ast_tree):
            # Check for excessive line length in strings
            if isinstance(node, ast.Str):
                if len(node.s) > 79:
                    self.issues.append(CodeIssue(
                        getattr(node, 'lineno', 0),
                        'best_practice',
                        "String literal is too long (> 79 characters)",
                        'LOW'
                    ))

    def get_report(self) -> str:
        """
        Generate a detailed report of all issues found during analysis.

        Returns:
            str: Formatted report of all issues
        """
        if not self.issues:
            return "No issues found. Code looks good! 🎉"

        # Sort issues by severity and line number
        sorted_issues = sorted(
            self.issues,
            key=lambda x: (
                {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x.severity],
                x.line_number
            )
        )

        report = ["Code Review Report", "=================\n"]
        
        # Group issues by severity
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            severity_issues = [i for i in sorted_issues if i.severity == severity]
            if severity_issues:
                report.append(f"{severity} Priority Issues:")
                report.append("-" * 20)
                for issue in severity_issues:
                    location = f"Line {issue.line_number}: " if issue.line_number else ""
                    report.append(f"{location}{issue.message}")
                report.append("")

        return "\n".join(report)

def main():
    """Main function to demonstrate the AI Code Reviewer usage."""
    # Example Python code to analyze
    example_code = """
def calculate_sum(numbers):
    #bad comment
    total = sum(numbers)
    print(undefined_variable)  # This will raise an issue
    return total

class ExampleClass:
    def method_without_docstring(self):
        pass

    def complicated_method(self):
        # TODO: Simplify this method
        result = 0
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    result += i * j * k
        return result
"""

    # Initialize and run the code reviewer
    reviewer = AICodeReviewer()
    reviewer.load_code(example_code)
    reviewer.analyze()
    print(reviewer.get_report())

if __name__ == "__main__":
    main()
