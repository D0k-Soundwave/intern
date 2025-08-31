"""
MCP-based Claude Code + Ollama Integration System

This system provides intelligent routing between Claude Code and Ollama
based on interview-driven capability assessment using MCP protocol.

Per requirements.md: No API keys needed, zero technical debt policy.
"""
import os
import sys
import logging

# Force requirements check on import per requirements.md
requirements_file = os.path.join(os.path.dirname(__file__), '..', 'Docs', '1. requirements.md')
if not os.path.exists(requirements_file):
    sys.stderr.write("ERROR: requirements.md not found\n")
    sys.exit(1)

# Log reminder per requirements.md
logging.info("REMINDER: Review requirements.md before making changes")

__version__ = "1.0.0"
__author__ = "Claude Code + Ollama Integration System"