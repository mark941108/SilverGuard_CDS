---
name: python-vibe
description: Student-optimized Python development workflow. Use when writing Python code, including web apps (Streamlit/Gradio), data science, and ML projects. Emphasizes cost efficiency, safety checks, and smart model routing.
---

# 🐍 Python Vibe Skill

> **Version:** 2026.1.0  
> **Mode:** Student Optimized Configuration - Max Performance / Min Cost

This skill defines the preferred coding style, tool permissions, and model routing for Python development projects.

---

## 🎯 Core Philosophy

1.  **Cost Efficiency First**: Use the fastest/cheapest model for routine tasks; reserve powerful models for complex logic.
2.  **Safety Checks Strict**: Always ask for permission before running commands or writing files.
3.  **Minimal Context Footprint**: Limit scan depth and ignore irrelevant folders to save tokens.

---

## ⚙️ Configuration (Reference)

```json
{
  "version": "2026.1.0",
  "description": "Student Optimized Configuration - Max Performance / Min Cost",
  
  "settings": {
    "student_optimization_mode": true,
    "safety_checks": "strict"
  },

  "context_control": {
    "max_scan_depth": 5,
    "ignored_folders": [
      "node_modules",
      ".git",
      ".venv",
      "dist",
      "build",
      "__pycache__"
    ]
  },

  "model_routing": {
    "default_model": "gemini-3-flash",
    "rules": [
      {
        "task_type": ["refactor", "complex_logic", "security_audit"],
        "use_model": "claude-4.5-sonnet",
        "reason": "Highest logical reasoning capability"
      },
      {
        "task_type": ["documentation", "unit_test", "formatting", "routine"],
        "use_model": "gemini-3-flash",
        "reason": "Fast and high free quota"
      }
    ]
  },

  "skills_permission": {
    "terminal": {
      "enabled": true,
      "execution_mode": "ask_for_permission",
      "allowed_commands": ["npm", "pip", "git", "python", "streamlit"]
    },
    "browser": {
      "enabled": true,
      "headless": true
    },
    "filesystem": {
      "read": true,
      "write": "ask_for_permission"
    }
  }
}
```

---

## 🧪 Python Coding Standards

### Style Guide

-   **Formatter**: Use `black` with default settings.
-   **Linter**: Prefer `ruff` over `flake8` for speed.
-   **Docstrings**: Use Google style docstrings.
-   **Type Hints**: Strongly encouraged for all public functions.

### Import Order

```python
# 1. Standard Library
import os
import sys
from pathlib import Path

# 2. Third-Party Libraries
import numpy as np
import pandas as pd
import torch

# 3. Local Modules
from .utils import helper_function
```

### Error Handling

-   Always use explicit exception types (e.g., `ValueError`, `FileNotFoundError`), never bare `except:`.
-   For user-facing errors, provide clear, actionable messages.

---

## 🚀 Web App Development (Streamlit/Gradio)

When building web apps, follow this priority:

1.  **Streamlit** for quick data dashboards and internal tools.
2.  **Gradio** for ML demos and AI model interfaces (especially for Hugging Face Spaces).

### Gradio Boilerplate

```python
import gradio as gr

def process(input_data):
    # Your logic here
    return f"Processed: {input_data}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 My App")
    input_box = gr.Textbox(label="Input")
    output_box = gr.Textbox(label="Output")
    btn = gr.Button("Submit", variant="primary")
    btn.click(process, inputs=[input_box], outputs=[output_box])

demo.launch()
```

---

## 📦 Dependency Management

-   **Production**: Use `requirements.txt` with pinned versions (e.g., `transformers==4.40.0`).
-   **Development**: Use `pyproject.toml` with version ranges for flexibility.
-   **Pinning Strategy**: For critical libraries like `transformers` or `torch`, always pin to the tested version to avoid breaking changes.

---

## 🛡️ Safety Reminders

-   **Never hardcode API keys or secrets.** Use environment variables or `.env` files.
-   **Before running `pip install`**, confirm the package is from a trusted source.
-   **Before executing `python <script>` on user files**, confirm the action.

---

## 📝 Commit Message Style (Conventional Commits)

```
feat: Add new user login feature
fix: Resolve null pointer in data processing
docs: Update README with installation instructions
refactor: Simplify API response handling
test: Add unit tests for payment module
chore: Update dependencies
```

---

## Origin

This skill is tailored to the user's preferences as observed during the MedGemma Impact Challenge (January 2026). It reflects a "student optimization" mindset: maximizing performance while minimizing API costs and cognitive load.
