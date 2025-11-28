---
title: Tds Quiz Solver
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Project Description

# LLM Quiz Solver

Automated quiz solving system using LLMs and browser automation for the TDS LLM Analysis Quiz project.

## Features

- 🤖 LLM-powered question understanding and solving
- 🌐 Headless browser automation with Playwright
- 📊 Multi-format data parsing (PDF, CSV, Excel, JSON, HTML)
- 🔄 Automatic retry mechanisms
- 📈 Code generation and execution for complex analysis
- 🎨 Visualization generation (charts, plots)
- ⚡ Async/await throughout for performance
- 📝 Structured logging with JSON output
- 🐳 Docker support for easy deployment

## Tech Stack

- **Framework**: FastAPI 0.109+
- **LLM**: OpenAI GPT-4o
- **Browser**: Playwright (Chromium)
- **Data Processing**: Pandas, NumPy
- **Parsing**: PyPDF2, openpyxl, BeautifulSoup4
- **Visualization**: Matplotlib, Plotly
- **Logging**: structlog
- **Retry**: tenacity
- **Testing**: pytest, pytest-asyncio

## Project Structure

```
llm-quiz-solver/
├── src/
│   ├── api/           # FastAPI application
│   ├── core/          # Core orchestration
│   ├── services/      # Business logic services
│   ├── parsers/       # File parsers
│   └── utils/         # Utilities
├── tests/             # Test suite
├── deployment/        # Docker & K8s configs
├── requirements.txt
└── README.md
```
