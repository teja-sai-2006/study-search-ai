# StudyMate - AI-Powered Learning Assistant

## Overview

StudyMate is a comprehensive AI-powered learning platform built with Streamlit that provides intelligent document processing, content analysis, and study tools. The application offers multiple specialized modes for different learning tasks including chat-based Q&A, document summarization, flashcard creation, study planning, and progress tracking. It supports multi-format document uploads with OCR fallback capabilities and integrates both online AI models (OpenAI, Gemini, Anthropic) with offline model alternatives for reliable operation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses Streamlit as the primary frontend framework with a modular UI design pattern. The interface is organized into distinct modes (chat, summarize, customize, etc.) that are rendered through dedicated modules in the `ui/modes/` directory. Reusable UI components are housed in `ui/components/` for consistent interface elements like file uploaders, chat interfaces, and export managers. The main application entry point (`app.py`) orchestrates the overall user experience and navigation between different modes.

### Backend Architecture
The core processing logic is distributed across specialized utility modules in the `utils/` directory. The LLM Engine (`llm_engine.py`) handles multi-model AI integration with intelligent fallback mechanisms - it attempts online models first (OpenAI, Gemini, Anthropic) and falls back to offline models when API keys are unavailable or quota limits are reached. Document processing (`document_processor.py`) supports multiple file formats with OCR fallback using pdf2image and pytesseract when standard parsing fails. The search engine (`search_engine.py`) implements vector-based semantic search using FAISS or Chroma databases with simple text search as fallback.

### Session and State Management
Session management is handled through the `SessionManager` class which provides persistent storage of user sessions, document uploads, and application state. Settings are managed via `SettingsManager` which handles user preferences, API keys, and configuration persistence. The application maintains state across user interactions using Streamlit's session state mechanism combined with local file persistence.

### Document Processing Pipeline
The document processing system supports PDF, DOCX, TXT, MD, RTF, Excel, CSV, ZIP, and image formats. When primary parsers fail, the system automatically falls back to OCR processing for text extraction. Images are processed through both OCR (pytesseract) and AI-powered captioning (BLIP model) for comprehensive content understanding. Tables are extracted using multiple methods (Camelot, PDFplumber, Tabula) with automatic deduplication and quality assessment.

### AI Model Management
The system implements a tiered model approach with online models as primary options and offline models as fallbacks. Model availability is checked at startup, and the system gracefully handles API failures or missing credentials. Offline models include DialoGPT, FLAN-T5, Granite, and Mistral variants, providing specialized capabilities for different tasks (chat, summarization, analysis).

## External Dependencies

### AI Model APIs
- **OpenAI API**: Primary LLM provider for GPT-4 and GPT-3.5-turbo models requiring OPENAI_API_KEY environment variable
- **Google Gemini API**: Alternative LLM provider supporting vision capabilities requiring GEMINI_API_KEY
- **Anthropic Claude API**: Claude-3 model family access requiring ANTHROPIC_API_KEY
- **Hugging Face**: Offline model hosting and transformer library integration for fallback models

### Document Processing Libraries
- **PyPDF2/PDFplumber**: Primary PDF text extraction with fallback to OCR
- **python-docx**: Microsoft Word document processing
- **openpyxl/xlrd**: Excel file processing and table extraction
- **pdf2image**: PDF to image conversion for OCR fallback
- **pytesseract**: OCR text extraction from images and PDF pages

### Vector Database and Search
- **FAISS**: Primary vector database for semantic document search
- **Chroma**: Alternative vector database implementation
- **sentence-transformers**: Text embedding generation using all-MiniLM-L6-v2 model

### Web Integration
- **DuckDuckGo Search API**: Web search capabilities for information retrieval
- **Wikipedia API**: Structured knowledge access for educational content
- **arXiv API**: Academic paper search and access
- **trafilatura**: Clean web content extraction and scraping
- **BeautifulSoup**: HTML parsing and content cleaning

### Image Processing
- **PIL (Pillow)**: Basic image processing and format conversion
- **Transformers (BLIP)**: AI-powered image captioning and description
- **OpenCV**: Advanced image processing for document analysis

### UI and Visualization
- **Streamlit**: Primary web framework for the user interface
- **streamlit-chat**: Enhanced chat interface components
- **pandas**: Data manipulation and table processing
- **plotly**: Interactive charts and visualizations for progress tracking

### File Handling
- **zipfile**: Archive processing and bulk document handling
- **tempfile**: Secure temporary file management for uploads
- **pathlib**: Cross-platform file path handling

### Configuration and Persistence
- **JSON**: Configuration file management and session persistence
- **logging**: Application monitoring and error tracking
- **datetime**: Timestamp management and scheduling features