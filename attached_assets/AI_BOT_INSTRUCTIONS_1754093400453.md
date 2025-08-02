
















✅ The document is now fully updated with your **final instruction**, including:

* How **online/offline fallback** works
* That **only one model is active at a time**
* Full explanation for how each feature is powered by the AI/LLM
* And how the coding bot should handle any API failure scenarios

You're now ready to pass this doc to the frontend/backend AI developer or plug into an AI coding assistant.

Would you like me to export this now as `.md`, `.txt`, or `.pdf`?
   **📢 Final Instruction for the AI Bot (StudyMate Frontend Builder)**

---

### 🔧 GLOBAL RULES

* Do **not** change sidebar items or their function names (e.g. `render_chat_mode()`, `render_summarize()`, etc.).
* Use all Python libraries listed in `requirements.txt` to their full capabilities.
* Ensure **no syntax or formatting errors**, especially for LaTeX, Math, or Markdown blocks.
* UI should explain each feature clearly on open.
* All modes must support and access:

  * Multi-file uploads
  * Folder uploads
  * Live updating via `Live Uploads`
  * OCR fallback (pdf2image + pytesseract)
  * Vector DB via FAISS/Chroma

---

### 💬 Chat Mode (`render_chat_mode()`)

* Uses document-based Q\&A by default.
* If question is **outside document scope**, AI switches to web tools (`duckduckgo`, `wikipedia`, etc.).
* Web content fetched via Python tool and integrated into AI responses.
* Response is streamed using `streamlit-chat`.
* User can toggle which model to use (OpenAI GPT, Gemini, Claude, or local offline models).
* If online API key is missing or usage quota fails mid-process, **fallback to offline LLM**.

---

### 📝 Summarize Mode (`render_summarize()`)

* Multi-select uploaded files.
* Ask user:

  * Difficulty: Beginner / Intermediate / Advanced
  * Style (multi-select): Paragraph, Bullet points, Table
  * Include images: Yes/No
* Multiple summary types can be selected simultaneously.
* AI summarizes based on selected tone and structure.

---

### 🛠 Customize Mode (`render_customize()`)

* Select any uploaded file or paste custom text.
* Configure:

  * Tone (Simple, Formal, Clear)
  * Depth (Brief, In-depth)
  * Output format (error-free Markdown)
* Ensure that Markdown content is syntax clean and professional.

---

### 🎯 Topic Search (`render_topic_search()`)

* User inputs comma-separated list of topics (up to 20).
* Each topic is searched independently:

  * Local DB
  * Internet (Wikipedia, Arxiv, DuckDuckGo)
* Allow user to choose summary type:

  * Basic, Intermediate, Advanced, or Full Web Dump
* Allow export to Summary or Flashcards.

---

### 🖼 Image Mode (`render_image()`)

* User can:

  * Upload new image
  * Select images from previously uploaded PDFs
* AI outputs:

  * Full description of content
  * Caption and analysis using OCR + image parser
* Automatic access to uploaded PDF image content — no re-upload needed.

---

### 📊 Advanced Tables (`render_advanced_tables()`)

* Extract tables from:

  * Uploaded PDFs (via `camelot`, `pdfplumber`, `tabula`)
  * Charts/images inside PDF pages
  * Directly uploaded tables
* Show page selectors
* Output:

  * Summary
  * Export options (CSV, Excel, JSON)

---

### 🌐 Web Search (`render_web_search()`)

* Input: multiple topics (comma-separated)
* Output:

  * Each topic detailed separately
  * User selects style (basic/intermediate/advanced/or original result)
* Clean results only, no ads or spam
* Exportable to summary or flashcards

---

### 📚 Study Planner (`render_study_planner()`)

* Select files + enter topics manually
* Configure:

  * Duration (1 to 6 weeks)
  * Daily hours (1 to 24)
  * Start/End time for the day
  * Breaks every: 20/30/40/50 minutes
* Output:

  * Intelligent daily plan with goals (summary, flashcards)
  * Includes food/rest breaks
  * Saves to session

---

### 📈 Study Progress (`render_study_progress()`)

* Track usage of flashcards, summaries, study plan
* Display progress in % + milestone checklist
* Read from session/history/bookmarks

---

### 🧠 Flashcards/Notes (`render_flashcards()`)

* Select uploaded files (multi-select)
* Input: comma-separated topics
* Ask:

  * Max cards (up to 100)
  * Flashcard types (multi-select):

    * True/False
    * Q\&A
    * One-word
    * Fill-in-the-blank
    * Application-based
    * Definitions
    * Formulas (for math/physics)
  * Difficulty: Easy / Intermediate / Difficult
* Fully utilize:

  * PDF parser
  * Wikipedia / DuckDuckGo / Arxiv
  * FAISS and embeddings
* Save to:

  * Bookmarks
  * Session
  * Export panel (PDF/Markdown/JSON)

---

### ⚙ Settings, ❓ Help, 🔍 System Status

* No changes needed.
* Add descriptive tooltips if needed.

---

### ✅ MODEL CONFIG

* Only Tier 1 models remain (OpenAI, Gemini, Claude)
* Offline LLMs replaced with **5 HuggingFace models** stored locally
* AI model toggle shows:

  * Online (API key) options
  * Offline models (1 at a time only)
* If online API key is missing, or API call fails mid-process:

  * Automatically fallback to offline model
  * Optionally fallback to another online model if available
* All AI features are powered by the selected LLM (online or offline)

---

### 🔌 HOW IT WORKS (for coding bot)

* Every major feature is AI-powered:

  * Flashcards → LLM + vector search + web fallback
  * Summary → LLM + structure builder + OCR + image + tables
  * Chat → Document + Web-based responses
  * Study Plan → LLM + difficulty prediction + time schedule logic
* Core backend logic:

  * If user selected online model → Use API via `.env` keys
  * If no key or usage limit reached → auto-switch to offline model from `models/`
  * Offline models are accessed via local Python inference scripts
  * Only **one model is used at a time** (never combined)

---
Awesome — that means you're right at the finish line! 🎯 Here's a final checklist + optional last words to tell the bot, just to *seal the deal* and avoid any misunderstanding:

---

### ✅ ✅ **Final Additions to Tell the AI Bot:**

#### 🔒 1. **Don't Break the Logic or Rewire Existing UI Flow**

* Don't change any core route function names like `render_chat_mode()`, etc.
* Use the file structure as it is (especially `/ui/components`, `/core`, `/models`, `/utils`, etc.).
* Maintain the sidebar order and existing mode switches.

#### 🧠 2. **Every Feature = Intelligent & Guided UI**

* On click, show helper tooltips or a brief “What this does” explanation in **each mode**.
* No feature should feel empty or technical. Every user-facing screen should feel **humanized and guided**.

#### 🛡️ 3. **Fallback & Model Switching Logic**

* If a model is selected and fails (e.g., API expired), **automatically switch to available offline models** without crashing.
* All processing (flashcards, summarization, chat, etc.) must work in **both online and offline models** — one at a time.

#### 📁 4. **Multi-file + Folder Uploads Always Enabled**

* Every mode (summarize, chat, flashcard, image, etc.) should **support multi-file selection** and **folder parsing** (if user used that).

#### 🧠 5. **Model Capabilities Awareness**

* Don’t try to make a small model (like DialoGPT or 2B models) perform huge summaries or long chats unless chunked.
* Show a toast or message like:

  > “⚠️ Selected model has lower capacity. Consider using a larger model or shorter input.”

#### 💡 6. **Offline Models Should Be Used Intelligently**

* Tell the bot to **map features** to the best offline models when fallback happens:

  * Summarization → `flan-t5` or `mistral`
  * Q\&A → `granite` or `mistral`
  * Dialogue → `DialoGPT`
  * Table → Leave to parser tools (`camelot`, `pdfplumber`) — not LLMs

---

### 🧾 Bonus (Optional):

If the bot supports "developer instructions" or “AI memory/config JSON,” you can even ask it:

> “Save all this config into a file like `ai_config.json` or `frontend_instruction.md` so it remembers everything when restarted.”

---

Let me know if you'd like me to generate that file for upload too (for safety/documentation). You're almost done with an incredible platform here — proud of how deep you’ve taken it! 🔥

















Perfect follow-up question.

You're building **more than an LLM app** — you're creating a **modular educational AI platform** with document parsing, multimodal input (image/audio), vector search, web tools, export, and smart UI features.

Here’s a complete list of **tools, libraries, and technologies you'll need beyond LLMs**, organized by functionality:

---

## 🧩 1. **Core Python Utilities**

These are used *throughout* the app for file handling, storage, logic, and system introspection.

| Purpose              | Tools                                                      |
| -------------------- | ---------------------------------------------------------- |
| File handling, paths | `os`, `shutil`, `pathlib`, `zipfile`, `glob`               |
| Data storage         | `json`, `pickle`, `uuid`, `datetime`, `base64`             |
| System info          | `platform`, `psutil` (RAM/CPU info for performance tuning) |
| Regex/cleaning       | `re`, `unicodedata`, `html2text`, `html.parser`            |

---

## 📂 2. **Document & File Processing**

Used to read, parse, and extract content from documents and folders.

| File Type       | Libraries                                      |
| --------------- | ---------------------------------------------- |
| **PDFs**        | `PyMuPDF (fitz)`, `pdfminer.six`, `pdfplumber` |
| **DOCX**        | `python-docx`                                  |
| **TXT/MD/RTF**  | Native file read, or `markdown2` for .md       |
| **Excel/CSV**   | `pandas`, `openpyxl`, `xlrd`, `csv`            |
| **ZIP/Folders** | `zipfile`, `os.walk`, `pathlib`                |
| **OCR in PDF**  | `pytesseract`, `pdf2image`                     |

---

## 🔍 3. **Vector Embedding + Search (Semantic Retrieval)**

Essential for document understanding, chat context, and topic search.

| Purpose              | Libraries                                              |
| -------------------- | ------------------------------------------------------ |
| Vector search        | `faiss`, `chromadb`                                    |
| Embeddings (offline) | `sentence-transformers`, `InstructorXL`, `bge-base-en` |
| Chunking & index     | `langchain`, `tiktoken`, `nltk` (optional)             |

---

## 🖼️ 4. **Image Processing & Captioning**

Used in **Image Mode** to extract or describe content in diagrams, scanned documents, charts.

| Feature                          | Tools                                                            |
| -------------------------------- | ---------------------------------------------------------------- |
| OCR (text from image)            | `pytesseract`, `PIL`, `pdf2image`                                |
| Captioning / image understanding | `transformers` with `BLIP`, `CLIP`, or `ImageGPT`                |
| Image UI                         | `streamlit-image-coordinates`, `cv2`, `matplotlib` (for display) |

---

## 📊 5. **Tables, Forms, and Advanced Data Extraction**

Used in **Advanced Tables** to process structured content.

| Format         | Tools                                                                |
| -------------- | -------------------------------------------------------------------- |
| Tables in PDFs | `camelot`, `tabula-py` (requires Java), `pdfplumber`                 |
| Excel + CSV    | `pandas`, `openpyxl`, `csv`, `xlrd`                                  |
| Forms          | Combine `pdf2image` + `pytesseract` + layout parser (`layoutparser`) |

---

## 🌐 6. **Web Search + External Knowledge**

Used in **Topic Search**, **Web Search**, or fallback LLM queries.

| Source                  | Tools                                    |
| ----------------------- | ---------------------------------------- |
| DuckDuckGo              | `duckduckgo-search`                      |
| Wikipedia               | `wikipedia-api`, `wikipedia`             |
| arXiv papers            | `arxiv`                                  |
| Web scraping (optional) | `requests`, `BeautifulSoup`, `html2text` |
| Web browser (headless)  | `selenium` or `Playwright` (advanced)    |

---

## 📤 7. **Export & Session Management**

To save chat logs, flashcards, summaries, and session states.

| Purpose           | Libraries                                                        |
| ----------------- | ---------------------------------------------------------------- |
| Exporting content | `markdown`, `fpdf`, `pdfkit`, `html2text`                        |
| ZIP creation      | `zipfile`, `shutil`                                              |
| Session storage   | `pickle`, `json`, `uuid`, `base64`, `TinyDB` (optional local DB) |

---

## 🎙️ 8. **Voice Input (Optional, for Mic Users)**

Used for real-time speech-to-text input.

| Feature            | Tools                                                 |
| ------------------ | ----------------------------------------------------- |
| Speech recognition | `speech_recognition`, `pyaudio`, `whisper` (optional) |
| Microphone access  | `pyaudio`, `sounddevice`                              |

---

## 🎛️ 9. **User Interface (via Streamlit)**

Used to build interactive pages, sidebars, widgets, sliders, and controls.

| Purpose              | Libraries                                                                |
| -------------------- | ------------------------------------------------------------------------ |
| UI layout            | `streamlit`, `streamlit-option-menu`, `streamlit_tags`, `streamlit_chat` |
| Sidebar menus        | `st.sidebar`, `st.selectbox`, `st.radio`, `st.expander`                  |
| File upload          | `st.file_uploader`, `st.camera_input`                                    |
| Theming              | `streamlit.config.toml`, dark mode toggle                                |
| Custom JS (optional) | `streamlit.components.v1`                                                |

---

## 🧪 10. **Testing / Monitoring / Logging**

Optional, but useful for production stability.

| Purpose       | Tools                                |
| ------------- | ------------------------------------ |
| Logging       | `logging`, `loguru`                  |
| Metrics       | `psutil`, `time`, `tracemalloc`      |
| Error capture | Try/except + `sentry_sdk` (optional) |

---

## 🛡️ 11. **Security / API Key Management**

For storing secrets and managing access.

| Feature       | Tools                                             |
| ------------- | ------------------------------------------------- |
| Secrets file  | `streamlit.secrets`, `dotenv` (`.env`)            |
| Secure config | `cryptography`, `fernet`, `pyAesCrypt` (optional) |
| API test      | Use basic test call for validation                |

---


---

## ✅ BONUS (Optional Enhancers)

| Use Case         | Tools                                            |
| ---------------- | ------------------------------------------------ |
| App analytics    | `Mixpanel`, `PostHog`, or log dashboard          |
| Error reporting  | `sentry_sdk`                                     |
| UI themes        | `Tailwind`, `shadcn/ui` (for web apps)           |
| Fast API wrapper | `FastAPI`, `Flask` (for backend or API endpoint) |

---

## 🚀 RECOMMENDATION

Would you like me to:

* ✅ Create a `requirements.txt` grouped by feature/module?
* ✅ Or generate a `setup.py` / `conda.yaml` file?
* ✅ Or output the `pip install` commands for only the things you're missing?

Let me know how you want to proceed.
