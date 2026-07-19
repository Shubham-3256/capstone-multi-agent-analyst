# Frequently Asked Questions

Frequently asked questions regarding setup, integration, and execution.

---

### Q: Why does PDF generation fail with WeasyPrint libraries?
**A**: WeasyPrint requires the Cairo and Pango shared library packages to compile HTML into PDF. Refer to the [Installation Manual](installation.md) to install the appropriate system libraries for your OS (Ubuntu, macOS, or Windows). Alternatively, deploy using our **Docker container** to completely bypass local library installations.

---

### Q: How do I select which LLM provider to use?
**A**: The platform uses Pydantic Settings configuration variables. Fill in `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` in your `.env`. If multiple keys are provided, the default choice (controlled by `MODEL_NAME`, e.g. `gpt-4o`) will take precedence.

---

### Q: Can I run this offline?
**A**: Yes! The ingestion, cleaning, feature engineering, and model training subsystems run 100% locally. Generative text summaries (via LLMs) require internet access to talk to provider endpoints, but they can be bypassed or cached.

---

### Q: What is the minimum dataset size requirement?
**A**: The system executes successfully for datasets of any size. For extremely small datasets (fewer than 10 rows), the pipeline dynamically bypasses train-val splitting, switches model cross-validation to Leave-One-Out (LOOCV), and displays an information banner in the UI to notify the user.

---

### Q: Why does SQLite throw "database is locked" errors in thread pools?
**A**: SQLite has limited support for concurrent writes. To prevent locking issues in multi-threaded environments, we use thread-local database sessions and configure the SQLAlchemy connection engine with `connect_args={"check_same_thread": False}`.
