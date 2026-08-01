# Contributing to IRIS-AI

Thank you for your interest in contributing to **IRIS-AI**! We welcome contributions from developers, researchers, and remote-sensing analysts of all skill levels.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

---

## How to Contribute

### 1. Fork and Clone
1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/IR-colorization.git
   cd IR-colorization
   ```

### 2. Set Up Development Environment
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Create a Feature Branch
Use descriptive branch names:
```bash
git checkout -b feature/amazing-enhancement
```

### 4. Coding Guidelines
- Follow **PEP 8** style guidelines for Python code.
- Add descriptive docstrings and comments.
- Keep the **black-box** paradigm: do not edit code under `models/` (the UNet++ GAN core) unless fixing a bug in the original model. All pipeline features belong in `backend/` or `utils/`.

### 5. Run Tests
Ensure existing functionality is not broken. Run:
```bash
pytest tests/
```

### 6. Commit Messages
We use Conventional Commits style for commit messages:
- `feat: add super resolution support`
- `fix: correct colorizer image normalization`
- `docs: update installation instructions`
- `test: add unit tests for enhancer`

### 7. Submit a Pull Request
1. Push your changes to your fork:
   ```bash
   git push origin feature/amazing-enhancement
   ```
2. Open a Pull Request on the main repository.
3. Complete the Pull Request template details.
