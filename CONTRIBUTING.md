# Contributing to FATES-Emulator

Thank you for your interest in contributing to FATES-Emulator! This framework builds on the methodology from Li et al. (2023) GMD and aims to make FATES calibration accessible to the research community.

## 🎯 Ways to Contribute

### 1. Report Bugs
- Use GitHub Issues to report bugs
- Include: OS, Python version, error message, minimal example
- Check existing issues first to avoid duplicates

### 2. Suggest Features
- Open an issue with the "enhancement" label
- Describe the feature and use case
- Discuss before implementing major features

### 3. Improve Documentation
- Fix typos, clarify explanations
- Add examples or tutorials
- Translate documentation

### 4. Add Code
- Fix bugs
- Implement new features
- Add tests
- Improve performance

### 5. Share Examples
- Add new site configurations
- Create Jupyter notebooks
- Share calibration results

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Conda (recommended) or pip

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/fates-emulator.git
cd fates-emulator

# Create development environment
conda env create -f environment.yml
conda activate fates-emulator

# Install in editable mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=fates_emulator tests/

# Run specific test
pytest tests/test_sampling.py
```

## 📝 Code Style

### Python Style Guide
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions (NumPy style)
- Keep functions focused and < 50 lines when possible

### Example Function

```python
def train_emulator(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_budget: int = 600
) -> FATESEmulator:
    """
    Train FATES emulator using FLAML AutoML.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (parameters)
    y_train : pd.Series
        Training target (FATES output)
    time_budget : int
        AutoML time budget in seconds
        
    Returns
    -------
    FATESEmulator
        Trained emulator
        
    Examples
    --------
    >>> emulator = train_emulator(X_train, y_train, time_budget=300)
    >>> emulator.save('model.pkl')
    """
    # Implementation...
```

### Code Formatting

We use:
- **black** for code formatting
- **flake8** for linting
- **isort** for import sorting

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Sort imports
isort src/ tests/
```

## 🔀 Git Workflow

### Branch Naming
- `feature/add-new-emulator` - New features
- `fix/calibration-bug` - Bug fixes
- `docs/update-readme` - Documentation
- `test/add-unit-tests` - Tests

### Commit Messages
Follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(emulator): add support for LightGBM models
fix(calibration): correct coexistence constraint calculation
docs(tutorial): add Jupyter notebook for Step 2
```

### Pull Request Process

1. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation
   - Run tests locally

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

4. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open Pull Request**
   - Use PR template
   - Link related issues
   - Request review

6. **Address Review Comments**
   - Make requested changes
   - Push updates
   - Re-request review

7. **Merge**
   - Maintainer will merge after approval
   - Delete branch after merge

## ✅ Pull Request Checklist

- [ ] Code follows style guide
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] No hardcoded paths
- [ ] Config files used for settings
- [ ] Proper error handling
- [ ] Logging (not print statements)

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── test_sampling.py          # Unit tests for sampling
├── test_emulator.py           # Unit tests for emulator
├── test_calibration.py        # Unit tests for calibration
├── test_preprocessing.py      # Unit tests for preprocessing
└── integration/
    └── test_workflow.py       # Integration tests
```

### Writing Tests

```python
import pytest
from fates_emulator import sampling

def test_generate_lhs_samples():
    """Test Latin Hypercube sampling."""
    samples = sampling.generate_lhs_samples(
        param_names=['a', 'b'],
        bounds=[[0, 1], [0, 1]],
        n_samples=100,
        seed=42
    )
    
    assert len(samples) == 100
    assert list(samples.columns) == ['a', 'b']
    assert samples['a'].min() >= 0
    assert samples['a'].max() <= 1
```

## 📚 Documentation Guidelines

### Docstring Format (NumPy Style)

```python
def function_name(param1, param2):
    """
    Short description (one line).
    
    Longer description if needed. Explain what the function does,
    not how it does it.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
        
    Returns
    -------
    type
        Description of return value
        
    Raises
    ------
    ValueError
        When and why this is raised
        
    See Also
    --------
    related_function : Related functionality
    
    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    
    Notes
    -----
    Additional notes, algorithms, references.
    """
```

### Documentation Files

Add/update relevant documentation in `docs/`:
- Step-by-step guides
- API reference
- Examples and tutorials
- FAQ

## 🤝 Code Review Process

### For Reviewers
- Be constructive and respectful
- Explain the "why" behind suggestions
- Approve when code meets standards
- Test locally if possible

### For Contributors
- Respond to all comments
- Ask questions if unclear
- Mark conversations as resolved
- Be open to feedback

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in papers using the framework (for significant contributions)

## 📞 Getting Help

- **Documentation**: Start with `docs/`
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: lingcheng.li@pnnl.gov for sensitive matters

## 📜 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior:**
- Trolling, insulting/derogatory comments, personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement
Project maintainers have the right to remove, edit, or reject comments, commits, code, issues, and other contributions that do not align with this Code of Conduct.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, helps make FATES-Emulator better for the research community!

---

**Questions?** Open an issue or contact lingcheng.li@pnnl.gov

**Citation**: Li, L., et al. (2023). A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0). Geosci. Model Dev., 16, 4017–4040. https://doi.org/10.5194/gmd-16-4017-2023

