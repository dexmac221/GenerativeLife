# Contributing to GenerativeLife

Thank you for your interest in contributing to GenerativeLife! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/dexmac221/GenerativeLife.git
   cd GenerativeLife
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Development

### Code Style

- Follow Python PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to new functions and classes
- Keep functions focused and modular

### Testing

- Test your changes with different configurations
- Run basic simulations to ensure stability:
  ```bash
  python ai_arena.py --agents 3 --turns 20
  ```

### Project Structure

- `modular/core/` - Core simulation logic (World, Agent, Simulation)
- `modular/ai/` - AI controllers and decision-making systems
- `modular/ui/` - Visualization components
- `modular/utils/` - Utilities and configuration
- `prompts/` - LLM prompt templates

## 🎯 Areas for Contribution

### 🧠 AI Controllers

- New decision-making algorithms
- Alternative LLM integrations
- Rule-based fallback systems

### 🎭 Agent Personalities

- New personality types
- Behavioral pattern analysis
- Crisis response systems

### 🌍 World Features

- New environmental elements
- Resource types and dynamics
- Weather and seasonal systems

### 📊 Data Analysis

- Training data analysis tools
- Behavior pattern visualization
- Performance optimization

### 🎮 User Interface

- Enhanced visualization modes
- Real-time monitoring tools
- Configuration interfaces

## 📝 Submitting Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits:

   ```bash
   git commit -m "Add new personality type: strategic planner"
   ```

3. **Test your changes**:

   ```bash
   python ai_arena.py --agents 5 --turns 30
   python analyze_data.py  # If you added data analysis features
   ```

4. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Screenshots/examples if applicable
   - Testing notes

## 🐛 Bug Reports

When reporting bugs, include:

- Python version and OS
- Full error message and traceback
- Steps to reproduce
- Configuration used

## 💡 Feature Requests

For new features, please:

- Check existing issues first
- Describe the use case
- Provide examples if possible
- Consider implementation approach

## 📚 Documentation

- Update README.md for significant changes
- Add docstrings to new functions
- Include examples in code comments
- Update configuration documentation

## 🔄 Review Process

- Maintain backward compatibility when possible
- Ensure code follows project patterns
- Test with multiple agent configurations
- Verify data collection still works

## 📞 Getting Help

- Create an issue for questions
- Include relevant code and configuration
- Be specific about what you're trying to achieve

Thank you for contributing to GenerativeLife! 🎉
