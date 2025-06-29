# GenerativeLife Documentation Index

## Quick Links

- [Main README](../README.md) - Primary project documentation
- [Configuration Guide](../config.yaml) - All configuration options explained

## Core Components

### Simulation Engine

Located in `/modular/` directory:

- **Core** - World simulation, agent logic, and main simulation loop
- **AI** - LLM controllers, spatial intelligence, and prompt enhancement
- **UI** - Real-time visualization and threaded rendering
- **Utils** - Configuration management and data collection

### Configuration and Data

- `config.yaml` - Central configuration file
- `prompts/` - Personality-driven prompt templates
- `data/` - Training data collection and session storage

- `analyze_training_data.py` - Training data analysis and statistics
- `breeding_demo.py` - Demonstration of breeding mechanics
- `data_collection_demo.py` - Example data collection workflows
- `demo_modular.py` - Modular architecture demonstration
- `lora_training_template.py` - LoRA fine-tuning template

## Architecture Overview

### Core Components

- **World**: Environment grid with resources and obstacles
- **Agent**: Autonomous entities with AI-driven decision making
- **Simulation**: Main simulation loop and orchestration
- **Controllers**: AI decision-making systems (Simple, VLM)

### AI Systems

- **Simple Controller**: Rule-based decision making for baseline behavior
- **VLM Controller**: Large Language Model powered reasoning and decision making
- **Prompt System**: Personality-driven template system for diverse agent behaviors

### Visualization

- **Threaded Visualizer**: Real-time PyGame rendering with agent details
- **Speech Bubbles**: Agent thought display and debugging information
- **Statistics Panel**: Population and resource tracking

## Configuration

### Key Settings

- **World size**: Grid dimensions and resource distribution
- **Agent properties**: Starting population, breeding parameters, survival mechanics
- **AI settings**: Model selection, prompt templates, inference parameters
- **Simulation**: Turn limits, speed, data collection options

### Personality Templates

Located in `/prompts/` directory:

- `cautious_explorer.txt` - Safety-focused exploration
- `aggressive_hoarder.txt` - Resource accumulation priority
- `social_collaborator.txt` - Cooperation and mate-seeking
- `curious_wanderer.txt` - Exploration and discovery focus
- Plus crisis and specialized templates

## Contributing

### Development Setup

1. Follow installation instructions in main README
2. Run basic simulation test: `python ai_arena.py --agents 2 --turns 10`
3. Check configuration: Review `config.yaml` settings
4. Test different models: Modify model settings in config

### Code Organization

- Follow existing modular structure
- Add tests for new features
- Update documentation for significant changes
- Use debug tools for development and testing

### Performance Considerations

- VLM inference can be slow; consider caching and batching
- Large populations may require performance optimization
- Monitor memory usage with extensive data collection

## Troubleshooting

### Common Issues

1. **VLM connection errors**: Check Ollama installation and model availability
2. **Slow performance**: Reduce population size or disable GUI
3. **Memory issues**: Clear data collection or reduce simulation length
4. **Import errors**: Verify all dependencies are installed

### Debug Workflow

1. Run basic simulation test first: `python ai_arena.py --agents 2 --turns 10`
2. Check logs in `data/sessions/` for detailed execution traces
3. Verify configuration in `config.yaml`
4. Test with different models and settings

## License and Attribution

GenerativeLife is open source software. See main README for licensing details and contribution guidelines.
