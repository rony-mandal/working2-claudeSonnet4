# Advanced Narrative Spread Simulation with GANs

This project simulates the spread of narratives in a population using agent-based modeling with Mesa 3.2.0. **Enhanced with Generative Adversarial Networks (GANs)** for realistic narrative generation, developed for DRDO-ISSA Lab.

## 🚀 New Features with GAN Integration

### 🤖 AI-Powered Narrative Generation
- **Dynamic Narrative Creation**: GANs generate contextually relevant narratives during simulation
- **Intelligent Counter-Narratives**: AI creates sophisticated opposing narratives instead of simple negations
- **Adaptive Content Injection**: System automatically generates new narratives based on simulation state
- **Realistic Language Patterns**: GAN learns from training data to produce human-like narrative text

### 📊 Enhanced Analytics
- **GAN Performance Metrics**: Track AI-generated vs. original narrative performance
- **Sentiment Analysis Comparison**: Compare sentiment patterns between human and AI narratives
- **Generation Rate Monitoring**: Analyze how often new narratives are created
- **Event Timeline Visualization**: Track all simulation events including GAN injections

## 🛠 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster GAN training)
- At least 4GB RAM (8GB recommended for larger simulations)

### Installation

1. **Clone the repository and navigate to the project directory**

2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories:**
   ```bash
   mkdir models data
   ```

4. **Run the enhanced dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage Guide

### Basic Operation

1. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

2. **Configure GAN Settings (Sidebar):**
   - ✅ Enable GAN Mode for AI-powered narrative generation
   - 📂 Option to load pre-trained models
   - ⚙️ Adjust generation parameters

3. **Choose Data Source:**
   - **Manual Input**: Enter custom narratives
   - **Preloaded Data**: Select from scenario datasets (War/Conflict, Economic Crisis, Health Emergency, etc.)

4. **Set Simulation Parameters:**
   - Number of agents (10-1000)
   - Simulation steps (1-100)
   - GAN injection rate (how often new narratives are generated)

5. **Advanced Options:**
   - Enable/disable counter-narratives
   - Activate adaptive GAN injection
   - Configure network topology

### GAN-Specific Features

#### 🤖 GAN Mode Benefits
- **Realistic Narratives**: AI generates contextually appropriate content
- **Emergent Patterns**: New narrative themes emerge during simulation
- **Counter-Intelligence**: Sophisticated opposition narratives
- **Scalability**: Generate hundreds of unique narratives automatically

#### 📈 Enhanced Visualizations
- **Narrative Spread Analysis**: Track AI vs. human narrative adoption
- **GAN Performance Dashboard**: Monitor generation quality and frequency
- **Sentiment Dynamics**: Compare emotional impact of different narrative types
- **Network Evolution**: Visualize how AI narratives spread through agent networks

## 📁 File Structure

```
├── app.py                          # Enhanced Streamlit dashboard
├── simulation/
│   ├── __init__.py
│   ├── agents.py                   # Agent behavior models
│   ├── model.py                    # Original simulation model
│   ├── enhanced_model.py           # GAN-enhanced simulation model
│   └── gan_generator.py            # GAN implementation
├── processing/
│   ├── __init__.py
│   └── narrative_processor.py     # Text processing utilities
├── data/                          # Narrative datasets
│   ├── psyops_narratives.csv
│   ├── climate_narratives.csv
│   ├── economic_narratives.csv
│   ├── election_narratives.csv
│   ├── health_narratives.csv
│   └── tech_narratives.csv
├── models/                        # Saved GAN models
│   └── narrative_gan_model.pkl    # Pre-trained GAN (generated)
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 🔬 Technical Implementation

### GAN Architecture

#### Generator Network
- **Input**: 100-dimensional noise vector
- **Architecture**: LSTM-based sequence generation
- **Output**: Realistic narrative text sequences
- **Training**: Adversarial loss with discriminator feedback

#### Discriminator Network
- **Input**: Text sequences (real or generated)
- **Architecture**: Bidirectional LSTM + classifier
- **Output**: Probability of text being real
- **Function**: Distinguishes human vs. AI-generated narratives

### Agent-Based Model Enhancements

#### Enhanced Agent Types
- **Influencers**: High spreading capability, affected by GAN narratives
- **Regular Users**: Standard behavior, susceptible to all narrative types
- **Skeptics**: Resistant to narratives, but can be influenced by sophisticated GAN content

#### Dynamic Events
- **Narrative Boost**: Increase belief in existing narratives
- **Narrative Debunk**: Decrease belief through counter-evidence
- **GAN Injection**: Introduce AI-generated narratives
- **Adaptive Events**: Context-aware event generation

## 📊 Data Analysis Features

### Real-time Metrics
- **Believer Tracking**: Monitor narrative adoption over time
- **Sentiment Evolution**: Track emotional tone changes
- **Network Dynamics**: Visualize information flow patterns
- **GAN Performance**: Analyze AI generation effectiveness

### Export Capabilities
- **CSV Export**: Download complete simulation results
- **Model Saving**: Preserve trained GAN models
- **Network Data**: Export agent connection graphs
- **Event Logs**: Detailed simulation event history

## 🔧 Configuration Options

### GAN Parameters
```python
# Adjustable in gan_generator.py
vocab_size = 5000        # Vocabulary size
embedding_dim = 128      # Word embedding dimensions
hidden_dim = 256         # LSTM hidden dimensions
max_length = 20          # Maximum narrative length
epochs = 100             # Training epochs
batch_size = 32          # Training batch size
learning_rate = 0.0002   # Adam optimizer learning rate
```

### Simulation Parameters
```python
# Configurable in enhanced_model.py
num_agents = 100                    # Population size
enable_counter_narratives = True    # Counter-narrative generation
enable_gan = True                   # GAN functionality
gan_injection_rate = 10             # Generate narratives every N steps
```

## 🚨 System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.5GHz
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Python**: 3.8+

### Recommended for Optimal Performance
- **CPU**: Quad-core 3.0GHz or higher
- **RAM**: 8GB or more
- **GPU**: CUDA-compatible (for faster training)
- **Storage**: 5GB free space

## 🛡 Offline Operation

The system is designed for **complete offline operation**:
- ✅ No external API calls
- ✅ All models trained locally
- ✅ Self-contained narrative generation
- ✅ Local data processing only
- ✅ Secure environment suitable for defense applications

## 🔍 Troubleshooting

### Common Issues

1. **GAN Training Slow**
   - Solution: Reduce epochs or batch size in configuration
   - Alternative: Use CPU-only mode if GPU issues occur

2. **Memory Issues**
   - Solution: Reduce number of agents or vocabulary size
   - Alternative: Run shorter simulations

3. **Model Loading Errors**
   - Solution: Delete models folder and retrain
   - Alternative: Disable pre-trained model loading

4. **Visualization Performance**
   - Solution: Reduce number of simulation steps
   - Alternative: Use smaller agent networks

### Debug Mode
Enable debug output by modifying the logging level in the application code.

## 📝 Research Applications

### DRDO-ISSA Lab Use Cases
- **Information Warfare Simulation**: Model adversarial narrative campaigns
- **Counter-Narrative Strategy**: Test defensive information operations
- **Social Media Analysis**: Understand narrative spread patterns
- **Psychological Operations**: Study influence operation effectiveness
- **Defense Planning**: Evaluate information security vulnerabilities

### Academic Research
- **Computational Social Science**: Study information diffusion
- **AI Safety Research**: Analyze GAN-generated content impact
- **Network Science**: Investigate complex system behaviors
- **Sentiment Analysis**: Track emotional contagion patterns

## 🤝 Contributing

1. Fork the repository
2. Create feature branches
3. Test thoroughly with different scenarios
4. Document changes comprehensively
5. Submit pull requests with clear descriptions

## 📄 License

This project is developed for DRDO-ISSA Lab research purposes.

## 🆘 Support

For technical support or research collaboration:
- Check troubleshooting section
- Review code documentation
- Consult with DRDO-ISSA Lab team

---

**🎯 Enhanced for Defense Research | 🤖 Powered by GANs | 📊 Real-time Analytics**