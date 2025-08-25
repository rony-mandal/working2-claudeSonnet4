#!/usr/bin/env python3
"""
Setup script for Advanced Narrative Spread Simulation with GANs
DRDO-ISSA Lab Project
"""

import os
import sys
import subprocess
import argparse

def create_directories():
    """Create necessary directories for the project"""
    directories = ['models', 'data', 'logs', 'exports']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Successfully installed all requirements")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    print("🔥 Checking PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🚀 CUDA is available! Device count: {torch.cuda.device_count()}")
            print(f"   Primary GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CUDA not available, will use CPU for training")
    except ImportError:
        print("❌ PyTorch not found. Installing CPU version...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'])
            print("✅ PyTorch CPU version installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing PyTorch: {e}")
            return False
    return True

def verify_installation():
    """Verify that all components are properly installed"""
    print("🔍 Verifying installation...")
    
    required_packages = [
        'streamlit', 'mesa', 'torch', 'numpy', 'pandas', 
        'plotly', 'networkx', 'nltk', 'sentence_transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All packages verified successfully!")
    return True

def create_sample_config():
    """Create a sample configuration file"""
    config_content = '''# Advanced Narrative Simulation Configuration
# DRDO-ISSA Lab Project

# Simulation Parameters
DEFAULT_AGENTS = 100
DEFAULT_STEPS = 30
DEFAULT_SCENARIOS = ["War/Conflict", "Economic Crisis", "Health Emergency"]

# GAN Parameters
GAN_VOCAB_SIZE = 5000
GAN_EMBEDDING_DIM = 128
GAN_HIDDEN_DIM = 256
GAN_MAX_LENGTH = 20
GAN_EPOCHS = 50
GAN_BATCH_SIZE = 16

# Model Paths
GAN_MODEL_PATH = "models/narrative_gan_model.pkl"
EXPORT_PATH = "exports/"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/simulation.log"

# Performance
USE_GPU = True  # Set to False to force CPU usage
MAX_MEMORY_MB = 4096  # Maximum memory usage in MB
'''
    
    config_path = 'config.py'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"✅ Created configuration file: {config_path}")
    else:
        print(f"📝 Configuration file already exists: {config_path}")

def run_initial_test():
    """Run a basic test to ensure everything works"""
    print("🧪 Running initial test...")
    
    try:
        # Test basic imports
        import streamlit
        from simulation.agents import NarrativeAgent
        from processing.narrative_processor import process_narratives
        
        # Test basic functionality
        test_narratives = ["Test narrative one", "Test narrative two"]
        processed = process_narratives(test_narratives)
        
        if processed:
            print("✅ Basic functionality test passed")
            return True
        else:
            print("❌ Basic functionality test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup Advanced Narrative Simulation with GANs')
    parser.add_argument('--skip-test', action='store_true', help='Skip the initial functionality test')
    parser.add_argument('--cpu-only', action='store_true', help='Install CPU-only version of PyTorch')
    args = parser.parse_args()

    print("🚀 Advanced Narrative Spread Simulation Setup")
    print("=" * 50)
    print("DRDO-ISSA Lab Project")
    print("Enhanced with Generative Adversarial Networks")
    print("=" * 50)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Checking PyTorch", check_pytorch),
        ("Creating sample config", create_sample_config),
        ("Verifying installation", verify_installation),
    ]
    
    if not args.skip_test:
        steps.append(("Running initial test", run_initial_test))
    
    # Execute setup steps
    for step_name, step_function in steps:
        print(f"\n🔧 {step_name}...")
        if not step_function():
            print(f"\n❌ Setup failed at step: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run the application: streamlit run app.py")
    print("2. Open your browser and navigate to the provided URL")
    print("3. Enable GAN mode for advanced AI features")
    print("4. Choose a scenario and run your first simulation")
    print("\n📚 See README.md for detailed usage instructions")
    print("🔬 Perfect for DRDO-ISSA Lab research applications")

if __name__ == "__main__":
    main()