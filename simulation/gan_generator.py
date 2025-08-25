import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from typing import List, Dict, Tuple
import re

class NarrativeGAN:
    """
    GAN for generating realistic narratives based on training data
    Designed for offline operation without external API calls
    """
    
    def __init__(self, vocab_size=5000, embedding_dim=128, hidden_dim=256, max_length=20):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vocabulary mappings
        self.word_to_idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.vocab_built = False
        
        # Initialize networks
        self.generator = None
        self.discriminator = None
        self.trained = False
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts"""
        word_freq = {}
        
        # Tokenize and count words
        for text in texts:
            words = self._tokenize(text.lower())
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add most frequent words to vocabulary
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:self.vocab_size-4]:  # -4 for special tokens
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.word_to_idx)} words")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        words = self._tokenize(text.lower())
        sequence = [self.word_to_idx.get('<START>')]
        
        for word in words[:self.max_length-2]:  # -2 for START and END tokens
            sequence.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))
        
        sequence.append(self.word_to_idx.get('<END>'))
        
        # Pad sequence
        while len(sequence) < self.max_length:
            sequence.append(self.word_to_idx.get('<PAD>'))
        
        return sequence[:self.max_length]
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text"""
        words = []
        for idx in sequence:
            word = self.idx_to_word.get(idx, '<UNK>')
            if word in ['<START>', '<PAD>']:
                continue
            elif word == '<END>':
                break
            else:
                words.append(word)
        
        return ' '.join(words)
    
    def initialize_networks(self):
        """Initialize Generator and Discriminator networks"""
        
        class Generator(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, max_length):
                super(Generator, self).__init__()
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim
                self.max_length = max_length
                
                # Noise to hidden
                self.noise_to_hidden = nn.Linear(100, hidden_dim)
                
                # LSTM for sequence generation
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                
                # Embedding layer
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                
                # Output projection
                self.output_projection = nn.Linear(hidden_dim, vocab_size)
                
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, noise, max_length=None):
                if max_length is None:
                    max_length = self.max_length
                
                batch_size = noise.size(0)
                
                # Initialize hidden state from noise
                h_0 = self.noise_to_hidden(noise).unsqueeze(0)
                c_0 = torch.zeros_like(h_0)
                
                # Start with START token
                current_input = torch.LongTensor([[1]] * batch_size).to(noise.device)  # START token
                outputs = []
                
                hidden = (h_0, c_0)
                
                for _ in range(max_length):
                    embedded = self.embedding(current_input)
                    lstm_out, hidden = self.lstm(embedded, hidden)
                    lstm_out = self.dropout(lstm_out)
                    
                    output = self.output_projection(lstm_out.squeeze(1))
                    outputs.append(output)
                    
                    # Use output as next input (teacher forcing alternative)
                    current_input = output.argmax(dim=1).unsqueeze(1)
                
                return torch.stack(outputs, dim=1)
        
        class Discriminator(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, max_length):
                super(Discriminator, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, sequences):
                embedded = self.embedding(sequences)
                lstm_out, _ = self.lstm(embedded)
                # Use last output for classification
                last_output = lstm_out[:, -1, :]
                return self.classifier(last_output)
        
        self.generator = Generator(self.vocab_size, self.embedding_dim, self.hidden_dim, self.max_length).to(self.device)
        self.discriminator = Discriminator(self.vocab_size, self.embedding_dim, self.hidden_dim, self.max_length).to(self.device)
        
        print("GAN networks initialized successfully")
    
    def train(self, training_texts: List[str], epochs=100, batch_size=32, learning_rate=0.0002):
        """Train the GAN on narrative texts"""
        
        if not self.vocab_built:
            self.build_vocab(training_texts)
        
        if self.generator is None:
            self.initialize_networks()
        
        # Prepare training data
        sequences = []
        for text in training_texts:
            seq = self.text_to_sequence(text)
            sequences.append(seq)
        
        sequences = torch.LongTensor(sequences).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        print(f"Starting GAN training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_g_loss = 0
            total_d_loss = 0
            
            # Create batches
            num_batches = len(sequences) // batch_size
            
            for batch_idx in range(num_batches):
                # Get real batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                real_sequences = sequences[start_idx:end_idx]
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Real samples
                real_labels = torch.ones(real_sequences.size(0), 1).to(self.device)
                real_output = self.discriminator(real_sequences)
                d_real_loss = criterion(real_output, real_labels)
                
                # Fake samples
                noise = torch.randn(batch_size, 100).to(self.device)
                fake_sequences_logits = self.generator(noise)
                fake_sequences = fake_sequences_logits.argmax(dim=2)
                
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                fake_output = self.discriminator(fake_sequences.detach())
                d_fake_loss = criterion(fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                fake_output = self.discriminator(fake_sequences)
                g_loss = criterion(fake_output, real_labels)  # Generator wants discriminator to think fakes are real
                g_loss.backward()
                g_optimizer.step()
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_g_loss = total_g_loss / num_batches
                avg_d_loss = total_d_loss / num_batches
                print(f"Epoch [{epoch+1}/{epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
                
                # Generate sample
                sample_noise = torch.randn(1, 100).to(self.device)
                with torch.no_grad():
                    sample_output = self.generator(sample_noise)
                    sample_sequence = sample_output.argmax(dim=2)[0].cpu().numpy()
                    sample_text = self.sequence_to_text(sample_sequence)
                    print(f"Sample generated text: '{sample_text}'")
        
        self.trained = True
        print("GAN training completed!")
    
    def generate_narratives(self, num_narratives=5, temperature=1.0) -> List[str]:
        """Generate new narratives using trained GAN"""
        if not self.trained:
            print("Warning: GAN not trained yet. Returning random combinations.")
            return self._generate_fallback_narratives(num_narratives)
        
        generated_texts = []
        
        with torch.no_grad():
            for _ in range(num_narratives):
                noise = torch.randn(1, 100).to(self.device)
                output_logits = self.generator(noise)
                
                # Apply temperature
                output_logits = output_logits / temperature
                
                # Convert to sequence
                sequence = output_logits.argmax(dim=2)[0].cpu().numpy()
                text = self.sequence_to_text(sequence)
                
                if text.strip():  # Only add non-empty texts
                    generated_texts.append(text.strip())
        
        return generated_texts
    
    def generate_counter_narrative(self, original_narrative: str, sentiment_flip=True) -> str:
        """Generate counter-narrative for a given narrative"""
        if not self.trained:
            # Fallback: simple negation-based counter-narrative
            return self._create_simple_counter(original_narrative)
        
        # Use GAN to generate and then modify for opposition
        generated = self.generate_narratives(1)[0]
        
        if sentiment_flip:
            # Simple sentiment flipping logic
            positive_words = ['good', 'great', 'safe', 'secure', 'peaceful', 'stable', 'improving']
            negative_words = ['bad', 'terrible', 'unsafe', 'insecure', 'violent', 'unstable', 'worsening']
            
            words = generated.split()
            for i, word in enumerate(words):
                if word in positive_words:
                    words[i] = random.choice(negative_words)
                elif word in negative_words:
                    words[i] = random.choice(positive_words)
            
            generated = ' '.join(words)
        
        return generated
    
    def _create_simple_counter(self, original: str) -> str:
        """Create simple counter-narrative by negation"""
        # Simple rule-based counter-narrative generation
        counter_templates = [
            f"No, {original.lower()}",
            f"Actually, {original.lower().replace('is', 'is not')}",
            f"The truth is opposite: {original.lower()}",
            f"Reports show {original.lower()} is false"
        ]
        return random.choice(counter_templates)
    
    def _generate_fallback_narratives(self, num_narratives: int) -> List[str]:
        """Fallback narrative generation when GAN is not trained"""
        templates = [
            "The situation is {adjective}",
            "Reports indicate {event} is {status}",
            "Sources confirm {entity} has {action}",
            "Analysis shows {trend} is {direction}",
            "Officials state {policy} will {effect}"
        ]
        
        adjectives = ['critical', 'stable', 'improving', 'deteriorating', 'uncertain']
        events = ['the crisis', 'the incident', 'the situation', 'the development']
        statuses = ['confirmed', 'denied', 'under investigation', 'resolved']
        entities = ['the government', 'the organization', 'the system', 'the network']
        actions = ['responded', 'failed', 'succeeded', 'intervened']
        trends = ['the pattern', 'the movement', 'the change', 'the shift']
        directions = ['accelerating', 'slowing', 'reversing', 'continuing']
        policies = ['the new law', 'the regulation', 'the measure', 'the directive']
        effects = ['succeed', 'fail', 'cause problems', 'bring benefits']
        
        narratives = []
        for _ in range(num_narratives):
            template = random.choice(templates)
            narrative = template.format(
                adjective=random.choice(adjectives),
                event=random.choice(events),
                status=random.choice(statuses),
                entity=random.choice(entities),
                action=random.choice(actions),
                trend=random.choice(trends),
                direction=random.choice(directions),
                policy=random.choice(policies),
                effect=random.choice(effects)
            )
            narratives.append(narrative)
        
        return narratives
    
    def save_model(self, filepath: str):
        """Save trained GAN model"""
        if not self.trained:
            print("Warning: Saving untrained model")
        
        model_data = {
            'generator_state': self.generator.state_dict() if self.generator else None,
            'discriminator_state': self.discriminator.state_dict() if self.discriminator else None,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_length': self.max_length,
            'trained': self.trained,
            'vocab_built': self.vocab_built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained GAN model"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore parameters
        self.word_to_idx = model_data['word_to_idx']
        self.idx_to_word = model_data['idx_to_word']
        self.vocab_size = model_data['vocab_size']
        self.embedding_dim = model_data['embedding_dim']
        self.hidden_dim = model_data['hidden_dim']
        self.max_length = model_data['max_length']
        self.trained = model_data['trained']
        self.vocab_built = model_data['vocab_built']
        
        # Initialize and load networks
        if model_data['generator_state'] and model_data['discriminator_state']:
            self.initialize_networks()
            self.generator.load_state_dict(model_data['generator_state'])
            self.discriminator.load_state_dict(model_data['discriminator_state'])
        
        print(f"Model loaded from {filepath}")
        return True