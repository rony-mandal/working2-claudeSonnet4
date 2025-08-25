import mesa
from mesa import Model
import numpy as np
import pandas as pd
from .agents import NarrativeAgent
from .gan_generator import NarrativeGAN
import os
import pickle

class EnhancedNarrativeModel(Model):
    def __init__(self, num_agents, initial_narratives, enable_counter_narratives=True, 
                 enable_gan=True, gan_model_path=None):
        super().__init__()
        self.num_agents = num_agents
        self.narratives = initial_narratives.copy()
        self.counter_narratives = {}
        self.enable_counter_narratives = enable_counter_narratives
        self.enable_gan = enable_gan
        
        # Initialize GAN
        self.gan = None
        if enable_gan:
            self.gan = NarrativeGAN()
            self._initialize_or_load_gan(gan_model_path)
        
        # Create agents
        for i in range(num_agents):
            agent = NarrativeAgent(self)
        
        # Create connections between agents
        for agent in self.agents:
            if len(self.agents) >= 5:
                agent_list = list(self.agents)
                agent.connections = np.random.choice(agent_list, size=5, replace=False).tolist()
            else:
                agent.connections = [a for a in self.agents if a != agent]
        
        # Seed initial narrative
        if initial_narratives and self.agents:
            first_narrative_id = list(initial_narratives.keys())[0]
            first_agent = list(self.agents)[0]
            first_agent.beliefs[first_narrative_id] = 1.0
        
        # Data collection
        self.data = {'step': []}
        for nid in initial_narratives:
            self.data[f'narrative_{nid}_believers'] = []
        self.data['avg_sentiment'] = []
        self.network_data = []
        self._step_count = 0
        
        # GAN-generated narrative tracking
        self.gan_generated_narratives = {}
        self.narrative_id_counter = max(initial_narratives.keys()) if initial_narratives else 0
    
    def _initialize_or_load_gan(self, gan_model_path=None):
        """Initialize or load pre-trained GAN model"""
        if gan_model_path and os.path.exists(gan_model_path):
            print(f"Loading pre-trained GAN from {gan_model_path}")
            self.gan.load_model(gan_model_path)
        else:
            print("Training GAN on existing narratives...")
            if self.narratives:
                # Extract text from narratives for training
                training_texts = [narrative['text'] for narrative in self.narratives.values()]
                
                # Add some additional training data for better performance
                additional_training_data = self._get_expanded_training_data()
                training_texts.extend(additional_training_data)
                
                # Train GAN with reduced epochs for faster initialization
                self.gan.train(training_texts, epochs=50, batch_size=16)
                
                # Save the trained model
                model_dir = 'models'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                self.gan.save_model(os.path.join(model_dir, 'narrative_gan_model.pkl'))
            else:
                print("Warning: No initial narratives provided for GAN training")
    
    def _get_expanded_training_data(self):
        """Get additional training data to improve GAN performance"""
        # Enhanced training corpus based on common narrative patterns
        additional_data = [
            "The crisis is escalating rapidly",
            "Authorities have lost control of the situation",
            "Reports indicate widespread panic among civilians",
            "Emergency services are overwhelmed by demands",
            "The government is hiding critical information",
            "Foreign agents are spreading disinformation",
            "Social media platforms are censoring the truth",
            "Economic markets are showing signs of collapse",
            "Supply chains have been severely disrupted",
            "Public health officials contradict earlier statements",
            "Law enforcement struggles to maintain order",
            "International observers express deep concern",
            "The situation is stabilizing according to sources",
            "Recovery efforts are showing promising results",
            "Community leaders call for calm and unity",
            "Fact-checkers have verified the information",
            "Scientific evidence supports official claims",
            "Transparency measures are being implemented",
            "International cooperation is strengthening response",
            "Local businesses adapt to new circumstances",
            "Educational institutions continue normal operations",
            "Healthcare workers report manageable conditions",
            "Transportation systems function without major issues",
            "Communication networks remain fully operational",
            "Relief efforts reach all affected areas successfully"
        ]
        return additional_data
    
    def generate_new_narratives(self, num_narratives=3):
        """Generate new narratives using GAN"""
        if not self.gan or not self.gan.trained:
            print("GAN not available or trained, skipping narrative generation")
            return []
        
        try:
            new_texts = self.gan.generate_narratives(num_narratives)
            new_narratives = []
            
            for text in new_texts:
                if text and len(text.split()) > 2:  # Basic quality filter
                    self.narrative_id_counter += 1
                    nid = self.narrative_id_counter
                    
                    # Create narrative with GAN-generated embedding and sentiment
                    narrative = {
                        'text': text,
                        'embedding': np.random.randn(384),  # Placeholder embedding
                        'sentiment': self._estimate_sentiment(text),
                        'gan_generated': True
                    }
                    
                    self.gan_generated_narratives[nid] = narrative
                    self.narratives[nid] = narrative
                    new_narratives.append((nid, narrative))
                    
                    # Initialize data tracking for new narrative
                    self.data[f'narrative_{nid}_believers'] = [0] * self._step_count
            
            return new_narratives
        
        except Exception as e:
            print(f"Error generating narratives with GAN: {e}")
            return []
    
    def _estimate_sentiment(self, text):
        """Simple rule-based sentiment estimation"""
        positive_words = ['good', 'great', 'safe', 'secure', 'peaceful', 'stable', 'improving', 
                         'successful', 'positive', 'recovery', 'progress', 'cooperation']
        negative_words = ['bad', 'terrible', 'dangerous', 'unsafe', 'violent', 'unstable', 
                         'failing', 'crisis', 'panic', 'collapse', 'threat', 'concern']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return 0.3 + (positive_count - negative_count) * 0.1
        elif negative_count > positive_count:
            return -0.3 - (negative_count - positive_count) * 0.1
        else:
            return 0.0
    
    def generate_gan_counter_narrative(self, target_narrative_id):
        """Generate counter-narrative using GAN"""
        if not self.gan or target_narrative_id not in self.narratives:
            return None
        
        try:
            original_text = self.narratives[target_narrative_id]['text']
            counter_text = self.gan.generate_counter_narrative(original_text)
            
            if counter_text and len(counter_text.split()) > 2:
                self.narrative_id_counter += 1
                counter_nid = self.narrative_id_counter
                
                counter_narrative = {
                    'text': counter_text,
                    'embedding': np.random.randn(384),  # Placeholder embedding
                    'sentiment': -self.narratives[target_narrative_id]['sentiment'],
                    'gan_generated': True,
                    'counter_to': target_narrative_id
                }
                
                self.counter_narratives[counter_nid] = counter_narrative
                self.narratives[counter_nid] = counter_narrative
                
                # Initialize data tracking
                self.data[f'narrative_{counter_nid}_believers'] = [0] * self._step_count
                
                return counter_nid, counter_narrative
        
        except Exception as e:
            print(f"Error generating GAN counter-narrative: {e}")
        
        return None
    
    def adaptive_narrative_injection(self):
        """Inject new narratives based on simulation state"""
        if not self.gan or self._step_count % 10 != 0:  # Every 10 steps
            return
        
        # Analyze current narrative dominance
        dominant_narratives = []
        for nid in self.narratives:
            believers = sum(1 for agent in self.agents if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            if believers > len(self.agents) * 0.3:  # If more than 30% believe
                dominant_narratives.append(nid)
        
        # Generate new narratives to maintain diversity
        if len(dominant_narratives) < 2:
            new_narratives = self.generate_new_narratives(2)
            if new_narratives:
                # Seed new narratives in random agents
                for nid, narrative in new_narratives:
                    if self.agents:
                        random_agent = np.random.choice(list(self.agents))
                        random_agent.beliefs[nid] = 0.7
                        print(f"Injected GAN-generated narrative {nid}: '{narrative['text']}'")
    
    def events(self):
        """Enhanced events with GAN-generated narratives"""
        if self._step_count % 7 == 0 and self.narratives:  # Events every 7 steps
            event_type = np.random.choice(["boost", "debunk", "new_narrative"], p=[0.4, 0.3, 0.3])
            
            if event_type == "new_narrative" and self.gan and self.gan.trained:
                # Generate and inject new narrative event
                new_narratives = self.generate_new_narratives(1)
                if new_narratives:
                    nid, narrative = new_narratives[0]
                    affected_agents = np.random.choice(list(self.agents), size=min(15, len(self.agents)), replace=False)
                    
                    for agent in affected_agents:
                        agent.beliefs[nid] = 0.6
                    
                    # Log event
                    self.data.setdefault('event_log', []).append({
                        'step': self._step_count,
                        'type': 'gan_narrative_injection',
                        'narrative': nid,
                        'text': narrative['text'][:50] + '...',
                        'affected': len(affected_agents)
                    })
            
            else:
                # Original event logic
                target_nid = np.random.choice(list(self.narratives.keys()))
                affected_agents = np.random.choice(list(self.agents), size=min(20, len(self.agents)), replace=False)
                
                if event_type == "boost":
                    for agent in affected_agents:
                        if target_nid not in agent.beliefs:
                            agent.beliefs[target_nid] = 0.0
                        agent.beliefs[target_nid] = min(1.0, agent.beliefs[target_nid] + 0.3)
                elif event_type == "debunk":
                    for agent in affected_agents:
                        if target_nid in agent.beliefs:
                            agent.beliefs[target_nid] = max(0.0, agent.beliefs[target_nid] - 0.4)
                
                # Log event
                self.data.setdefault('event_log', []).append({
                    'step': self._step_count,
                    'type': event_type,
                    'narrative': target_nid,
                    'affected': len(affected_agents)
                })
    
    def step(self):
        """Enhanced step function with GAN integration"""
        self.agents.do("step")
        self._step_count += 1
        
        # Generate counter-narratives (enhanced with GAN)
        if (self.enable_counter_narratives and 
            self._step_count % 5 == 0 and 
            self.narratives):
            
            dominant_nid = max(self.narratives, key=lambda x: sum(1 for a in self.agents if x in a.beliefs and a.beliefs[x] > 0.5))
            
            # Use GAN for counter-narrative if available
            if self.gan and self.gan.trained and np.random.random() < 0.7:  # 70% chance to use GAN
                result = self.generate_gan_counter_narrative(dominant_nid)
                if result:
                    counter_nid, counter_narrative = result
                    # Seed counter-narrative in first agent
                    agents_list = list(self.agents)
                    if agents_list:
                        agents_list[0].beliefs[counter_nid] = 1.0
                        print(f"Generated GAN counter-narrative {counter_nid}: '{counter_narrative['text']}'")
            else:
                # Fallback to original counter-narrative logic
                counter_text = f"No, {self.narratives[dominant_nid]['text'].lower().replace('is', 'is not')}"
                if counter_text not in [n['text'] for n in self.narratives.values()]:
                    self.narrative_id_counter += 1
                    counter_nid = self.narrative_id_counter
                    self.counter_narratives[counter_nid] = {
                        'text': counter_text,
                        'embedding': self.narratives[dominant_nid]['embedding'],
                        'sentiment': -self.narratives[dominant_nid]['sentiment']
                    }
                    self.narratives[counter_nid] = self.counter_narratives[counter_nid]
                    
                    # Initialize data for new counter-narrative with zeros for previous steps
                    self.data[f'narrative_{counter_nid}_believers'] = [0] * (self._step_count - 1)
                    
                    # Seed counter-narrative in first agent
                    agents_list = list(self.agents)
                    if agents_list:
                        agents_list[0].beliefs[counter_nid] = 1.0
        
        # Adaptive narrative injection
        self.adaptive_narrative_injection()
        
        # Trigger external events
        self.events()
        
        # Collect data for this step
        step_data = {'step': self._step_count}
        
        # Count believers for each narrative
        for nid in {**self.narratives, **self.counter_narratives}:
            believers = sum(1 for agent in self.agents if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            step_data[f'narrative_{nid}_believers'] = believers
        
        if self.agents:
            step_data['avg_sentiment'] = np.mean([agent.sentiment for agent in self.agents])
        else:
            step_data['avg_sentiment'] = 0.0
        
        # Append data ensuring consistency
        for key, value in step_data.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                # If this is a new key (new narrative), pad with zeros and append
                self.data[key] = [0] * (self._step_count - 1) + [value]
        
        # Collect network data
        self.network_data.append({
            'step': self._step_count,
            'nodes': [(a.unique_id, a.type) for a in self.agents],
            'edges': [(a.unique_id, n.unique_id) for a in self.agents for n in a.connections]
        })

    def get_data_frame(self):
        """Enhanced data frame with GAN narrative tracking"""
        if not self.data['step']:
            return pd.DataFrame()
        
        # Create a copy of data excluding non-numeric fields like event_log
        df_data = {}
        max_length = len(self.data['step'])
        
        for key in self.data:
            if key == 'event_log':
                continue
            
            current_length = len(self.data[key])
            if current_length < max_length:
                # Pad with zeros for missing steps
                padded_data = self.data[key] + [0] * (max_length - current_length)
                df_data[key] = padded_data
            else:
                df_data[key] = self.data[key]
        
        return pd.DataFrame(df_data)

    def get_network_data(self):
        return self.network_data
    
    def get_event_data(self):
        """Get event log data as a separate DataFrame"""
        if 'event_log' in self.data and self.data['event_log']:
            return pd.DataFrame(self.data['event_log'])
        return pd.DataFrame()
    
    def get_gan_statistics(self):
        """Get statistics about GAN-generated narratives"""
        gan_narratives = {nid: narrative for nid, narrative in self.narratives.items() 
                         if narrative.get('gan_generated', False)}
        
        stats = {
            'total_narratives': len(self.narratives),
            'gan_generated': len(gan_narratives),
            'original_narratives': len(self.narratives) - len(gan_narratives),
            'gan_enabled': self.enable_gan and self.gan is not None,
            'gan_trained': self.gan.trained if self.gan else False
        }
        
        # Calculate average sentiment for GAN vs original
        if gan_narratives:
            gan_sentiments = [n['sentiment'] for n in gan_narratives.values()]
            stats['avg_gan_sentiment'] = np.mean(gan_sentiments)
        else:
            stats['avg_gan_sentiment'] = 0.0
        
        original_narratives = {nid: narrative for nid, narrative in self.narratives.items() 
                             if not narrative.get('gan_generated', False)}
        if original_narratives:
            original_sentiments = [n['sentiment'] for n in original_narratives.values()]
            stats['avg_original_sentiment'] = np.mean(original_sentiments)
        else:
            stats['avg_original_sentiment'] = 0.0
        
        return stats
    
    def save_gan_model(self, filepath=None):
        """Save the trained GAN model"""
        if not self.gan:
            print("No GAN model to save")
            return False
        
        if filepath is None:
            model_dir = 'models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            filepath = os.path.join(model_dir, 'narrative_gan_model.pkl')
        
        self.gan.save_model(filepath)
        return True
    
    def export_narratives(self, filepath='generated_narratives.csv'):
        """Export all narratives including GAN-generated ones"""
        narrative_data = []
        
        for nid, narrative in self.narratives.items():
            narrative_data.append({
                'id': nid,
                'text': narrative['text'],
                'sentiment': narrative['sentiment'],
                'gan_generated': narrative.get('gan_generated', False),
                'counter_to': narrative.get('counter_to', None)
            })
        
        df = pd.DataFrame(narrative_data)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(narrative_data)} narratives to {filepath}")
        return df