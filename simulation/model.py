import mesa
from mesa import Model
import numpy as np
import pandas as pd
from .agents import NarrativeAgent

class NarrativeModel(Model):
    def __init__(self, num_agents, initial_narratives, enable_counter_narratives=True):
        super().__init__()
        self.num_agents = num_agents
        self.narratives = initial_narratives.copy()
        self.counter_narratives = {}
        self.enable_counter_narratives = enable_counter_narratives
        
        for i in range(num_agents):
            agent = NarrativeAgent(self)
        
        for agent in self.agents:
            if len(self.agents) >= 5:
                agent_list = list(self.agents)
                agent.connections = np.random.choice(agent_list, size=5, replace=False).tolist()
            else:
                agent.connections = [a for a in self.agents if a != agent]
        
        if initial_narratives and self.agents:
            first_narrative_id = list(initial_narratives.keys())[0]
            first_agent = list(self.agents)[0]
            first_agent.beliefs[first_narrative_id] = 1.0
        
        self.data = {'step': []}
        for nid in initial_narratives:
            self.data[f'narrative_{nid}_believers'] = []
        self.data['avg_sentiment'] = []
        self.network_data = []
        self._step_count = 0

    def events(self):
        if self._step_count % 7 == 0 and self.narratives:  # Events every 7 steps
            event_type = np.random.choice(["boost", "debunk"], p=[0.6, 0.4])
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
            
            # Log event for potential visualization
            self.data.setdefault('event_log', []).append({
                'step': self._step_count,
                'type': event_type,
                'narrative': target_nid,
                'affected': len(affected_agents)
            })

    def step(self):
        self.agents.do("step")
        self._step_count += 1
        
        # Generate counter-narratives every 5 steps ONLY if enabled
        if (self.enable_counter_narratives and 
            self._step_count % 5 == 0 and 
            self.narratives):
            
            dominant_nid = max(self.narratives, key=lambda x: sum(1 for a in self.agents if x in a.beliefs and a.beliefs[x] > 0.5))
            counter_text = f"No, {self.narratives[dominant_nid]['text'].lower().replace('is', 'is not')}"
            if counter_text not in [n['text'] for n in self.narratives.values()]:
                counter_nid = max(self.narratives.keys()) + 1
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
        
        # Trigger external events
        self.events()
        
        # Collect data for this step
        step_data = {'step': self._step_count}
        
        # Count believers for each narrative - FIXED TYPO HERE
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
                # If this is a new key (new counter-narrative), pad with zeros and append
                self.data[key] = [0] * (self._step_count - 1) + [value]
        
        # Collect network data
        self.network_data.append({
            'step': self._step_count,
            'nodes': [(a.unique_id, a.type) for a in self.agents],
            'edges': [(a.unique_id, n.unique_id) for a in self.agents for n in a.connections]
        })

    def get_data_frame(self):
        # Ensure all arrays have the same length before creating DataFrame
        if not self.data['step']:  # If no steps recorded yet
            return pd.DataFrame()
        
        # Create a copy of data excluding non-numeric fields like event_log
        df_data = {}
        max_length = len(self.data['step'])
        
        for key in self.data:
            if key == 'event_log':  # Skip event_log as it's a list of dicts
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
        return pd.DataFrame()  # Return empty DataFrame if no events