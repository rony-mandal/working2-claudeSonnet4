import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from simulation.model import NarrativeModel
from processing.narrative_processor import process_narratives, load_narrative_data, get_available_scenarios

def run_dashboard():
    st.title("Narrative Spread Simulation")
    
    # Data source selection
    data_source = st.radio("Data Source", ["Manual Input", "Preloaded Data"])
    
    if data_source == "Manual Input":
        narrative_input = st.text_area("Enter narratives (one per line):")
        narrative_texts = [text.strip() for text in narrative_input.split('\n') if text.strip()]
        narratives = process_narratives(narrative_texts) if narrative_texts else {}
    else:
        # Scenario selection dropdown
        available_scenarios = get_available_scenarios()
        if not available_scenarios:
            st.error("No narrative scenario files found in data/ directory!")
            return
        
        selected_scenario = st.selectbox(
            "Select Narrative Scenario",
            options=list(available_scenarios.keys()),
            index=0,
            help="Choose a predefined narrative scenario to simulate"
        )
        
        narratives = load_narrative_data(selected_scenario)
        
        if narratives:
            st.info(f"Loaded **{selected_scenario}** scenario with {len(narratives)} narratives")
            
            # Show loaded narratives in an expander
            with st.expander("📋 View Loaded Narratives"):
                for nid, narrative in narratives.items():
                    sentiment_emoji = "😟" if narrative['sentiment'] < -0.3 else ("😐" if narrative['sentiment'] < 0.3 else "😊")
                    st.write(f"{sentiment_emoji} **Narrative {nid}:** {narrative['text']} *(sentiment: {narrative['sentiment']:.2f})*")
    
    if not narratives:
        st.warning("Please enter at least one narrative or ensure preloaded data exists.")
        return
    
    # Simulation parameters
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        num_agents = st.slider("Number of agents", 10, 1000, 100)
    with col2:
        steps = st.slider("Simulation steps", 1, 100, 20)
    
    # Counter-narrative toggle
    st.subheader("🔄 Advanced Options")
    enable_counter_narratives = st.checkbox(
        "Enable Counter-Narratives",
        value=True,
        help="When enabled, the simulation will automatically generate opposing narratives every 5 steps based on dominant narratives. This creates more realistic information warfare scenarios."
    )
    
    if enable_counter_narratives:
        st.info("ℹ️ Counter-narratives will be automatically generated during simulation to oppose dominant narratives.")
    else:
        st.info("ℹ️ Only original narratives will spread - no counter-narratives will be generated.")

    if st.button("🚀 Run Simulation", type="primary"):
        # Show simulation info
        st.subheader("🎯 Simulation Results")
        
        with st.spinner("Running narrative spread simulation..."):
            model = NarrativeModel(num_agents, narratives, enable_counter_narratives=enable_counter_narratives)
            
            # Progress bar for simulation steps
            progress_bar = st.progress(0)
            for step in range(steps):
                model.step()
                progress_bar.progress((step + 1) / steps)
            
            progress_bar.empty()
        
        # Get simulation data
        df = model.get_data_frame()
        believer_columns = [col for col in df.columns if 'narrative_' in col and '_believers' in col]
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📈 Narrative Spread", "😊 Sentiment Analysis", "🕸️ Network Visualization"])
        
        with tab1:
            st.subheader("Narrative Believers Over Time")
            if believer_columns:
                fig_believers = px.line(
                    df, x='step', y=believer_columns, 
                    title='How Many People Believe Each Narrative Over Time',
                    labels={'value': 'Number of Believers', 'step': 'Simulation Step'}
                )
                fig_believers.update_layout(
                    hovermode='x unified',
                    legend=dict(title="Narratives")
                )
                st.plotly_chart(fig_believers, use_container_width=True)
                
                # Show final statistics
                if len(df) > 0:
                    final_row = df.iloc[-1]
                    st.subheader("📊 Final Results")
                    
                    # Create a list to store narrative results for sorting
                    narrative_results = []
                    
                    for col in believer_columns:
                        narrative_id = col.split('_')[1]  # Extract narrative ID
                        narrative_id_int = int(narrative_id)
                        
                        # Get narrative text - check both original narratives and model narratives
                        if narrative_id_int in narratives:
                            narrative_text = narratives[narrative_id_int]['text']
                            is_counter = False
                        elif hasattr(model, 'narratives') and narrative_id_int in model.narratives:
                            narrative_text = model.narratives[narrative_id_int]['text']
                            is_counter = narrative_id_int not in narratives
                        else:
                            narrative_text = f"Generated narrative {narrative_id}"
                            is_counter = True
                        
                        final_believers = final_row[col]
                        
                        narrative_results.append({
                            'id': narrative_id_int,
                            'text': narrative_text,
                            'believers': final_believers,
                            'is_counter': is_counter
                        })
                    
                    # Sort by number of believers (descending) for better readability
                    narrative_results.sort(key=lambda x: x['believers'], reverse=True)
                    
                    # Display results vertically
                    for result in narrative_results:
                        # Determine colors and icons
                        if result['is_counter']:
                            icon = "🔄"
                            type_label = "Counter"
                            color = "#ff6b6b"  # Red for counter-narratives
                        else:
                            icon = "📢"
                            type_label = "Original"
                            color = "#4dabf7"  # Blue for original narratives
                        
                        # Create a more detailed display
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"""
                                <div style="padding: 10px; border-left: 4px solid {color}; margin: 5px 0; background-color: rgba(128, 128, 128, 0.1); border-radius: 5px;">
                                    <div style="font-size: 14px; font-weight: bold; color: {color};">
                                        {icon} {type_label} Narrative {result['id']}
                                    </div>
                                    <div style="font-size: 12px; margin: 5px 0; font-style: italic;">
                                        "{result['text']}"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.metric(
                                    label="Believers",
                                    value=result['believers'],
                                    delta=None
                                )
        
        with tab2:
            st.subheader("Average Sentiment Over Time")
            if 'avg_sentiment' in df.columns:
                fig_sentiment = px.line(
                    df, x='step', y='avg_sentiment', 
                    title='How Public Mood Changes During Simulation',
                    labels={'avg_sentiment': 'Average Sentiment', 'step': 'Simulation Step'}
                )
                fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", 
                                      annotation_text="Neutral Sentiment")
                fig_sentiment.update_layout(hovermode='x')
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with tab3:
            st.subheader("Agent Network Visualization")
            if model.network_data:
                last_network = model.network_data[-1]
                G = nx.Graph()
                for node_id, node_type in last_network['nodes']:
                    G.add_node(node_id, type=node_type)
                G.add_edges_from(last_network['edges'])
                
                pos = nx.spring_layout(G)
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_types = [G.nodes[node]['type'] for node in G.nodes()]
                node_ids = list(G.nodes())
                node_texts = [f"ID: {nid}<br>Type: {typ}" for nid, typ in zip(node_ids, node_types)]
                
                # Create network visualization
                fig_network = go.Figure()
                
                # Add edges
                fig_network.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(color='lightgray', width=0.5),
                    hoverinfo='none',
                    showlegend=False,
                    name='Connections'
                ))
                
                # Add nodes by type for proper legend
                colors = {'Influencer': '#ff4444', 'Regular': '#4444ff', 'Skeptic': '#44ff44'}
                for agent_type, color in colors.items():
                    type_indices = [i for i, t in enumerate(node_types) if t == agent_type]
                    if type_indices:
                        type_x = [node_x[i] for i in type_indices]
                        type_y = [node_y[i] for i in type_indices]
                        type_ids = [node_ids[i] for i in type_indices]
                        type_texts = [node_texts[i] for i in type_indices]
                        
                        fig_network.add_trace(go.Scatter(
                            x=type_x, y=type_y,
                            mode='markers',
                            hoverinfo='text',
                            hovertext=type_texts,
                            marker=dict(size=8, color=color, line=dict(width=1, color='white')),
                            name=f"{agent_type} ({len(type_indices)})",
                            showlegend=True
                        ))
                
                fig_network.update_layout(
                    title=f'Agent Network at Step {last_network["step"]} (Total: {len(node_ids)} agents)',
                    showlegend=True,
                    legend=dict(
                        itemsizing='constant',
                        title=dict(text='Agent Types')
                    ),
                    hovermode='closest',
                    margin=dict(t=60, b=20, l=20, r=20),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Network statistics
                st.subheader("🔍 Network Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Agents", len(node_ids))
                with col2:
                    st.metric("Total Connections", len(G.edges()))
                with col3:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
                with col4:
                    type_counts = pd.Series(node_types).value_counts()
                    most_common_type = type_counts.index[0] if len(type_counts) > 0 else "None"
                    st.metric("Dominant Type", most_common_type)

if __name__ == "__main__":
    run_dashboard()