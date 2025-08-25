import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import numpy as np
import os
from simulation.model import NarrativeModel
from simulation.enhanced_model import EnhancedNarrativeModel
from processing.narrative_processor import process_narratives, load_narrative_data, get_available_scenarios

def run_dashboard():
    st.title("🎯 Advanced Narrative Spread Simulation with GANs")
    st.markdown("*Enhanced simulation system with GAN-powered narrative generation for DRDO-ISSA Lab*")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("🔧 Model Configuration")
        use_gan = st.checkbox(
            "🤖 Enable GAN Mode",
            value=True,
            help="Use Generative Adversarial Networks for dynamic narrative generation"
        )
        
        if use_gan:
            st.success("✅ GAN Mode: AI will generate realistic narratives")
            
            # GAN model options
            st.subheader("GAN Settings")
            use_pretrained = st.checkbox(
                "Load Pre-trained Model",
                value=False,
                help="Load existing GAN model if available"
            )
            
            if use_pretrained:
                model_path = st.text_input("Model Path", value="models/narrative_gan_model.pkl")
            else:
                model_path = None
                
        else:
            st.info("📊 Classic Mode: Using original simulation")
            model_path = None
    
    # Data source selection
    st.header("📋 Data Configuration")
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
    st.header("⚙️ Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_agents = st.slider("Number of agents", 10, 1000, 100)
    with col2:
        steps = st.slider("Simulation steps", 1, 100, 30)
    with col3:
        if use_gan:
            gan_injection_rate = st.slider("GAN Generation Rate", 1, 20, 10, help="Generate new narratives every N steps")
    
    # Advanced options
    st.header("🔄 Advanced Options")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_counter_narratives = st.checkbox(
            "Enable Counter-Narratives",
            value=True,
            help="Generate opposing narratives during simulation"
        )
    
    with col2:
        if use_gan:
            adaptive_injection = st.checkbox(
                "Adaptive GAN Injection",
                value=True,
                help="Automatically inject narratives based on simulation state"
            )
    
    # Information about the simulation mode
    if use_gan:
        st.info("🤖 **GAN Mode Active**: The system will dynamically generate new narratives using AI, creating more realistic information warfare scenarios with emergent narrative patterns.")
        if enable_counter_narratives:
            st.info("🔄 **Smart Counter-Narratives**: AI will generate contextually appropriate opposing narratives instead of simple negations.")
    else:
        if enable_counter_narratives:
            st.info("🔄 **Basic Counter-Narratives**: Simple rule-based opposing narratives will be generated.")
    
    # Run simulation button
    if st.button("🚀 Run Advanced Simulation", type="primary", use_container_width=True):
        # Show simulation info
        st.header("🎯 Simulation Results")
        
        # Model selection
        if use_gan:
            model_class = EnhancedNarrativeModel
            st.info("🤖 Running Enhanced Simulation with GAN Integration...")
        else:
            model_class = NarrativeModel
            st.info("📊 Running Classic Simulation...")
        
        with st.spinner("Initializing simulation model..."):
            try:
                if use_gan:
                    model = model_class(
                        num_agents, 
                        narratives, 
                        enable_counter_narratives=enable_counter_narratives,
                        enable_gan=True,
                        gan_model_path=model_path if use_pretrained else None
                    )
                else:
                    model = model_class(num_agents, narratives, enable_counter_narratives=enable_counter_narratives)
                
                # Progress bar for simulation steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for step in range(steps):
                    model.step()
                    progress = (step + 1) / steps
                    progress_bar.progress(progress)
                    
                    if use_gan and hasattr(model, 'get_gan_statistics'):
                        gan_stats = model.get_gan_statistics()
                        status_text.text(f"Step {step + 1}/{steps} | GAN Narratives: {gan_stats['gan_generated']}")
                    else:
                        status_text.text(f"Step {step + 1}/{steps}")
                
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Error during simulation: {str(e)}")
                return
        
        # Get simulation data
        df = model.get_data_frame()
        believer_columns = [col for col in df.columns if 'narrative_' in col and '_believers' in col]
        
        # Create enhanced tabs for GAN mode
        if use_gan and hasattr(model, 'get_gan_statistics'):
            tabs = st.tabs(["📈 Narrative Spread", "🤖 GAN Analytics", "😊 Sentiment Analysis", "🕸️ Network Visualization", "📊 Event Timeline"])
        else:
            tabs = st.tabs(["📈 Narrative Spread", "😊 Sentiment Analysis", "🕸️ Network Visualization"])
        
        # Tab 1: Narrative Spread
        with tabs[0]:
            st.subheader("Narrative Believers Over Time")
            if believer_columns:
                fig_believers = px.line(
                    df, x='step', y=believer_columns, 
                    title='How Many People Believe Each Narrative Over Time',
                    labels={'value': 'Number of Believers', 'step': 'Simulation Step'}
                )
                fig_believers.update_layout(
                    hovermode='x unified',
                    legend=dict(title="Narratives"),
                    height=500
                )
                st.plotly_chart(fig_believers, use_container_width=True)
                
                # Enhanced final statistics
                if len(df) > 0:
                    final_row = df.iloc[-1]
                    st.subheader("📊 Final Results")
                    
                    # Create narrative results for sorting
                    narrative_results = []
                    
                    for col in believer_columns:
                        narrative_id = col.split('_')[1]
                        narrative_id_int = int(narrative_id)
                        
                        # Get narrative information
                        if hasattr(model, 'narratives') and narrative_id_int in model.narratives:
                            narrative = model.narratives[narrative_id_int]
                            narrative_text = narrative['text']
                            is_gan = narrative.get('gan_generated', False)
                            is_counter = narrative.get('counter_to', None) is not None
                        else:
                            narrative_text = f"Narrative {narrative_id}"
                            is_gan = False
                            is_counter = False
                        
                        final_believers = final_row[col]
                        
                        narrative_results.append({
                            'id': narrative_id_int,
                            'text': narrative_text,
                            'believers': final_believers,
                            'is_gan': is_gan,
                            'is_counter': is_counter
                        })
                    
                    # Sort by believers
                    narrative_results.sort(key=lambda x: x['believers'], reverse=True)
                    
                    # Display with enhanced visualization
                    for result in narrative_results:
                        # Determine colors and icons
                        if result['is_gan'] and result['is_counter']:
                            icon = "🤖🔄"
                            type_label = "GAN Counter"
                            color = "#ff6b35"
                        elif result['is_gan']:
                            icon = "🤖"
                            type_label = "GAN Generated"
                            color = "#6c5ce7"
                        elif result['is_counter']:
                            icon = "🔄"
                            type_label = "Counter"
                            color = "#fd79a8"
                        else:
                            icon = "📢"
                            type_label = "Original"
                            color = "#0984e3"
                        
                        # Enhanced display
                        with st.container():
                            col1, col2, col3 = st.columns([4, 1, 1])
                            
                            with col1:
                                st.markdown(f"""
                                <div style="padding: 10px; border-left: 4px solid {color}; margin: 5px 0; background-color: rgba(128, 128, 128, 0.1); border-radius: 5px;">
                                    <div style="font-size: 14px; font-weight: bold; color: {color};">
                                        {icon} {type_label} #{result['id']}
                                    </div>
                                    <div style="font-size: 12px; margin: 5px 0; font-style: italic;">
                                        "{result['text'][:100]}{'...' if len(result['text']) > 100 else ''}"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Believers", result['believers'])
                            
                            with col3:
                                percentage = (result['believers'] / num_agents) * 100
                                st.metric("Reach", f"{percentage:.1f}%")
        
        # Tab 2: GAN Analytics (only if GAN enabled)
        if use_gan and hasattr(model, 'get_gan_statistics'):
            with tabs[1]:
                st.subheader("🤖 GAN Performance Analytics")
                
                gan_stats = model.get_gan_statistics()
                
                # Display GAN statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Narratives", gan_stats['total_narratives'])
                with col2:
                    st.metric("GAN Generated", gan_stats['gan_generated'])
                with col3:
                    st.metric("Original", gan_stats['original_narratives'])
                with col4:
                    generation_rate = (gan_stats['gan_generated'] / gan_stats['total_narratives'] * 100) if gan_stats['total_narratives'] > 0 else 0
                    st.metric("Generation Rate", f"{generation_rate:.1f}%")
                
                # Sentiment comparison
                if gan_stats['gan_generated'] > 0:
                    st.subheader("📊 Sentiment Analysis Comparison")
                    
                    sentiment_data = pd.DataFrame({
                        'Type': ['Original Narratives', 'GAN Generated'],
                        'Average Sentiment': [gan_stats['avg_original_sentiment'], gan_stats['avg_gan_sentiment']]
                    })
                    
                    fig_sentiment_comp = px.bar(
                        sentiment_data,
                        x='Type',
                        y='Average Sentiment',
                        title='Sentiment Comparison: Original vs GAN Generated',
                        color='Average Sentiment',
                        color_continuous_scale='RdYlBu'
                    )
                    st.plotly_chart(fig_sentiment_comp, use_container_width=True)
                
                # Export GAN model option
                st.subheader("💾 Model Management")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save GAN Model"):
                        if model.save_gan_model():
                            st.success("✅ GAN model saved successfully!")
                        else:
                            st.error("❌ Failed to save GAN model")
                
                with col2:
                    if st.button("📤 Export Generated Narratives"):
                        if hasattr(model, 'export_narratives'):
                            df_narratives = model.export_narratives()
                            csv = df_narratives.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name="generated_narratives.csv",
                                mime="text/csv"
                            )
        
        # Sentiment Analysis Tab
        sentiment_tab_index = 2 if use_gan and hasattr(model, 'get_gan_statistics') else 1
        with tabs[sentiment_tab_index]:
            st.subheader("Average Sentiment Over Time")
            if 'avg_sentiment' in df.columns:
                fig_sentiment = px.line(
                    df, x='step', y='avg_sentiment', 
                    title='How Public Mood Changes During Simulation',
                    labels={'avg_sentiment': 'Average Sentiment', 'step': 'Simulation Step'}
                )
                fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", 
                                      annotation_text="Neutral Sentiment")
                fig_sentiment.update_layout(hovermode='x', height=400)
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Sentiment statistics
                final_sentiment = df['avg_sentiment'].iloc[-1] if len(df) > 0 else 0
                sentiment_change = final_sentiment - df['avg_sentiment'].iloc[0] if len(df) > 1 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Sentiment", f"{final_sentiment:.3f}")
                with col2:
                    st.metric("Sentiment Change", f"{sentiment_change:.3f}", delta=f"{sentiment_change:.3f}")
                with col3:
                    sentiment_volatility = df['avg_sentiment'].std() if len(df) > 1 else 0
                    st.metric("Volatility", f"{sentiment_volatility:.3f}")
        
        # Network Visualization Tab
        network_tab_index = 3 if use_gan and hasattr(model, 'get_gan_statistics') else 2
        with tabs[network_tab_index]:
            st.subheader("Agent Network Visualization")
            if model.network_data:
                last_network = model.network_data[-1]
                G = nx.Graph()
                for node_id, node_type in last_network['nodes']:
                    G.add_node(node_id, type=node_type)
                G.add_edges_from(last_network['edges'])
                
                pos = nx.spring_layout(G, k=1, iterations=50)
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
                
                # Create enhanced network visualization
                fig_network = go.Figure()
                
                # Add edges
                fig_network.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.3)', width=0.5),
                    hoverinfo='none',
                    showlegend=False,
                    name='Connections'
                ))
                
                # Add nodes by type with enhanced colors
                colors = {
                    'Influencer': '#e74c3c',  # Red
                    'Regular': '#3498db',     # Blue  
                    'Skeptic': '#2ecc71'      # Green
                }
                
                for agent_type, color in colors.items():
                    type_indices = [i for i, t in enumerate(node_types) if t == agent_type]
                    if type_indices:
                        type_x = [node_x[i] for i in type_indices]
                        type_y = [node_y[i] for i in type_indices]
                        type_texts = [node_texts[i] for i in type_indices]
                        
                        fig_network.add_trace(go.Scatter(
                            x=type_x, y=type_y,
                            mode='markers',
                            hoverinfo='text',
                            hovertext=type_texts,
                            marker=dict(
                                size=10,
                                color=color,
                                line=dict(width=2, color='white'),
                                opacity=0.8
                            ),
                            name=f"{agent_type} ({len(type_indices)})",
                            showlegend=True
                        ))
                
                fig_network.update_layout(
                    title=f'Agent Network at Step {last_network["step"]} (Total: {len(node_ids)} agents)',
                    showlegend=True,
                    legend=dict(
                        itemsizing='constant',
                        title=dict(text='Agent Types'),
                        bgcolor="rgba(255,255,255,0.8)"
                    ),
                    hovermode='closest',
                    margin=dict(t=60, b=20, l=20, r=20),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    height=600
                )
                
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Enhanced network statistics
                st.subheader("🔍 Network Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Agents", len(node_ids))
                with col2:
                    st.metric("Total Connections", len(G.edges()))
                with col3:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
                with col4:
                    density = nx.density(G) if G.nodes() else 0
                    st.metric("Network Density", f"{density:.3f}")
                
                # Agent type distribution
                type_counts = pd.Series(node_types).value_counts()
                fig_types = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Agent Type Distribution",
                    color_discrete_map=colors
                )
                st.plotly_chart(fig_types, use_container_width=True)
        
        # Event Timeline Tab (only for GAN mode)
        if use_gan and hasattr(model, 'get_gan_statistics') and len(tabs) > 4:
            with tabs[4]:
                st.subheader("📊 Event Timeline")
                event_df = model.get_event_data()
                
                if not event_df.empty:
                    # Create timeline visualization
                    fig_timeline = px.scatter(
                        event_df,
                        x='step',
                        y='type',
                        size='affected',
                        color='type',
                        title='Simulation Events Over Time',
                        labels={'step': 'Simulation Step', 'type': 'Event Type', 'affected': 'Agents Affected'}
                    )
                    fig_timeline.update_layout(height=400)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Event statistics
                    st.subheader("Event Summary")
                    event_summary = event_df['type'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_event_counts = px.bar(
                            x=event_summary.index,
                            y=event_summary.values,
                            title="Event Type Frequency",
                            labels={'x': 'Event Type', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_event_counts, use_container_width=True)
                    
                    with col2:
                        st.write("**Event Details:**")
                        for _, event in event_df.iterrows():
                            event_text = event.get('text', 'N/A')
                            if len(event_text) > 50:
                                event_text = event_text[:50] + "..."
                            
                            st.write(f"**Step {event['step']}** - {event['type'].title()}: {event_text}")
                else:
                    st.info("No events recorded during simulation")

if __name__ == "__main__":
    run_dashboard()