import streamlit as st
import cv2
import tempfile
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from pathlib import Path
import time
from datetime import datetime, timedelta
import base64
from moviepy.editor import VideoFileClip
import os
from dotenv import load_dotenv
import plotly.express as px
from io import BytesIO

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="FINAS Demo x BlacX",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #2e6fdf;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #2e6fdf;
    }
    .video-timestamp {
        font-family: monospace;
        color: #666;
    }
    .video-player-container {
        position: relative;
        width: 100%;
        margin-bottom: 1rem;
    }
    .metadata-timeline {
        background: white;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .timestamp-marker {
        cursor: pointer;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
    .timestamp-marker:hover {
        background-color: #f0f7ff;
    }
    .timestamp-marker.active {
        background-color: #e3f2fd;
        border-left: 4px solid #2e6fdf;
    }
    </style>
""", unsafe_allow_html=True)

class VideoProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def extract_frames(self, video_path, sample_rate=1):
        """Extract frames from video at given sample rate (seconds)"""
        frames = []
        timestamps = []
        frame_images = []
        frame_numbers = []
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_rate)  # Calculate frame interval based on FPS
        duration = total_frames / fps if fps > 0 else 0
        
        video_metadata = {
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames,
            'frame_interval': frame_interval
        }
        
        if fps <= 0:
            print("Warning: Could not determine video FPS. Using default frame counting.")
            fps = 30  # Default fallback FPS
        
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                # Calculate accurate timestamp based on frame number and FPS
                timestamp = current_frame / fps
                timestamp_str = str(timedelta(seconds=int(timestamp)))
                frame_text = f"Frame: {current_frame} | Time: {timestamp_str}"
                
                # Add text overlay to frame
                frame = cv2.putText(
                    frame,
                    frame_text,
                    (10, 30),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    1,  # Font scale
                    (255, 255, 255),  # Color (white)
                    2,  # Thickness
                    cv2.LINE_AA  # Line type
                )
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_images.append(frame_rgb)
                
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frames.append(base64_frame)
                timestamps.append(timestamp)
                frame_numbers.append(current_frame)
                
            current_frame += 1
            
        cap.release()
        return frames, timestamps, frame_images, video_metadata, frame_numbers

    def analyze_frame(self, frame_base64, timestamp):
        """Analyze a single frame using OpenAI Vision API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe this video frame at timestamp {timestamp:.2f} seconds. Focus on visible objects, actions, and scene details."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_base64}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing frame: {str(e)}"

    def extract_audio(self, video_path):
        """Extract and transcribe audio from video with timestamps"""
        try:
            video = VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix='.mp3')
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"]
                )
            
            os.unlink(audio_path)
            return transcript
        except Exception as e:
            return f"Error processing audio: {str(e)}"

class MetadataManager:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.metadata_df = None
        self.video_metadata = None
        
    def create_metadata_df(self, frame_metadata, timestamps, audio_data, video_metadata, frame_numbers):
        """Create a DataFrame with synchronized metadata"""
        self.video_metadata = video_metadata
        audio_segments = []
        
        if isinstance(audio_data, str):
            audio_segments = [audio_data] * len(timestamps)
        else:
            for timestamp in timestamps:
                relevant_text = self.get_audio_segment_for_timestamp(audio_data, timestamp)
                audio_segments.append(relevant_text)
        
        formatted_timestamps = [str(timedelta(seconds=int(t))) for t in timestamps]
        
        self.metadata_df = pd.DataFrame({
            'frame_number': frame_numbers,
            'timestamp': timestamps,
            'formatted_time': formatted_timestamps,
            'frame_description': frame_metadata,
            'audio_transcript': audio_segments
        })
    
    def get_audio_segment_for_timestamp(self, audio_data, timestamp):
        """Extract relevant audio segment for given timestamp"""
        try:
            if isinstance(audio_data, str):
                return audio_data
            
            window_start = max(0, timestamp - 0.5)
            window_end = timestamp + 0.5
            
            if hasattr(audio_data, 'segments'):
                segments = audio_data.segments
            else:
                segments = audio_data.get('segments', [])
            
            relevant_words = []
            
            for segment in segments:
                segment_start = segment.start if hasattr(segment, 'start') else segment.get('start', 0)
                segment_end = segment.end if hasattr(segment, 'end') else segment.get('end', 0)
                
                if segment_start <= window_end and segment_end >= window_start:
                    text = segment.text if hasattr(segment, 'text') else segment.get('text', '')
                    if text:
                        relevant_words.append(text.strip())
            
            return ' '.join(relevant_words) if relevant_words else ""
            
        except Exception as e:
            print(f"Error processing audio segment: {str(e)}")
            return ""

    def query_metadata(self, query):
        """Query metadata using OpenAI"""
        if self.metadata_df is None:
            return "No metadata available. Please process a video first."
            
        try:
            # Convert metadata to a more readable format for the AI
            metadata_context = []
            for _, row in self.metadata_df.iterrows():
                metadata_context.append(
                    f"Time {format_timestamp(row['timestamp'])}:\n"
                    f"Visual: {row['frame_description']}\n"
                    f"Audio: {row['audio_transcript']}\n"
                )
            
            metadata_text = "\n".join(metadata_context)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant analyzing video content. Provide detailed and accurate answers about the video based on the provided frame descriptions and audio transcripts. Include relevant timestamps in your responses when appropriate."
                    },
                    {
                        "role": "user",
                        "content": f"Based on the following video metadata:\n\n{metadata_text}\n\nQuery: {query}"
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying metadata: {str(e)}"

    def analyze_frame(self, frame, timestamp):
        """Analyze a single frame using OpenAI's vision model"""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Create the messages for GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Updated from gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what you see in this video frame."},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    }
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing frame: {str(e)}"

def create_video_player(video_file, metadata_df):
    """Create an interactive video player with metadata timeline"""
    # Create two columns for video and timeline
    video_col, timeline_col = st.columns([1, 1])
    
    with video_col:
        # Create video player container with reduced width
        video_container = st.empty()
        # Display video with custom HTML/CSS to control size
        st.markdown("""
            <style>
            .video-container video {
                max-width: 100%;
                max-height: 300px;  /* Reduced height */
                width: auto;
                margin: 0 auto;
                display: block;
            }
            </style>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            video_container.video(video_file)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with timeline_col:
        # Add custom CSS for scrollable timeline
        st.markdown("""
            <style>
            .timeline-container {
                height: 300px;  /* Fixed height */
                overflow-y: auto;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
            .timeline-entry {
                padding: 8px 12px;
                margin: 4px 0;
                border-left: 3px solid #2e6fdf;
                background: white;
                border-radius: 4px;
                transition: all 0.2s ease;
            }
            .timeline-entry:hover {
                background: #f0f7ff;
                transform: translateX(4px);
            }
            .timestamp-button {
                width: 100%;
                text-align: left;
                padding: 4px 8px;
                background: transparent;
                border: none;
                cursor: pointer;
            }
            .timestamp-button:hover {
                background: #e9ecef;
                border-radius: 4px;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎬 Timeline")
        with st.container(height=400):  # Fixed height container
            for idx, (timestamp, description) in enumerate(zip(metadata_df['timestamp'], metadata_df['frame_description'])):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(f"⏱️ {format_timestamp(timestamp)}", 
                                   key=f"ts_{idx}",
                                   help="Click to jump to this timestamp"):
                            video_container.video(video_file, start_time=int(timestamp))
                    with col2:
                        st.markdown(f"<small>{description[:100]}...</small>", 
                                  unsafe_allow_html=True)
                st.markdown("<hr style='margin: 4px 0'>", unsafe_allow_html=True)
        
        # Add precise timestamp navigation below the scrollable timeline
        st.markdown("### ⏱️ Jump to Time")
        col1, col2 = st.columns([3, 1])
        with col1:
            target_time = st.slider(
                "Select time",
                min_value=0.0,
                max_value=float(metadata_df['timestamp'].max()),
                value=0.0,
                step=1.0,
                format="%d seconds"
            )
        with col2:
            if st.button("Jump", help="Jump to selected timestamp"):
                video_container.video(video_file, start_time=int(target_time))

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS"""
    return str(timedelta(seconds=int(seconds))).split('.')[0]

def init_session_state():
    """Initialize session state variables"""
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'current_timestamp' not in st.session_state:
        st.session_state.current_timestamp = 0
    if 'metadata_df' not in st.session_state:
        st.session_state.metadata_df = None

def main():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1e3d8f;'>
                🎬 FINAS Demo x BlacX
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key = st.text_input(
            "Enter OpenAI API Key:",
            value=st.session_state.OPENAI_API_KEY,
            type="password",
            key="api_key_input"
        )
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue.")
        return
    
    st.session_state.OPENAI_API_KEY = api_key
    
    if 'metadata_manager' not in st.session_state:
        st.session_state.metadata_manager = MetadataManager(api_key)
    
    processor = VideoProcessor(api_key)
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Video (MP4, AVI)", type=['mp4', 'avi'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        video_col, control_col = st.columns([1, 2])
        
        with control_col:
            if st.button("🔄 Process Video"):
                with st.spinner("Processing video..."):
                    try:
                        # Get video properties first
                        cap = cv2.VideoCapture(video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        
                        if fps <= 0:
                            st.warning("Could not determine video frame rate. Using default settings.")
                        
                        # Process frames with known FPS
                        frames, timestamps, frame_images, video_metadata, frame_numbers = processor.extract_frames(
                            video_path, 
                            sample_rate=2  # Adjust sample rate as needed
                        )
                        
                        # Display frame rate information
                        st.info(f"Video FPS: {video_metadata['fps']:.2f}")
                        
                        status_container = st.empty()
                        frame_display = st.empty()
                        progress_bar = st.progress(0)
                        
                        frame_metadata = []
                        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                            status_container.markdown(f"""
                                <div class="status-box">
                                    <h4>Processing Frame {frame_numbers[i]} ({format_timestamp(timestamp)})</h4>
                                    <p class="video-timestamp">Timestamp: {format_timestamp(timestamp)}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            frame_display.image(frame_images[i], caption=f"Frame {frame_numbers[i]} | Time: {format_timestamp(timestamp)}")
                            metadata = processor.analyze_frame(frame, timestamp)
                            frame_metadata.append(metadata)
                            
                            progress_bar.progress((i + 1) / len(frames))
                        
                        status_container.markdown("""
                            <div class="status-box">
                                <h4>Processing Audio...</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        audio_data = processor.extract_audio(video_path)
                        
                        st.session_state.metadata_manager.create_metadata_df(
                            frame_metadata, 
                            timestamps, 
                            audio_data,
                            video_metadata,
                            frame_numbers
                        )
                        
                        status_container.empty()
                        frame_display.empty()
                        progress_bar.empty()
                        
                        st.success("✅ Video processing complete!")
                        st.session_state.video_processed = True
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                    finally:
                        os.unlink(video_path)
        
        if st.session_state.metadata_manager.metadata_df is not None:
            st.markdown("### 📺 Interactive Video Player")
            create_video_player(uploaded_file, st.session_state.metadata_manager.metadata_df)
            
            # Chat Interface
            st.markdown("""
                <div style='background-color: #fff; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                    <h3 style='color: #1e3d8f;'>💬 Chat with Video Content</h3>
                </div>
            """, unsafe_allow_html=True)
            
            user_query = st.text_input("🔍 Ask about the video:", key="chat_input")
            
            if user_query:
                with st.spinner("Processing query..."):
                    response = st.session_state.metadata_manager.query_metadata(user_query)
                    st.markdown(f"""
                        <div style='background-color: #f0f7ff; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                            <p><strong>Response:</strong> {response}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Metadata Display
            with st.expander("📊 Show Raw Metadata"):
                metadata_tabs = st.tabs(["Timeline View", "Detailed View", "Export"])
                
                with metadata_tabs[0]:
                    # Timeline view with frame numbers and formatted timestamps
                    st.dataframe(
                        st.session_state.metadata_manager.metadata_df[[
                            'frame_number',
                            'formatted_time', 
                            'frame_description'
                        ]],
                        use_container_width=True
                    )
                
                with metadata_tabs[1]:
                    # Detailed view
                    st.dataframe(
                        st.session_state.metadata_manager.metadata_df,
                        use_container_width=True
                    )
                
                with metadata_tabs[2]:
                    # Export options
                    st.markdown("### Export Metadata")
                    export_format = st.selectbox(
                        "Choose export format",
                        ["CSV", "JSON", "Excel"]
                    )
                    
                    if st.button("Export Metadata"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            if export_format == "CSV":
                                csv = st.session_state.metadata_manager.metadata_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"video_metadata_{timestamp}.csv",
                                    mime="text/csv"
                                )
                            elif export_format == "JSON":
                                json_str = st.session_state.metadata_manager.metadata_df.to_json(
                                    orient="records",
                                    date_format="iso"
                                )
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name=f"video_metadata_{timestamp}.json",
                                    mime="application/json"
                                )
                            elif export_format == "Excel":
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    st.session_state.metadata_manager.metadata_df.to_excel(
                                        writer,
                                        index=False,
                                        sheet_name='Video Metadata'
                                    )
                                st.download_button(
                                    label="Download Excel",
                                    data=output.getvalue(),
                                    file_name=f"video_metadata_{timestamp}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        except Exception as e:
                            st.error(f"Error exporting metadata: {str(e)}")

            # Video Analytics
            with st.expander("📈 Video Analytics"):
                if st.session_state.metadata_manager.metadata_df is not None:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Duration",
                            format_timestamp(st.session_state.metadata_manager.metadata_df['timestamp'].max())
                        )
                    
                    with col2:
                        st.metric(
                            "Analyzed Frames",
                            len(st.session_state.metadata_manager.metadata_df)
                        )
                    
                    with col3:
                        st.metric(
                            "Average Scene Length",
                            f"{st.session_state.metadata_manager.metadata_df['timestamp'].diff().mean():.2f}s"
                        )
                    
                    # Scene duration distribution visualization
                    st.markdown("### Scene Duration Distribution")
                    scene_durations = st.session_state.metadata_manager.metadata_df['timestamp'].diff().dropna()
                    fig = px.histogram(
                        scene_durations,
                        nbins=20,
                        labels={'value': 'Duration (seconds)', 'count': 'Number of Scenes'},
                        title='Distribution of Scene Durations'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional visualizations
                    st.markdown("### Content Timeline")
                    timeline_data = st.session_state.metadata_manager.metadata_df.copy()
                    timeline_data['scene_duration'] = timeline_data['timestamp'].diff()
                    
                    fig2 = px.line(
                        timeline_data,
                        x='timestamp',
                        y='scene_duration',
                        title='Scene Duration Over Time',
                        labels={
                            'timestamp': 'Video Timeline (seconds)',
                            'scene_duration': 'Scene Duration (seconds)'
                        }
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Download processed data
            if st.button("💾 Save Analysis Results"):
                try:
                    # Prepare data for download
                    export_data = {
                        'metadata': st.session_state.metadata_manager.metadata_df.to_dict('records'),
                        'video_info': {
                            'duration': st.session_state.metadata_manager.video_metadata['duration'],
                            'fps': st.session_state.metadata_manager.video_metadata['fps'],
                            'total_frames': st.session_state.metadata_manager.video_metadata['total_frames']
                        },
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    
                    # Convert to JSON
                    json_data = json.dumps(export_data, indent=2)
                    
                    # Create download button
                    st.download_button(
                        label="Download Analysis Results",
                        data=json_data,
                        file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error saving analysis results: {str(e)}")

if __name__ == "__main__":
    init_session_state()
    main()