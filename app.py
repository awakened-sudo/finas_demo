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
    h1 {
        color: #1e3d8f;
        text-align: center;
        padding: 2rem 0;
        font-weight: bold;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
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
        frame_images = []  # Store actual frames for display
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_rate)
        
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_images.append(frame_rgb)
                
                # Convert frame to base64 for API
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frames.append(base64_frame)
                timestamps.append(current_frame / fps)
                
            current_frame += 1
            
        cap.release()
        return frames, timestamps, frame_images

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
                max_completion_tokens=300,
                temperature=0.7
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
                    response_format="verbose_json",  # Get detailed JSON with timestamps
                    timestamp_granularities=["segment", "word"]  # Get both segment and word timestamps
                )
            
            os.unlink(audio_path)
            return transcript
        except Exception as e:
            return f"Error processing audio: {str(e)}"

class MetadataManager:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.metadata_df = None
        
    def create_metadata_df(self, frame_metadata, timestamps, audio_data):
        """Create a DataFrame with synchronized metadata"""
        # Process audio segments with timestamps
        audio_segments = []
        
        if isinstance(audio_data, str):  # Error case
            audio_segments = [audio_data] * len(timestamps)
        else:
            # Match audio segments to frame timestamps
            for timestamp in timestamps:
                relevant_text = self.get_audio_segment_for_timestamp(audio_data, timestamp)
                audio_segments.append(relevant_text)
        
        self.metadata_df = pd.DataFrame({
            'timestamp': timestamps,
            'frame_description': frame_metadata,
            'audio_transcript': audio_segments
        })
    
    def get_audio_segment_for_timestamp(self, audio_data, timestamp):
        """Extract relevant audio segment for given timestamp with precise timing"""
        try:
            if isinstance(audio_data, str):  # Error case
                return audio_data
            
            window_start = max(0, timestamp - 0.5)
            window_end = timestamp + 0.5
            
            # Handle the new OpenAI Whisper API response format
            if hasattr(audio_data, 'segments'):
                segments = audio_data.segments
            else:
                segments = audio_data.get('segments', [])
            
            relevant_words = []
            
            for segment in segments:
                # Get segment timestamps
                segment_start = segment.start if hasattr(segment, 'start') else segment.get('start', 0)
                segment_end = segment.end if hasattr(segment, 'end') else segment.get('end', 0)
                
                # Check if segment overlaps with our window
                if segment_start <= window_end and segment_end >= window_start:
                    # Get the text for this segment
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
            context = self.metadata_df.to_json(orient='records')
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI assistant analyzing video content. Provide detailed and accurate answers about the video based on frame descriptions and audio transcripts."
                    },
                    {
                        "role": "user", 
                        "content": f"Given this video metadata: {context}\n\nQuery: {query}"
                    }
                ],
                temperature=0.7,
                max_completion_tokens=150
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying metadata: {str(e)}"

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS.mm"""
    return str(timedelta(seconds=seconds)).split('.')[0]

def main():
    # Custom title with logo
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1e3d8f;'>
                🎬 FINAS Demo x BlacX
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for API key
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # API Key input
        api_key = st.text_input(
            "Enter OpenAI API Key:",
            value=st.session_state.OPENAI_API_KEY,
            type="password",
            key="api_key_input"
        )
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue.")
        return
    
    # Update session state with API key
    st.session_state.OPENAI_API_KEY = api_key
    
    # Initialize components with API key
    if 'metadata_manager' not in st.session_state:
        st.session_state.metadata_manager = MetadataManager(api_key)
    
    processor = VideoProcessor(api_key)
    
    # Video Upload
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Video (MP4, AVI)", type=['mp4', 'avi'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Create columns for video and processing controls
        video_col, control_col = st.columns([2, 1])
        
        with video_col:
            st.video(uploaded_file)
        
        with control_col:
            # Video Processing Button
            if st.button("🔄 Process Video"):
                with st.spinner("Initializing video processing..."):
                    try:
                        # Extract frames
                        frames, timestamps, frame_images = processor.extract_frames(video_path, sample_rate=2)
                        
                        # Create a container for processing status
                        status_container = st.empty()
                        frame_display = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Process frames with visual feedback
                        frame_metadata = []
                        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                            # Update status
                            status_container.markdown(f"""
                                <div class="status-box">
                                    <h4>Processing Frame {i+1}/{len(frames)}</h4>
                                    <p class="video-timestamp">Timestamp: {format_timestamp(timestamp)}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Display current frame
                            frame_display.image(frame_images[i], caption=f"Current frame: {format_timestamp(timestamp)}")
                            
                            # Process frame
                            metadata = processor.analyze_frame(frame, timestamp)
                            frame_metadata.append(metadata)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(frames))
                        
                        # Process audio
                        status_container.markdown("""
                            <div class="status-box">
                                <h4>Processing Audio...</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        audio_data = processor.extract_audio(video_path)
                        
                        # Create metadata DataFrame
                        st.session_state.metadata_manager.create_metadata_df(
                            frame_metadata, timestamps, audio_data
                        )
                        
                        # Clear temporary displays
                        status_container.empty()
                        frame_display.empty()
                        progress_bar.empty()
                        
                        st.success("✅ Video processing complete!")
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                    finally:
                        os.unlink(video_path)
        
        # Chat Interface
        if st.session_state.metadata_manager.metadata_df is not None:
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
                st.dataframe(st.session_state.metadata_manager.metadata_df, use_container_width=True)

if __name__ == "__main__":
    main()