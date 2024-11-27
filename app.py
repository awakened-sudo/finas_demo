import streamlit as st
import cv2
import tempfile
import json
import numpy as np
import pandas as pd
from openai import OpenAI, LengthFinishReasonError
from pathlib import Path
import time
from datetime import datetime, timedelta
import base64
from moviepy.editor import VideoFileClip
import os
from dotenv import load_dotenv
import plotly.express as px
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

# Load environment variables
load_dotenv()

class EventData(BaseModel):
    eventID: str
    eventImageURL: str = ""
    inpoint: float
    outpoint: float

class CaptionTrack(BaseModel):
    eventData: List[EventData]

class SubtitleEntry(BaseModel):
    inpoint: str
    outpoint: str
    text: str

class Tracks(BaseModel):
    caption: CaptionTrack

class SourceData(BaseModel):
    description: str = ""
    title: Optional[str] = None
    file_id: int = Field(default_factory=lambda: int(time.time()))
    lls_kv_id: int = Field(default_factory=lambda: int(str(int(time.time()))[-8:]))
    thumbnail: str = Field(default_factory=lambda: f"{str(uuid.uuid4())[:8]}.png")
    clip_name: str = Field(default_factory=lambda: f"FIN-{str(int(time.time()))[-2:]}")
    clip_title: str = Field(default_factory=lambda: str(int(time.time())))
    duration: str = "00:00:00:00"
    proxy_uri: str = ""
    relative_path: str = "//"
    tracks: Dict[str, CaptionTrack] = Field(default_factory=dict, alias="_tracks")
    subtitles: Dict[str, List[SubtitleEntry]] = Field(default_factory=dict, alias="_subtitles")

class VideoMetadata(BaseModel):
    index: str = Field(default_factory=lambda: str(int(time.time())))
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    seq_no: int = Field(default_factory=lambda: int(time.time()))
    primary_term: int = 1
    found: bool = True
    source: SourceData

class FrameAnalysis(BaseModel):
    description: str
    objects_detected: List[str]
    scene_type: str

class QueryResponse(BaseModel):
    answer: str
    relevant_timestamps: List[str]
    confidence: float

class VideoProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.supported_languages = {
            'en-US': 'English',
            'ar-AR': 'Arabic',
            'zh-CN': 'Mandarin',
            'ta-IN': 'Tamil',
            'ms-MY': 'Malay'
        }
        
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following text to {target_language}. Maintain the original meaning and tone."
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "text"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Detect the language of the following text and respond with the language code only (e.g., 'en-US', 'ms-MY', etc.)"
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "text"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en-US"  # Default to English if detection fails

    def extract_frames(self, video_path, sample_rate=1):
        """Extract frames from video with enhanced metadata"""
        frames = []
        timestamps = []
        frame_images = []
        frame_numbers = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_rate)
        duration = total_frames / fps if fps > 0 else 0
        
        video_metadata = {
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames,
            'frame_interval': frame_interval
        }
        
        if fps <= 0:
            fps = 30  # Default fallback
            
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                timestamp = current_frame / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store frame data
                frame_images.append(frame_rgb)
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frames.append(base64_frame)
                timestamps.append(timestamp)
                frame_numbers.append(current_frame)
                
            current_frame += 1
            
        cap.release()
        return frames, timestamps, frame_images, video_metadata, frame_numbers

    def analyze_frame(self, frame_base64: str, timestamp: float) -> str:
        """Analyze frame using OpenAI Vision with structured output"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video frame analyzer. Describe the scene, detect objects, and categorize the scene type."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe this video frame at timestamp {timestamp:.2f} seconds."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_base64}"
                                }
                            }
                        ]
                    }
                ],
                response_format=FrameAnalysis,
                max_tokens=500
            )
            
            # Get the parsed response
            frame_analysis = completion.choices[0].message.parsed
            
            # Convert to JSON string to maintain compatibility with existing code
            return json.dumps({
                "description": frame_analysis.description,
                "objects_detected": frame_analysis.objects_detected,
                "scene_type": frame_analysis.scene_type
            })
            
        except Exception as e:
            if isinstance(e, LengthFinishReasonError):
                # Handle token limit error
                return json.dumps({
                    "description": "Error: Response exceeded token limit",
                    "objects_detected": [],
                    "scene_type": "error"
                })
            else:
                # Handle other errors
                return json.dumps({
                    "description": f"Error analyzing frame: {str(e)}",
                    "objects_detected": [],
                    "scene_type": "error"
                })

    def extract_audio_with_translations(self, video_path: str) -> dict:
        """Extract audio and generate translations"""
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
            
            # Process translations
            subtitles = {}
            source_language = self.detect_language(transcript.text)
            subtitles[source_language] = []
            
            # Convert original transcript to our format
            for segment in transcript.segments:
                subtitles[source_language].append({
                    "inpoint": str(segment.start),
                    "outpoint": str(segment.end),
                    "text": segment.text
                })
            
            # Generate translations
            for lang_code in self.supported_languages.keys():
                if lang_code != source_language:
                    subtitles[lang_code] = []
                    for segment in transcript.segments:
                        translated_text = self.translate_text(segment.text, self.supported_languages[lang_code])
                        subtitles[lang_code].append({
                            "inpoint": str(segment.start),
                            "outpoint": str(segment.end),
                            "text": translated_text
                        })
            
            return subtitles
            
        except Exception as e:
            print(f"Error in audio extraction: {e}")
            return {"error": str(e)}

class MetadataManager:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.metadata_df = None
        self.video_metadata = None
        self.structured_data = None
        
    def create_metadata_df(self, frame_metadata, timestamps, audio_data, video_metadata, frame_numbers):
        """Create structured metadata with enhanced organization"""
        self.video_metadata = video_metadata
        
        # Parse frame metadata
        parsed_metadata = []
        for metadata_str in frame_metadata:
            try:
                metadata = json.loads(metadata_str)
                parsed_metadata.append(metadata)
            except:
                parsed_metadata.append({
                    "description": metadata_str,
                    "objects_detected": [],
                    "scene_type": "unknown"
                })
        
        # Create base DataFrame
        self.metadata_df = pd.DataFrame({
            'frame_number': frame_numbers,
            'timestamp': timestamps,
            'formatted_time': [str(timedelta(seconds=int(t))) for t in timestamps],
            'frame_description': [m['description'] for m in parsed_metadata],
            'objects_detected': [m['objects_detected'] for m in parsed_metadata],
            'scene_type': [m['scene_type'] for m in parsed_metadata]
        })
        
        # Create event data
        event_data = []
        for _, row in self.metadata_df.iterrows():
            event_data.append({
                "eventID": row['frame_description'],
                "eventImageURL": "",
                "inpoint": float(row['timestamp']),
                "outpoint": float(row['timestamp'])
            })
        
        # Create source data with required structure
        source_data = {
            "description": video_metadata.get('description', ''),
            "title": None,
            "file_id": int(time.time()),
            "lls_kv_id": int(str(int(time.time()))[-8:]),
            "thumbnail": f"{str(uuid.uuid4())[:8]}.png",
            "clip_name": f"FIN-{str(int(time.time()))[-2:]}",
            "clip_title": str(int(time.time())),
            "duration": self.format_duration(video_metadata['duration']),
            "proxy_uri": "",
            "relative_path": "//",
            "tracks": {
                "caption": {
                    "eventData": event_data
                }
            },
            "subtitles": audio_data if isinstance(audio_data, dict) else {}
        }
        
        # Create complete metadata structure
        self.structured_data = {
            "index": str(int(time.time())),
            "id": str(uuid.uuid4()),
            "version": 1,
            "seq_no": int(time.time()),
            "primary_term": 1,
            "found": True,
            "source": source_data
        }

    def get_event_data(self):
        """Safely get event data from structured data"""
        try:
            return self.structured_data['source']['tracks']['caption']['eventData']
        except (KeyError, TypeError):
            return []

    def format_duration(self, seconds):
        """Format duration in HH:MM:SS:FF format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * 24)  # Assuming 24 fps
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def get_structured_output(self):
        """Return the complete structured output"""
        return self.structured_data

    def query_metadata(self, query):
        """Enhanced metadata querying with structured output support"""
        if self.metadata_df is None:
            return "No metadata available. Please process a video first."
        
        try:
            # Prepare context from structured data
            metadata_context = []
            for _, row in self.metadata_df.iterrows():
                metadata_context.append({
                    "timestamp": row['formatted_time'],
                    "description": row['frame_description'],
                    "objects": row['objects_detected'],
                    "scene_type": row['scene_type']
                })
            
            # Define the response schema using Pydantic
            class QueryResponse(BaseModel):
                answer: str
                relevant_timestamps: List[str]
                confidence: float
            
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing video content with structured metadata. Provide detailed answers based on the frame descriptions, detected objects, and scene types."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this video metadata:\n{json.dumps(metadata_context, indent=2)}\n\nQuery: {query}"
                    }
                ],
                response_format=QueryResponse
            )
            
            result = response.choices[0].message.parsed
            return f"""Answer: {result.answer}\n\nRelevant Timestamps: {', '.join(result.relevant_timestamps)}\nConfidence: {result.confidence:.2f}"""
            
        except Exception as e:
            return f"Error querying metadata: {str(e)}"

    def export_metadata(self, format_type="json"):
        """Export metadata in various formats"""
        if format_type == "json":
            return json.dumps(self.structured_data, indent=2)
        elif format_type == "csv":
            return self.metadata_df.to_csv(index=False)
        elif format_type == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                self.metadata_df.to_excel(writer, index=False, sheet_name='Video Metadata')
                
                # Add structured data sheet
                pd.DataFrame([self.structured_data]).to_excel(
                    writer, 
                    index=False, 
                    sheet_name='Structured Output'
                )
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def get_subtitle_languages(self):
        """Get list of available subtitle languages"""
        if self.structured_data and 'source' in self.structured_data:
            subtitles = self.structured_data['source'].get('subtitles', {})
            return list(subtitles.keys())
        return []

    def get_subtitles_for_language(self, language_code):
        """Get subtitles for specific language"""
        if self.structured_data and 'source' in self.structured_data:
            subtitles = self.structured_data['source'].get('subtitles', {})
            return subtitles.get(language_code, [])
        return []
    
def format_timestamp(seconds):
    """Format seconds into HH:MM:SS"""
    return str(timedelta(seconds=int(float(seconds)))).split('.')[0]


def create_video_player(video_file, metadata_manager):
    """Enhanced video player with subtitle support"""
    video_col, timeline_col = st.columns([1, 1])
    
    with video_col:
        video_container = st.empty()
        st.markdown("""
            <style>
            .video-container video {
                max-width: 100%;
                max-height: 400px;
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
        
        # Subtitle language selector
        available_languages = metadata_manager.get_subtitle_languages()
        if available_languages:
            selected_language = st.selectbox(
                "Select Subtitle Language",
                available_languages,
                format_func=lambda x: {
                    'en-US': 'English',
                    'ar-AR': 'Arabic',
                    'zh-CN': 'Mandarin',
                    'ta-IN': 'Tamil',
                    'ms-MY': 'Malay'
                }.get(x, x)
            )
            
            subtitles = metadata_manager.get_subtitles_for_language(selected_language)
            if subtitles:
                st.markdown("### Subtitles")
                for subtitle in subtitles:
                    st.markdown(f"""
                        <div class='subtitle-entry'>
                            <small>{subtitle['inpoint']} - {subtitle['outpoint']}</small><br/>
                            {subtitle['text']}
                        </div>
                    """, unsafe_allow_html=True)
    
    with timeline_col:
        st.markdown("### 🎬 Timeline")
        with st.container():
            # Get structured events from metadata
            events = metadata_manager.get_event_data()
            
            if events:
                for event in events:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button(f"⏱️ {format_timestamp(event['inpoint'])}", 
                                       key=f"ts_{event['inpoint']}"):
                                video_container.video(video_file, start_time=int(event['inpoint']))
                        with col2:
                            st.markdown(f"<small>{event['eventID'][:100]}...</small>", 
                                      unsafe_allow_html=True)
                        st.markdown("<hr style='margin: 4px 0'>", unsafe_allow_html=True)
            else:
                st.info("No timeline events available.")

def create_metadata_viewer(metadata_manager):
    """Enhanced metadata viewer with structured data"""
    tabs = st.tabs(["Timeline", "Structured Data", "Subtitles", "Export"])
    
    with tabs[0]:
        st.dataframe(
            metadata_manager.metadata_df[[
                'frame_number',
                'formatted_time', 
                'frame_description',
                'scene_type'
            ]],
            use_container_width=True
        )
    
    with tabs[1]:
        st.json(metadata_manager.structured_data)
    
    with tabs[2]:
        selected_language = st.selectbox(
            "Select Language",
            metadata_manager.get_subtitle_languages(),
            key="subtitle_viewer"
        )
        
        subtitles = metadata_manager.get_subtitles_for_language(selected_language)
        st.dataframe(pd.DataFrame(subtitles))
    
    with tabs[3]:
        col1, col2 = st.columns([3, 1])
        with col1:
            export_format = st.selectbox(
                "Choose export format",
                ["JSON", "CSV", "Excel"]
            )
        
        with col2:
            if st.button("Export"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if export_format == "JSON":
                        data = metadata_manager.export_metadata("json")
                        st.download_button(
                            "Download JSON",
                            data,
                            f"video_metadata_{timestamp}.json",
                            "application/json"
                        )
                    elif export_format == "CSV":
                        data = metadata_manager.export_metadata("csv")
                        st.download_button(
                            "Download CSV",
                            data,
                            f"video_metadata_{timestamp}.csv",
                            "text/csv"
                        )
                    elif export_format == "Excel":
                        data = metadata_manager.export_metadata("excel")
                        st.download_button(
                            "Download Excel",
                            data,
                            f"video_metadata_{timestamp}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")

def create_analytics_view(metadata_manager):
    """Enhanced analytics view with structured data insights"""
    st.markdown("### 📊 Video Analytics")
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Duration",
            metadata_manager.structured_data['source']['duration']
        )
    with col2:
        st.metric(
            "Total Scenes",
            len(metadata_manager.structured_data['source']['tracks']['caption']['eventData'])
        )
    with col3:
        st.metric(
            "Available Languages",
            len(metadata_manager.get_subtitle_languages())
        )

    
    # Scene analysis
    scene_types = metadata_manager.metadata_df['scene_type'].value_counts()
    fig = px.pie(
        values=scene_types.values,
        names=scene_types.index,
        title="Scene Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Object detection timeline
    st.markdown("### 🎯 Object Detection Timeline")
    object_timeline = []
    for _, row in metadata_manager.metadata_df.iterrows():
        for obj in row['objects_detected']:
            object_timeline.append({
                'timestamp': row['formatted_time'],
                'object': obj
            })
    
    if object_timeline:
        df_timeline = pd.DataFrame(object_timeline)
        fig = px.scatter(
            df_timeline,
            x='timestamp',
            y='object',
            title="Object Appearances Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

def init_session_state():
    """Initialize session state variables"""
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'current_timestamp' not in st.session_state:
        st.session_state.current_timestamp = 0
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

def main():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1e3d8f;'>
                🎬 FINAS Demo x BlacX - Enhanced Video Analysis
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # API Key Management
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
    
    # Initialize processors with API key
    processor = VideoProcessor(api_key)
    metadata_manager = MetadataManager(api_key)  # Create instance here
    
    # File Upload
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Video (MP4, AVI)", 
            type=['mp4', 'avi'],
            help="Upload a video file for analysis"
        )
    
    if uploaded_file:
        # Create temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Video Processing Section
        st.markdown("### 🎥 Video Processing")
        
        process_col, status_col = st.columns([1, 2])
        
        with process_col:
            if st.button("🔄 Process Video", help="Start video analysis"):
                with st.spinner("Processing video..."):
                    try:
                        # Initialize progress tracking
                        progress_bar = st.progress(0.0)
                        status_area = st.empty()
                        frame_preview = st.empty()
                        
                        # Extract frames (25% of total progress)
                        status_area.markdown("⏳ Extracting video frames...")
                        frames, timestamps, frame_images, video_metadata, frame_numbers = processor.extract_frames(
                            video_path,
                            sample_rate=2
                        )
                        progress_bar.progress(0.25)
                        
                        # Process frames with structured output (50% of total progress)
                        status_area.markdown("🔍 Analyzing frames...")
                        frame_metadata = []
                        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                            status_area.markdown(f"Processing frame {i+1}/{len(frames)}")
                            frame_preview.image(frame_images[i])
                            metadata = processor.analyze_frame(frame, timestamp)
                            frame_metadata.append(metadata)
                            # Calculate frame analysis progress (25% to 75% range)
                            frame_progress = 0.25 + (i / len(frames)) * 0.5
                            progress_bar.progress(min(frame_progress, 0.75))
                        
                        # Process audio and generate translations (15% of total progress)
                        status_area.markdown("🔊 Processing audio and generating translations...")
                        audio_data = processor.extract_audio_with_translations(video_path)
                        progress_bar.progress(0.90)
                        
                        # Create structured metadata (final 10% of progress)
                        status_area.markdown("📊 Organizing metadata...")
                        metadata_manager.create_metadata_df(  # Use local instance
                            frame_metadata,
                            timestamps,
                            audio_data,
                            video_metadata,
                            frame_numbers
                        )
                        
                        # Store metadata manager in session state after processing
                        st.session_state.metadata_manager = metadata_manager
                        
                        progress_bar.progress(1.0)
                        
                        # Clear temporary displays
                        status_area.empty()
                        frame_preview.empty()
                        
                        st.success("✅ Video processing complete!")
                        st.session_state.video_processed = True
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                    finally:
                        os.unlink(video_path)

        
        # Display processed content if available
        if st.session_state.video_processed and st.session_state.metadata_manager is not None:
            # Create tabs for different views
            main_tabs = st.tabs(["Video Player", "Analysis", "Metadata", "Export"])
            
            with main_tabs[0]:
                create_video_player(uploaded_file, st.session_state.metadata_manager)
            
            with main_tabs[1]:
                create_analytics_view(st.session_state.metadata_manager)
                
                # Interactive Query Section
                st.markdown("### 💬 Query Video Content")
                query = st.text_input(
                    "Ask about the video content:",
                    placeholder="e.g., What objects appear most frequently?"
                )
                
                if query:
                    with st.spinner("Analyzing query..."):
                        response = st.session_state.metadata_manager.query_metadata(query)
                        st.markdown(f"""
                            <div class='query-response'>
                                {response}
                            </div>
                        """, unsafe_allow_html=True)
            
            with main_tabs[2]:
                create_metadata_viewer(st.session_state.metadata_manager)
            
            with main_tabs[3]:
                st.markdown("### 💾 Export Options")
                export_format = st.selectbox(
                    "Select Export Format",
                    ["Structured JSON", "Full Analysis", "Subtitles Only"]
                )
                
                if st.button("Export Data"):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        if export_format == "Structured JSON":
                            data = json.dumps(
                                st.session_state.metadata_manager.structured_data,
                                indent=2
                            )
                            st.download_button(
                                "Download JSON",
                                data,
                                f"video_analysis_{timestamp}.json",
                                "application/json"
                            )
                        elif export_format == "Full Analysis":
                            # Export complete analysis including all metadata
                            full_data = {
                                "structured_data": st.session_state.metadata_manager.structured_data,
                                "frame_analysis": st.session_state.metadata_manager.metadata_df.to_dict('records'),
                                "video_info": st.session_state.metadata_manager.video_metadata
                            }
                            st.download_button(
                                "Download Full Analysis",
                                json.dumps(full_data, indent=2),
                                f"full_analysis_{timestamp}.json",
                                "application/json"
                            )
                        else:  # Subtitles Only
                            subtitle_data = st.session_state.metadata_manager.structured_data['_source']['_subtitles']
                            st.download_button(
                                "Download Subtitles",
                                json.dumps(subtitle_data, indent=2),
                                f"subtitles_{timestamp}.json",
                                "application/json"
                            )
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")

if __name__ == "__main__":
    init_session_state()
    main()