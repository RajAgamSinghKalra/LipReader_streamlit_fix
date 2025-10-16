import os
from typing import List
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

'''# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import sys
import os
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import tempfile

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 

import matplotlib.cm as cm

# Convert to viridis colormap
def prepare_viridis_gif(video_frames):
    try:
        # Get viridis colormap
        colormap = cm.get_cmap('viridis')
        
        video_processed = []
        for frame in video_frames:
            # Squeeze to 2D: (50, 100, 1) -> (50, 100)
            frame_2d = np.squeeze(frame)
            
            # Normalize to 0-1 range for colormap
            if frame_2d.max() != frame_2d.min():  # Avoid division by zero
                frame_normalized = (frame_2d - frame_2d.min()) / (frame_2d.max() - frame_2d.min())
            else:
                frame_normalized = frame_2d
                
            # Apply viridis colormap (returns RGBA)
            frame_colored = colormap(frame_normalized)
            
            # Convert to RGB and uint8 (0-255)
            frame_rgb = (frame_colored[..., :3] * 255).astype(np.uint8)
            
            video_processed.append(frame_rgb)
        
        return video_processed
    except Exception as e:
        st.error(f"Error creating GIF: {e}")
        return None

# Keep your existing dropdown for sample videos (optional)
options = os.listdir(os.path.join('..', 'data', 'all'))
selected_video = st.selectbox('Choose sample video', options)

if selected_video and selected_video != "Choose...":
    # Your existing code for sample videos
    col1, col2 = st.columns(2)
    
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','all', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video_viridis = prepare_viridis_gif(video)
        imageio.mimsave('animation.gif', video_viridis, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        
        # Filter out padding tokens
        decoder = decoder[decoder != 0]
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        if len(decoder) > 0:
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
        else:
            st.text("No prediction generated")'''
            
            
# Import all of the dependencies
import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
import tempfile
import cv2
import matplotlib.cm as cm

from pathlib import Path
import urllib.request
import gdown

WEIGHTS_FILENAME = "checkpoint.weights.h5"
WEIGHTS_FILE_ID = "1Rv81h5t_VP8ryrDUe9qtIsZAwe-0pL1p"
WEIGHTS_URL = f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}"
SEGMENT_LENGTH = 75


def _candidate_weight_paths() -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent
    candidates = [
        root_dir / WEIGHTS_FILENAME,
        base_dir / WEIGHTS_FILENAME,
        root_dir / "models" / WEIGHTS_FILENAME,
        Path.cwd() / WEIGHTS_FILENAME,
    ]
    # Deduplicate while preserving order
    seen = []
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
            unique.append(candidate)
    return unique


def _is_valid_h5(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import h5py  # type: ignore
    except ImportError:
        # If h5py is unavailable, fall back to basic size check above.
        return True
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def download_model_weights() -> Path:
    for candidate in _candidate_weight_paths():
        if _is_valid_h5(candidate):
            return candidate

    target_path = _candidate_weight_paths()[0]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    st.info("Downloading model weights...")
    try:
        result = gdown.download(WEIGHTS_URL, str(target_path), quiet=False, fuzzy=True)
    except Exception as exc:
        st.error(f"Failed to download model weights: {exc}")
        st.stop()

    if result is None or not _is_valid_h5(target_path):
        try:
            if target_path.exists():
                target_path.unlink()
        except Exception:
            pass
        st.error(
            "Downloaded weights file is invalid. Please verify the Google Drive link."
        )
        st.stop()

    st.success("Model weights downloaded successfully!")
    return target_path


MODEL_WEIGHTS_PATH = download_model_weights()


def download_dlib_model():
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(model_path):
        st.info("Downloading dlib model...")
        # Download compressed version
        compressed_path = model_path + ".bz2"
        urllib.request.urlretrieve(model_url, compressed_path)
        
        # Extract (you'll need bz2 module)
        import bz2
        with bz2.open(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(compressed_path)
download_dlib_model()

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 

# Import your custom functions (make sure these are in your deployment)
try:
    from utils import load_data, num_to_char
    from modelutil import load_model
except ImportError:
    st.error("Required modules not found. Please ensure utils.py and modelutil.py are in your deployment.")
    st.stop()

# Convert to viridis colormap
def prepare_viridis_gif(video_frames):
    try:
        # Get viridis colormap
        colormap = cm.get_cmap('viridis')
        
        video_processed = []
        for frame in video_frames:
            # Squeeze to 2D: (50, 100, 1) -> (50, 100)
            frame_2d = np.squeeze(frame)
            
            # Normalize to 0-1 range for colormap
            if frame_2d.max() != frame_2d.min():  # Avoid division by zero
                frame_normalized = (frame_2d - frame_2d.min()) / (frame_2d.max() - frame_2d.min())
            else:
                frame_normalized = frame_2d
                
            # Apply viridis colormap (returns RGBA)
            frame_colored = colormap(frame_normalized)
            
            # Convert to RGB and uint8 (0-255)
            frame_rgb = (frame_colored[..., :3] * 255).astype(np.uint8)
            
            video_processed.append(frame_rgb)
        
        return video_processed
    except Exception as e:
        st.error(f"Error creating GIF: {e}")
        return None


def split_into_segments(frames: np.ndarray, segment_length: int = SEGMENT_LENGTH) -> List[np.ndarray]:
    """Split frames into contiguous 75-frame segments, padding the final chunk."""
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with 4 dimensions, got shape {frames.shape}")

    total_frames = frames.shape[0]
    if total_frames == 0:
        return []

    segments: List[np.ndarray] = []
    for start in range(0, total_frames, segment_length):
        chunk = frames[start : start + segment_length]
        if chunk.shape[0] == 0:
            continue
        if chunk.shape[0] < segment_length:
            pad_frame = chunk[-1][None, ...]
            padding = np.repeat(pad_frame, segment_length - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        segments.append(chunk)

    return segments

'''# File uploader - main functionality for deployed app
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mpg'])

if uploaded_file is not None:
    # Save uploaded file to temporary location with correct extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name

    # Generate two columns 
    col1, col2 = st.columns(2)

    # Rendering the video 
    with col1: 
        st.info('Uploaded video')
        # Convert MPG to MP4 for browser display
        if file_extension == '.mpg':
            converted_path = temp_video_path.replace('.mpg', '.mp4')
            os.system(f'ffmpeg -i {temp_video_path} -vcodec libx264 {converted_path} -y')
            display_path = converted_path
        else:
            display_path = temp_video_path
        
        # Display the video
        video_file = open(display_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_file.close()

    with col2: 
        st.info('Processing video...')
        try:
            with st.spinner('Loading and processing video...'):
                # Load and process video
                if file_extension == '.mpg':
                    video = load_data(tf.convert_to_tensor(display_path))
                else:
                    video = load_data(tf.convert_to_tensor(temp_video_path))

                
            if video is not None and len(video) > 0:
                st.success(f"Video loaded successfully! Shape: {video.shape}")
                
                # Process for GIF
                with st.spinner('Creating visualization...'):
                    video_viridis = prepare_viridis_gif(video)
                
                if video_viridis is not None and len(video_viridis) > 0:
                    # Save GIF
                    imageio.mimsave('animation.gif', video_viridis, fps=10)
                    st.image('animation.gif', width=400, caption='What the model sees (mouth crops)')
                else:
                    st.error("Failed to process video frames for visualization")
                
                # Prediction
                st.info('Making prediction...')
                with st.spinner('Running model prediction...'):
                    model = load_model()
                    
                    # Ensure video has correct shape for prediction
                    if video.shape[0] < 75:
                        st.warning(f"Video has {video.shape[0]} frames. Padding to 75.")
                        padding = np.zeros((75 - video.shape[0], video.shape[1], video.shape[2], video.shape[3]))
                        video = np.concatenate([video, padding], axis=0)
                    elif video.shape[0] > 75:
                        st.warning(f"Video has {video.shape[0]} frames. Truncating to 75.")
                        video = video[:75]
                    
                    yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
                    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                    
                    # Filter out padding tokens (usually 0)
                    decoder = decoder[decoder != 0]
                
                st.info('Model Output (Tokens):')
                st.code(str(decoder))

                # Convert prediction to text
                st.info('Final Prediction:')
                if len(decoder) > 0:
                    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
                    st.success(f"**{converted_prediction}**")
                else:
                    st.warning("No meaningful prediction generated")
            else:
                st.error("No video frames were loaded. The video might be incompatible.")
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("This might be due to:")
            st.write("- Video format not supported")
            st.write("- Video too short/long")
            st.write("- Face not detected in video")
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                # Clean up converted MP4 file if it was created
                if file_extension == '.mpg':
                    converted_path = temp_video_path.replace('.mpg', '.mp4')
                    if os.path.exists(converted_path):
                        os.unlink(converted_path)
                if os.path.exists('animation.gif'):
                    os.remove('animation.gif')
                if os.path.exists('test_video.mp4'):
                    os.remove('test_video.mp4')
            except:
                pass'''


# File uploader - main functionality for deployed app
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mpg'])

if uploaded_file is not None:
    # Save uploaded file to temporary location with correct extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name

    # Generate two columns 
    col1, col2 = st.columns(2)

    # Rendering the video 
    with col1: 
        st.info('Uploaded video')
        
        # Convert MPG to MP4 for browser display
        if file_extension == '.mpg':
            # Create a new temporary file for the converted MP4
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as mp4_file:
                converted_path = mp4_file.name
            
            # Convert using ffmpeg
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', temp_video_path, '-vcodec', 'libx264', 
                converted_path, '-y'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(converted_path):
                display_path = converted_path
            else:
                st.warning("MPG conversion failed, trying to display original")
                display_path = temp_video_path
        else:
            display_path = temp_video_path
        
        # Display the video
        try:
            video_file = open(display_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()
        except Exception as e:
            st.error(f"Could not display video: {e}")
            # Fallback: display original uploaded file
            uploaded_file.seek(0)
            st.video(uploaded_file.read())

    with col2: 
        st.info('Processing video...')
        try:
            with st.spinner('Loading and processing video...'):
                # Load and process video using the original temp file
                video = load_data(tf.convert_to_tensor(temp_video_path))
                video_np = np.asarray(video)
            
            if video_np is not None and len(video_np) > 0:
                st.success(f"Video loaded successfully! Shape: {video_np.shape}")
                
                # Process for GIF
                with st.spinner('Creating visualization...'):
                    video_viridis = prepare_viridis_gif(video_np)
                
                if video_viridis is not None and len(video_viridis) > 0:
                    # Save GIF
                    imageio.mimsave('animation.gif', video_viridis, fps=10)
                    st.image('animation.gif', width=400, caption='What the model sees (mouth crops)')
                else:
                    st.error("Failed to process video frames for visualization")
                
                # Prediction
                st.info('Making prediction...')
                with st.spinner('Running model prediction...'):
                    model = load_model()
                    segments = split_into_segments(video_np, SEGMENT_LENGTH)

                    if not segments:
                        raise ValueError("No valid frame segments could be created from the video.")

                    segment_tokens: List[List[int]] = []
                    logits_list: List[np.ndarray] = []

                    for idx, segment in enumerate(segments, start=1):
                        segment_tensor = tf.convert_to_tensor(segment, dtype=tf.float32)
                        yhat = model.predict(tf.expand_dims(segment_tensor, axis=0), verbose=0)
                        logits_list.append(yhat)

                        decoded = tf.keras.backend.ctc_decode(
                            yhat, [SEGMENT_LENGTH], greedy=True
                        )[0][0].numpy()
                        decoded = decoded[decoded != 0]
                        segment_tokens.append(decoded.tolist())

                    combined_logits = np.concatenate(logits_list, axis=1)
                    total_timesteps = combined_logits.shape[1]
                    full_decoded = tf.keras.backend.ctc_decode(
                        combined_logits, [total_timesteps], greedy=True
                    )[0][0].numpy()
                    full_decoded = full_decoded[full_decoded != 0]

                st.info('Model Output (Tokens per segment):')
                for segment_idx, tokens in enumerate(segment_tokens, start=1):
                    st.write(f"Segment {segment_idx}:")
                    st.code(str(tokens))

                st.info('Final Prediction:')
                if full_decoded.size > 0:
                    tokens_tensor = tf.convert_to_tensor(full_decoded, dtype=tf.int64)
                    converted_prediction = tf.strings.reduce_join(
                        num_to_char(tokens_tensor)
                    ).numpy().decode('utf-8')
                    st.success(f"**{converted_prediction}**")
                else:
                    st.warning("No meaningful prediction generated")
            else:
                st.error("No video frames were loaded. The video might be incompatible.")
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("This might be due to:")
            st.write("- Video format not supported")
            st.write("- Video too short/long")
            st.write("- Face not detected in video")
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                # Clean up converted MP4 file if it was created
                if file_extension == '.mpg' and 'converted_path' in locals():
                    if os.path.exists(converted_path):
                        os.unlink(converted_path)
                if os.path.exists('animation.gif'):
                    os.remove('animation.gif')
            except:
                pass
# Add some instructions
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. **Upload a video** - Supported formats: MP4, AVI, MOV, MKV, WEBM
    2. **Wait for processing** - The app will extract mouth regions and run the model
    3. **View results** - See the original video, what the model sees, and the prediction
    
    **Requirements for good results:**
    - Clear view of the person's face
    - Good lighting
    - Person facing the camera
    - Video length: 2-5 seconds recommended
    """)

# Add requirements for deployment
st.sidebar.info("""
**Deployment Requirements:**
- TensorFlow
- OpenCV
- ImageIO
- NumPy
- Streamlit
- Your model files
- utils.py & modelutil.py
""")
