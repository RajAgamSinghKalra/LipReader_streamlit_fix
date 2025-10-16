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

import urllib.request
import os
import gdown

def download_model_weights():
    model_path = "models/checkpoint.weights.h5"
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(model_path):
        st.info("Downloading model weights...")
        
        # Direct download link (if file is publicly shared)
        url = "https://drive.google.com/file/d/1Rv81h5t_VP8ryrDUe9qtIsZAwe-0pL1p/view?usp=sharing"
        gdown.download(url, model_path, quiet=False)
        
        st.success("Model weights downloaded successfully!")
download_model_weights()


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
    
    # Create temporary file with proper handling
    if file_extension == '.mpg':
        temp_suffix = '.mpg'
    else:
        temp_suffix = file_extension
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name

    # Generate two columns 
    col1, col2 = st.columns(2)

    # Rendering the video 
    with col1: 
        st.info('Uploaded video')
        
        # Display the uploaded video directly from memory
        uploaded_file.seek(0)  # Reset file pointer
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

    with col2: 
        st.info('Processing video...')
        try:
            with st.spinner('Loading and processing video...'):
                # Load and process video using the temporary file
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
