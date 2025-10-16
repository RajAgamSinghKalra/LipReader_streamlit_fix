import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import RNN, LSTMCell
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from utils import char_to_num

WEIGHTS_FILENAME = "checkpoint.weights.h5"


def _candidate_weight_paths() -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent
    candidates = [
        root_dir / WEIGHTS_FILENAME,
        base_dir / WEIGHTS_FILENAME,
        root_dir / "models" / WEIGHTS_FILENAME,
        Path.cwd() / WEIGHTS_FILENAME,
    ]
    seen = []
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
            unique.append(candidate)
    return unique


def _resolve_weights_path() -> Path:
    for candidate in _candidate_weight_paths():
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{WEIGHTS_FILENAME}'. Ensure the model weights are "
        "available before running predictions."
    )

def load_model() -> Sequential:
    model = Sequential()

    # Conv3D layers - these will use GPU
    model.add(Conv3D(128, 3, input_shape=(75,60,120,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    # Use RNN wrapper with LSTMCell instead of LSTM layer
    # This gives you GPU support without CuDNN dependency
    model.add(Bidirectional(RNN(LSTMCell(128), return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(RNN(LSTMCell(128), return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
    
    weights_path = _resolve_weights_path()
    model.load_weights(str(weights_path))
    
    return model




'''import os 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('..','models','checkpoint'))

    return model'''
