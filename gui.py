import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = load_model('audio_emotion_detection.keras')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

def extract_mel_spectrogram(file_path, n_mels=128, fmax=8000):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def save_spectrogram_as_image(S_dB, file_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=22050, hop_length=512, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    st.title("Emotion Detection from A Female's Audio")
    
    uploaded_file = st.file_uploader("Upload an audio file with female voice only", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the uploaded audio file
        mel_spectrogram = extract_mel_spectrogram(file_path)
        image_path = os.path.join('uploads', 'mel_spectrogram.png')
        save_spectrogram_as_image(mel_spectrogram, image_path)
        
        # Load and preprocess the spectrogram image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict the emotion
        prediction = model.predict(img_array)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        st.write(f"Detected Emotion: {predicted_label}")
        st.image(image_path, caption='Mel Spectrogram')
        
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    main()
