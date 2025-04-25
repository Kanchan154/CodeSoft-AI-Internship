import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import pickle

## Configuration
MAX_LENGTH = 40
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
LSTM_UNITS = 512
DENSE_UNITS = 512
BATCH_SIZE = 64
EPOCHS = 20

## Data Preparation (Example with Flickr8k dataset)
def load_captions(filename):
    """Load image captions from text file"""
    captions = {}
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split()
            image_id, image_desc = tokens[0], ' '.join(tokens[1:])
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(image_desc)
    return captions

def clean_caption(caption):
    """Clean and preprocess captions"""
    caption = caption.lower()
    # Add preprocessing steps (remove special chars, etc.)
    return caption

## Feature Extraction
def extract_features(directory):
    """Extract features from all images using pre-trained ResNet50"""
    model = ResNet50(include_top=False, pooling='avg')
    features = {}
    
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = tf.keras.applications.resnet50.preprocess_input(image)
            feature = model.predict(image, verbose=0)
            features[img_name.split('.')[0]] = feature
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
    
    return features

## Data Generator
def data_generator(captions, features, tokenizer, max_length, vocab_size, batch_size):
    """Generator for training data"""
    X1, X2, y = [], [], []
    n = 0
    while True:
        for image_id, caption_list in captions.items():
            n += 1
            feature = features[image_id]
            
            for caption in caption_list:
                seq = tokenizer.texts_to_sequences([caption])[0]
                
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(feature[0])
                    X2.append(in_seq)
                    y.append(out_seq)
            
            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = [], [], []
                n = 0

## Model Architecture
def create_model(vocab_size, max_length):
    """Create the image captioning model"""
    # Image feature extractor
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(DENSE_UNITS, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(LSTM_UNITS)(se2)
    
    # Decoder model
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(DENSE_UNITS, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Combine models
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

## Inference Model
def create_inference_model(model, max_length):
    """Create model for generating captions on new images"""
    # Extract components from trained model
    resnet = model.layers[2]
    embedding = model.layers[3]
    lstm = model.layers[5]
    decoder = model.layers[6]
    
    # Rebuild inference model
    input_features = Input(shape=(2048,))
    input_seq = Input(shape=(max_length,))
    
    fe = model.layers[1](input_features)
    fe = model.layers[2](fe)
    
    se = embedding(input_seq)
    se = model.layers[4](se)
    se = lstm(se)
    
    decoder_input = Add()([fe, se])
    decoder_output = decoder(decoder_input)
    
    inference_model = Model(inputs=[input_features, input_seq], outputs=decoder_output)
    return inference_model

## Caption Generation
def generate_caption(model, tokenizer, image_features, max_length):
    """Generate caption for an image"""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    
    # Remove startseq and endseq
    caption = in_text.replace('startseq ', '').replace(' endseq', '')
    return caption

def idx_to_word(integer, tokenizer):
    """Convert index to word"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

## Main Execution
if __name__ == "__main__":
    # Example usage (you'll need to prepare your dataset first)
    print("Loading and preprocessing data...")
    
    # 1. Load captions
    captions = load_captions("flickr8k/captions.txt")
    print(f"Loaded {len(captions)} image captions")
    
    # 2. Clean captions
    for key, desc_list in captions.items():
        for i in range(len(desc_list)):
            desc_list[i] = clean_caption(desc_list[i])
    
    # 3. Extract image features (this takes time, save features after first run)
    if not os.path.exists("features.pkl"):
        features = extract_features("flickr8k/images/")
        pickle.dump(features, open("features.pkl", "wb"))
    else:
        features = pickle.load(open("features.pkl", "rb"))
    print(f"Loaded {len(features)} image features")
    
    # 4. Prepare tokenizer
    all_captions = []
    for key in captions.keys():
        [all_captions.append(d) for d in captions[key]]
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_captions)
    word_index = tokenizer.word_index
    print(f"Vocabulary size: {len(word_index)}")
    
    # 5. Create model
    model = create_model(VOCAB_SIZE, MAX_LENGTH)
    model.summary()
    
    # 6. Train model (example)
    train_generator = data_generator(captions, features, tokenizer, MAX_LENGTH, VOCAB_SIZE, BATCH_SIZE)
    steps = len(captions) // BATCH_SIZE
    model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=steps, verbose=1)
    
    # 7. Save model
    model.save("image_captioning_model.h5")
    
    # 8. Test with sample image
    inference_model = create_inference_model(model, MAX_LENGTH)
    sample_image = "example.jpg"
    sample_features = extract_features(sample_image)
    caption = generate_caption(inference_model, tokenizer, sample_features, MAX_LENGTH)
    print(f"Generated caption: {caption}")