import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Extended sentences where "চাঁদ" has different meanings
sentences = [
    "আজ রাতে চাঁদটা উজ্জ্বলভাবে জ্বলছে",  # "moon"
    "বৃষ্টি আসার আগে চাঁদটা ঠিক করতে হবে",  # "rooftop"
    "আজ পূর্ণিমার চাঁদ দেখতে সবাই ছাদে উঠেছে",  # "moon"
    "আমাকে চাঁদ দিন সময় দাও, আমি কাজটা শেষ করব",  # "few"
    "চাঁদ খুব সুন্দর দেখাচ্ছে আজ",  # "moon"
    "চাঁদ মেরামত করতে হবে, নাহলে পানি পড়বে",  # "rooftop"
    "আমাকে কয়েক চাঁদ সময় দিন",  # "few"
    "রাতের আকাশে চাঁদ খুব বড় দেখাচ্ছে",  # "moon"
    "চাঁদটি পুরানো হয়ে গেছে, নতুন লাগাতে হবে",  # "rooftop"
    "আমি কিছু চাঁদ পর আবার আসব"  # "few"
]

# Labels: 0 = "moon", 1 = "rooftop", 2 = "few"
labels = [0, 1, 0, 2, 0, 1, 2, 0, 1, 2]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
encoded_sentences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to make them of equal length
max_length = max(len(seq) for seq in encoded_sentences)
padded_sentences = pad_sequences(encoded_sentences, maxlen=max_length, padding='post')

# Convert labels to numpy array
y = np.array(labels)

# Build the LSTM Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=10, input_length=max_length),
    LSTM(16, activation='tanh'),
    Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sentences, y, epochs=50, verbose=1)

# Predict meaning of "চাঁদ" in a new sentence
new_sentence = "বৃষ্টি আসার আগে চাঁদটা ঠিক করতে হবে"  # Expected meaning: "rooftop"
encoded_new_sentence = tokenizer.texts_to_sequences([new_sentence])
encoded_new_sentence = pad_sequences(encoded_new_sentence, maxlen=max_length)

# Prediction
prediction = model.predict(encoded_new_sentence)
predicted_label = np.argmax(prediction)

# Meaning mapping
meaning_mapping = {0: "moon", 1: "rooftop", 2: "few"}
print("Predicted meaning of 'চাঁদ':", meaning_mapping[predicted_label])