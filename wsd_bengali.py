import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sentences where "চাঁদ" has different meanings
sentences = [
    "আজ রাতে চাঁদটা উজ্জ্বলভাবে জ্বলছে",  # "The moon is shining brightly tonight" - "চাঁদ" means moon
    "বৃষ্টি আসার আগে চাঁদটা ঠিক করতে হবে",  # "We need to fix the roof before it rains" - "চাঁদ" means rooftop
    "আজ পূর্ণিমার চাঁদ দেখতে সবাই ছাদে উঠেছে",  # "Everyone went to the roof to see the full moon today" - "চাঁদ" means moon
    "আমাকে চাঁদ দিন সময় দাও, আমি কাজটা শেষ করব"  # "Give me a few days, I will finish the work" - "চাঁদ" means few
]

# Labels for the meanings: 0 = "moon", 1 = "rooftop", 2 = "few"
labels = [0, 1, 0, 2]

# Simple word-to-index mapping for encoding
word_index = {
    "আজ": 1, "রাতে": 2, "চাঁদটা": 3, "উজ্জ্বলভাবে": 4, "জ্বলছে": 5,
    "বৃষ্টি": 6, "আসার": 7, "আগে": 8, "ঠিক": 9, "করতে": 10, "হবে": 11,
    "পূর্ণিমার": 12, "চাঁদ": 13, "দেখতে": 14, "সবাই": 15, "ছাদে": 16, "উঠেছে": 17,
    "আমাকে": 18, "দিন": 19, "সময়": 20, "দাও,": 21, "আমি": 22, "কাজটা": 23, "শেষ": 24, "করব": 25
}

# Convert sentences to numerical format
def encode_sentence(sentence, word_index):
    return [word_index[word] for word in sentence.split() if word in word_index]

encoded_sentences = [encode_sentence(sentence, word_index) for sentence in sentences]

# Pad the sequences to make them of equal length
padded_sentences = pad_sequences(encoded_sentences, maxlen=8, padding='post')  # Maxlen is set to 8

# Convert labels to numpy array
y = np.array(labels)

# Create an LSTM model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=8, input_length=8),  # Embedding layer
    LSTM(10, activation='tanh'),  # LSTM layer with 10 units
    Dense(3, activation='softmax')  # Output layer for multi-class classification (3 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on dummy data
model.fit(padded_sentences, y, epochs=10, verbose=1)

# New sentence to predict the meaning of "চাঁদ"
new_sentence = "বৃষ্টি আসার আগে চাঁদটা ঠিক করতে হবে"  # Example: "The roof needs fixing before it rains"

# Encode the new sentence
encoded_new_sentence = encode_sentence(new_sentence, word_index)
encoded_new_sentence = pad_sequences([encoded_new_sentence], maxlen=8)

# Predict
prediction = model.predict(encoded_new_sentence)
predicted_label = np.argmax(prediction)

# Print the predicted meaning
meaning_mapping = {0: "moon", 1: "rooftop", 2: "few"}
print("Predicted meaning of 'চাঁদ':", meaning_mapping[predicted_label])
