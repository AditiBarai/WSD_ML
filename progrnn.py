import tensorflow as tf 
from tensorflow.keras import Sequential  
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense  
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  

# Sentences where "কল" has different meanings
sentences = [
    "আমি কল থেকে জল খাই",  # "I drink water from the tap" - "কল" means tap
    "এই কারখানায় অনেক কল আছে",  # "There are many machines in this factory" - "কল" means machine
    "কল থেকে জল পড়ছে",  # "Water is falling from the tap" - "কল" means tap
    "কল চালানো খুব কঠিন কাজ",  # "Operating the machine is hard work" - "কল" means machine
]

# Labels for the meanings: 0 = "tap", 1 = "machine"
labels = [0, 1, 0, 1]

# Simple word-to-index mapping for encoding (normally you'd use a more complex method)
word_index = {
    "আমি": 1, "কল": 2, "থেকে": 3, "জল": 4, "খাই": 5,
    "এই": 6, "কারখানায়": 7, "অনেক": 8, "আছে": 9,
    "পড়ছে": 10, "চালানো": 11, "খুব": 12, "কঠিন": 13, "কাজ": 14,
    "মেরামত": 15, "করা": 16, "দরকার": 17
}

# Convert sentences to numerical format 
def encode_sentence(sentence, word_index):
    return [word_index[word] for word in sentence.split()]

encoded_sentences = np.array([encode_sentence(sentence, word_index) for sentence in sentences], dtype=object)

# Pad sequences
padded_sentences = pad_sequences(encoded_sentences, maxlen=6, padding='post')

# Convert to NumPy array
padded_sentences = np.array(padded_sentences)

# Convert labels to numpy array
y = np.array(labels)

# Create a simple RNN model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=8, input_length=6),  # len(word_index) + 1, 8-dim embeddings
    SimpleRNN(10, activation='relu'),  # RNN with 10 units
    Dense(1, activation='sigmoid')  # Output layer for binary classification (0 or 1)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on dummy data
model.fit(padded_sentences, y, epochs=10)

# New sentence to predict the meaning of "কল"
new_sentence = "কল মেরামত করা দরকার"  # "The machine needs repair" - "কল" means machine

# Encode the new sentence
encoded_new_sentence = encode_sentence(new_sentence, word_index)
encoded_new_sentence = pad_sequences([encoded_new_sentence], maxlen=6)  # Use pad_sequences

# Predict
prediction = model.predict(encoded_new_sentence)
print("Predicted meaning of 'কল':", "machine" if prediction > 0.5 else "tap")
print(tf.__version__)
