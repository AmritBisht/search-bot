import google.generativeai as genai

# Replace with your actual Gemini API key
API_KEY = "AIzaSyDV12vrr57IM98VmbEMdWm2fqDTMFFsNE4"

# Configure the client
genai.configure(api_key=API_KEY)

# Retrieve all available models
models = genai.list_models()

# Filter models that are related to embeddings
embedding_models = [model for model in models if "embedding" in model.name.lower()]

# Display the embedding models
print("Available Embedding Models:")
for model in embedding_models:
    print(f"- {model.name}")
