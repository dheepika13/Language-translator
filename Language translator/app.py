from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer

# Initialize the Flask application
app = Flask(__name__)

# Dictionary of supported language pairs
models = {
    "ENGLISH-URDU": "Helsinki-NLP/opus-mt-en-ur",
    "ENGLISH-HINDI": "Helsinki-NLP/opus-mt-en-hi",
    "ENGLISH-FRENCH": "Helsinki-NLP/opus-mt-en-fr",
    "ENGLISH-SPANISH": "Helsinki-NLP/opus-mt-en-es",
    "ENGLISH-ARABIC": "Helsinki-NLP/opus-mt-en-ar",
    "ENGLISH-GERMAN": "Helsinki-NLP/opus-mt-en-de",
}

tokenizers = {}
models_loaded = {}

for lang_pair, model_name in models.items():
    tokenizers[lang_pair] = MarianTokenizer.from_pretrained(model_name)
    models_loaded[lang_pair] = MarianMTModel.from_pretrained(model_name)

# Define translation function
def translate(text, dest_lang):
    model_name = models[dest_lang]
    tokenizer = tokenizers[dest_lang]
    model = models_loaded[dest_lang]

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Perform translation
    translated_ids = model.generate(inputs.input_ids, 
                                    num_beams=4, 
                                    max_length=512, 
                                    early_stopping=True)
    
    # Decode translated text
    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translated_text[0]

# Define the main route
@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    if request.method == 'POST':
        input_text = request.form['input_text']
        dest_lang = request.form['dest_lang']
        if dest_lang in models:
            translated_text = translate(input_text, dest_lang)
        else:
            translated_text = "Selected language model is not available."
    return render_template('index.html', input_text=request.form.get('input_text', ''), translated_text=translated_text, languages=models.keys())

if __name__ == '__main__':
    app.run(debug=True)
