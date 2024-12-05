from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

def tag_proper_nouns(text):

    print(text)

    honorifics = {
    "Mr.", "Mrs.", "Miss", "Ms.", "Mx.", "Sir", "Dame", "Dr.", "Prof.", "Lord", "Lady",
    "Sr.", "Sra.", "Srta.", "Don", "Doña", "Profesor", "Profesora", "Dott.", "Dott.ssa",
    "Sig.", "Sig.ra", "Sig.na", "Herr", "Frau", "Fräulein", "Maître", "Freiherr", "Freifrau",
    "Monsieur", "Madame", "Mademoiselle", "M.", "Mme.", "Mlle.", "Eng.", "De Heer", "Mevrouw",
    "Mejuffrouw", "Г-н", "Г-жа", "Г-ца", "Tovarishch", "Pan", "Pani", "Panna", "Herr", "Fru",
    "Fröken", "Frk.", "Tohtori", "Professori", "Kyrios", "Kyria", "Kathigitis", "Kathigitria",
    "Úr", "Asszony", "Kisasszony", "Paní", "Slečna", "Domnul", "Doamna", "Domnișoara",
    "Bey", "Bay", "Hanım", "Baron", "Baroness", "Duke", "Duchess", "Marquis", "Marquise",
    "Frau Doktor", "Herr Doktor", "Herr Professor", "Fru Doktor", "Herrn", "Dr.med.", "Ing."
}

    # Split the text into tokens
    tokens_punc = text.split(' ')
    tokens = []

    # Split eos punctuations unless it's part of an honorific.
    for token in tokens_punc:
        if token[-1] in {'.', '?', '!'}:
            if token not in honorifics:
                tokens.append(token[:-1])
                tokens.append(token[-1])
            else:
                tokens.append(token)
        else:
            tokens.append(token)

    print(tokens)

    # To get proper nouns mid-sentence
    tagged_tokens = []
    for i, token in enumerate(tokens):
        # Check capitalization
        if re.match(r'^[A-Z][a-z\'\-]*$', token):
            if i == 0 or tokens[i - 1] in {'.', '!', '?'} or token in honorifics:
                tagged_tokens.append(token)
            # Tag the token if it's not sentence-initial nor an honorific
            else:
                tagged_tokens.append(f"<PN>{token}</PN>")
        else:
            tagged_tokens.append(token)

    # To get potential proper nouns from start-of-sentence tokens by checking for recurrence
    for i, token in enumerate(tagged_tokens):
        if token[0] != '<' and f'<PN>{token}</PN>' in tagged_tokens:
            tagged_tokens[i] = f'<PN>{token}</PN>'

    return ' '.join(tagged_tokens)

# Example Usage
example_text = "Alice went to Paris. She met Dr. John there. He is Alice's friend. Dr. John and Alice had fun."
output = tag_proper_nouns(example_text)
print(output)

# # Load pre-trained T5 model and tokenizer
# model_name = "t5-small"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Add special tokens
# special_tokens_dict = {'additional_special_tokens': ['<PN>', '</PN>']}
# tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))

# # Preprocess data
# source_text = "Alice went to Paris."
# preprocessed_text = tag_proper_nouns(source_text)

# # Tokenize and fine-tune
# inputs = tokenizer(preprocessed_text, return_tensors="pt")
# labels = tokenizer("Alice fue a Paris.", return_tensors="pt")["input_ids"]
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# loss.backward()

# # Decode test sentence
# test_input = tokenizer(tag_proper_nouns("Bob visited Rome."), return_tensors="pt")
# translated_tokens = model.generate(**test_input)
# translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
# print(translated_text)
