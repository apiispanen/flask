import spacy # NLP processing
from spacy.matcher import Matcher
import spacy.cli
# spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
import numpy as np
import pandas as pd
# IMPORT THE ML LIBRARIES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dm_connect import execute_query

matcher = Matcher(nlp.vocab)

# Define a simple pattern for our custom entity of "interests"
pattern = [{"LOWER": {"IN": ["cutting", "sawing", "bending", "grinding", "beveling", "shaping", "forming", "pressing", "sanding", "drilling", "threading"]}}]
# Add the pattern to the matcher
matcher.add("FUNCTION", [pattern])

# Define a simple pattern for our custom entity of "interests"
pattern = [{"LOWER": {"IN": ["metal", "wood", "aluminum", "iron", "water", "sand", "steel", "copper", "rust", "slag", "carbon fiber"]}}]
# Add the pattern to the matcher
matcher.add("MATERIAL", [pattern])

# Load the product names from the CSV file
df = pd.read_csv('product_names.csv')
# Convert the DataFrame to a list
product_names = df.iloc[:, 0].tolist()
# Convert the product names to lowercase for matching
product_names = [name.lower() for name in product_names]
# Define the pattern for the matcher
pattern = [{"LOWER": {"IN": product_names}}]
# Add the pattern to the matcher
matcher.add("PRODUCT_NAME", [pattern])


# Load the product names from the CSV file
df = pd.read_csv('mfgs.csv')
# Convert the DataFrame to a list
mfg_names = df.iloc[:, 0].tolist()
# Convert the product names to lowercase for matching
mfg_names = [mfg.lower() for mfg in mfg_names]
# Define the pattern for the matcher
pattern = [{"LOWER": {"IN": mfg_names}}]
# Add the pattern to the matcher
matcher.add("MFG", [pattern])


def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        print(ent.label_, ent.text)
        entities[ent.label_] = ent.text
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # get string representation
        print("MATCHES", string_id, doc[start:end].text)
        entities[string_id] = doc[start:end].text
    return entities

# RETURNS THE SQL QUERY BASED ON THE INTENT AND ENTITIES
def generate_query(intent, entities):
    intent = intent.lower()
    
    
    intent_table_mapping = {
    "delivery_installation": "delivery_installation_table",
    "price_inquiry": "catalog_product_entity_decimal",
    "discount_inquiry": "salesrule_",
    "financing_inquiry": "financing_inquiry_table",
        "product_inquiry": "catalog_product_entity",
    "support_maintenance": "support_maintenance_table",
        "technical_specifications": "eav_attribute", 
    "warranty_info": "warranty_info_table",
    }   

    if intent not in intent_table_mapping:
        print("Invalid intent or intent not in mapping!")
        return

    table_name = intent_table_mapping[intent]


    where_clause = ""
    for entity, value in entities.items():
        if entity == "PRODUCT_NAME":
            # Use full-text search for product names
            where_clause += f" MATCH(product_name) AGAINST('{value}') AND"
        else:
            # For other entities, just use a simple equals comparison
            where_clause += f" {entity} = '{value}' AND"
            
    where_clause = where_clause.rstrip(" AND")  # remove trailing " AND"

    # Construct the SQL query
    query = f"SELECT * FROM mg6t_{table_name} WHERE{where_clause}"
    
    return query

# TIME FOR SOME DATA TRAINING   
data = pd.read_csv("intent_data.csv")
X_train, X_test, y_train, y_test = train_test_split(data["sentence"], data["intent"], test_size=0.2, random_state=42, stratify=data["intent"])


# TRAIN THE MODEL
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# print the shape of the pipeline model
print(pipeline.named_steps['tfidf'].transform(X_train).shape)


# Assuming y_train is a 1D numpy array
print(np.array(y_train).shape)


def predict_intent(text):
    return pipeline.predict([text])[0]


# # NOW LETS RUN THE MODEL FOR INTENTS
# sentence = "Is anything happening on May 5th in Harrisburg?"
# intent = predict_intent(sentence)
# print(f"Predicted intent: {intent}")

# # NEXT, IDENTIFY THE ENTITIES
# entities = extract_entities(sentence)
# print(f"Entities: {entities}")


def query_intent(text):
    # text = request.json.get('text')
    try:
        intent = predict_intent(text) # Can be done with the NLP cloud API
    except Exception as e:
        intent = ""
        print("(*&@#(!*@#*&!@#*&^#*&^#*&!^@#*&!^@# ERROR:",e)
    try:
        entities = extract_entities(text)
    except Exception as e:
        entities = {}
        print("(*&@#(!*@#*&!@#*&^#*&^#*&!^@#*&!^@# ERROR:",e)
    try:
        query = generate_query(intent, entities)
    except Exception as e:
        query = ""
        print("(*&@#(!*@#*&!@#*&^#*&^#*&!^@#*&!^@# ERROR:",e)
    try:
        results = execute_query(query) # ONCE WE HOOK UP THE DB THIS IS WHERE WE'LL EXECUTE THE QUERY
    except Exception as e:
        print("(*&@#(!*@#*&!@#*&^#*&^#*&!^@#*&!^@# ERROR:",e)
        results = ""
    return intent, entities, query, results


# Note: GPE is geopolitical entity
# print(query_intent("    "Find people who like hiking and camping"))