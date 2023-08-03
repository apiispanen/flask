# print("************** loading in ML libraries and updating model... *****************")

# import spacy # NLP processing
# from spacy.matcher import Matcher

# nlp = spacy.load("en_core_web_sm")
# import numpy as np
# import pandas as pd
# # IMPORT THE ML LIBRARIES
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from SpinnrAIWebService.mysql_connect import cursor_local, get_interests

# from fuzzywuzzy import fuzz
# import dateparser


# matcher = Matcher(nlp.vocab)

# # Initialize the lemmatizer
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

# # Modify your get_interests function to return lemmatized interests
# def get_lemmatized_interests():
#     lemmatized_interests = []
#     for interest in get_interests():
#         doc = nlp(interest)
#         lemmatized_interests.append(" ".join([token.lemma_ for token in doc]))
#     print("AFTER result of get_lemmatized_interest():", lemmatized_interests)
#     return lemmatized_interests


# # Define a simple pattern for our custom entity of "interests"
# pattern = [{"LEMMA": {"IN": get_lemmatized_interests()}}]

# # Add the pattern to the matcher
# matcher.add("INTERESTS", [pattern])

# # Modify your entity extraction function to lemmatize the text before matching
# def extract_entities(text):
#     doc = nlp(text)
#     entities = {}
#     for ent in doc.ents:
#         entities[ent.label_] = ent.text
#     matches = matcher(doc)
#     if matches:
#         _, start, end = matches[0]
#         entities["INTERESTS"] = doc[start:end].text
#     # print(entities)
#     return entities


# # TIME FOR SOME DATA TRAINING   
# data = pd.read_csv("intent_data.csv")
# X_train, X_test, y_train, y_test = train_test_split(data["sentence"], data["intent"], test_size=0.2, random_state=42, stratify=data["intent"])


# # TRAIN THE MODEL
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('classifier', LogisticRegression())
# ])

# pipeline.fit(X_train, y_train)

# y_pred = pipeline.predict(X_test)
# print(classification_report(y_test, y_pred))


# # print the shape of the pipeline model
# print(pipeline.named_steps['tfidf'].transform(X_train).shape)

# # Assuming y_train is a 1D numpy array
# print(np.array(y_train).shape)


# def predict_intent(text):
#     pred = pipeline.predict([text])[0]
#     pred_proba = pipeline.predict_proba([text]).max()
#     if pred_proba > 0.5:
#         return pred, pred_proba
#     else:
#         return 'Uncertain', pred_proba

# # # NOW LETS RUN THE MODEL FOR INTENTS
# # sentence = "Is anything happening on May 5th in Harrisburg?"
# # intent = predict_intent(sentence)
# # print(f"Predicted intent: {intent}")

# # # NEXT, IDENTIFY THE ENTITIES
# # entities = extract_entities(sentence)
# # print(f"Entities: {entities}")

# def query_intent(text):
#     def generate_query(intent, entities):
#         intent = intent.lower()
#         interests = entities.get("INTERESTS") # This is our custom entity "interests"
#         location = entities.get("GPE") # GPE is geopolitical entity

#         query = ""
#         if intent == "events":
#             date = entities.get("DATE") # Date entity
#             # if location:
#             #     query += f" AND location LIKE '%{location}%'"
#             if date:
#                 print(date)
#                 date = dateparser.parse(date)
                
#                 if date:
#                     date_str = date.strftime('%Y-%m-%d %H:%M:%S')
#                     query += f" AND (start_date >= '{date_str}' OR end_date <= '{date_str}')"
#             query = "SELECT title, description, start_date, location FROM events WHERE 1=1 " + query + " LIMIT 5"
#         elif intent == "people":
#             if interests:
#                 query = f" AND LOWER(interest.name) LIKE LOWER('%{interests}%')"
#             query = f"SELECT user.id, user.firstname, user.lastname, interest.name as interest_name FROM user INNER JOIN userinterest ON user.id = userinterest.userid INNER JOIN interest ON userinterest.refid = interest.refid WHERE 1=1 " + query + " LIMIT 5"
#         return query

   

#     def execute_query(sql, entities, intent, cursor_local=cursor_local):
#         cursor_local.execute(sql)
#         results = cursor_local.fetchall()

#         fuzzy_results = []
#         for row in results:
#             # print(results)
#             if intent.lower() == "events" and 'GPE' in entities:
#                 # print(entities['GPE'], row[3])
#                 location_score = fuzz.ratio(entities['GPE'], row[3]) # location is the 4th column
#                 print("EVENTS LOCATION SCORE:", location_score)
#                 if location_score > 50:  # adjust this threshold as per your requirement
#                     fuzzy_results.append({
#                        'title':row[0], 
#                        'description':row[1], 
#                        'start_date':row[2], 
#                        'location':row[3]
#                     }
#                     )
#             elif intent.lower() == "people" and 'INTERESTS' in entities:
#                 # print((entities['INTERESTS'], row[-1]))
#                 interests_score = fuzz.ratio(entities['INTERESTS'], row[-1]) # assuming interests is the last column
#                 print("PEOPLE INTERESTS SCORE:", interests_score)
#                 if interests_score > 65:  # adjust this threshold as per your requirement
#                     fuzzy_results.append( {
#                             "user.id":row[0], 
#                             "user.firstname":row[1], 
#                             "user.lastname":row[2], 
#                             "interest.name":row[3]
#                         })
                    
#             elif intent.lower() == 'squads' and 'INTERESTS' in entities:
#                 # print((entities['INTERESTS'], row[-1]))
#                 interests_score = fuzz.ratio(entities['INTERESTS'], row[-1]) # assuming interests is the last column
#                 print("SQUADS INTERESTS SCORE:", interests_score)
#                 if interests_score > 75:  # adjust this threshold as per your requirement
#                     fuzzy_results.append( {
#                             "title":row[0], 
#                             "question":row[1], 
#                             "city":row[2], 
#                             "zip_code":row[3],
#                             "state":row[4],
#                             "interest_name":row[5]
#                         })

#         # for row in fuzzy_results:
#         #     print("FUZZY RESULT:", row)
#         return fuzzy_results

#     intent, confidence = predict_intent(text) # Can be done with the NLP cloud API
#     entities = extract_entities(text)
#     print("ENTITIES TO FEED:",entities, "INTENT:",intent)
#     query = generate_query(intent, entities)
#     print(query)
#     results_dict = {"intent":intent, "confidence": confidence, "entities":entities, "query": query}
#     print("RESULTS DICT***********",results_dict)
#     return results_dict

#     results = execute_query(query, entities, intent) # pass entities and intent to execute_query
#     squads = {}
#     if entities.get("INTERESTS") or entities.get("GPE"):
#         sql = "SELECT s.title, s.question, s.city, s.zipcode, s.state, i.name 'interest_name' FROM squad s JOIN interest i ON i.refid = s.refinterestid LIMIT 5"
#         squads = execute_query(sql, entities, "squads")
#         if squads: 
#             squads =  [{ "title": squad["title"],
#                         "question":squad['question'],
#                         "interest_name":squad["interest_name"], 
#                         "city":squad['city'],
#                         "state":squad['state'],
#                         "zip_code":squad["zip_code"]
#                             } for squad in squads]
#         else: 
#             squads = None

#     return {"results":results, "squads":squads, "confidence": confidence}




# # Note: GPE is geopolitical entity
# # print(query_intent("Who likes hiking or skiing?"))


# def simple_query(sql, cursor_local=cursor_local):
#     cursor_local.execute(sql)
#     results = cursor_local.fetchall()
#     return results
