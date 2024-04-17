# Admission number: 18M01ABT045, Name: Kelvin Mutua Babu
# Project CSC421 - Developing a Sentiment Analyzer 

# Required Libraries
import pandas as pd  # Importing pandas library for data manipulation
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data into train and test sets
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TfidfVectorizer for feature extraction
from sklearn.naive_bayes import MultinomialNB  # Importing Multinomial Naive Bayes classifier
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score  # Importing evaluation metrics

# Selected Starbucks Corporation as the Multi-National Company (MNC)
# Collecting relevant reviews from Yelp, Trustpilot, and Google My Business
# Using reviews from various sources helps in capturing a diverse range of sentiments


# Extracted Reviews
reviews_data = {
"Review": [
    "Love the cozy ambiance and tasty coffee.",
    "The service was slow and the latte was too bitter for my liking.",
    "Great place to relax and enjoy a cup of coffee.",
    "Disappointed with the small portion sizes and overpriced drinks.",
    "The barista was friendly and made excellent recommendations.",
    "The coffee shop has a great atmosphere and friendly staff.",
    "The pastries are delicious and the coffee is top-notch.",
    "The wait time for drinks was unbearable and the staff seemed disorganized.",
    "Cozy place with a wide selection of drinks and a welcoming environment.",
    "The prices are too high for the quality of the coffee and service.",
    "I always look forward to visiting this coffee shop. The coffee is consistently good.",
    "The cleanliness of the coffee shop leaves much to be desired.",
    "The staff is always courteous and the service is efficient.",
    "Poor quality coffee and unprofessional staff.",
    "The coffee shop has a cozy and comfortable atmosphere.",
    "The lack of parking around the coffee shop is a major inconvenience.",
    "The coffee shop music is too loud and distracting.",
    "The price of the coffee is reasonable for the quality and taste.",
    "The baristas are very knowledgeable and provide great recommendations.",
    "The limited seating at the coffee shop can be frustrating during peak hours.",
    "The ambiance here is so welcoming and relaxing.",
    "I was disappointed with the coffee; it tasted stale.",
    "The staff seemed overwhelmed and unfriendly.",
    "The selection of pastries is amazing, but the coffee was mediocre.",
    "The prices are a bit high, but the quality of the coffee makes up for it.",
    "The seating area is comfortable, but it gets crowded quickly.",
    "I love the variety of drinks offered here.",
    "The customer service was excellent; the staff was very attentive.",
    "The coffee here is absolutely delicious.",
    "The wait time for my order was longer than expected.",
    "The atmosphere is lively and energetic.",
    "The prices are reasonable, and the portion sizes are generous.",
    "The barista recommended a new drink to me, and it was fantastic.",
    "The decor is modern and stylish.",
    "I found a quiet corner to work in; it's a great place to study.",
    "The staff is always smiling and welcoming.",
    "The coffee shop is conveniently located near my office.",
    "The wifi connection is reliable, which is great for remote work.",
    "I appreciate the eco-friendly practices of this coffee shop.",
    "The pastries are always fresh and delicious.",
    "The coffee tastes burnt; it's disappointing.",
    "The music is too loud; it's hard to have a conversation.",
    "The prices are a bit steep for what you get.",
    "The staff is knowledgeable and helpful.",
    "The coffee shop has a cozy atmosphere; it feels like home.",
    "The service was slow, but the coffee was worth the wait.",
    "The selection of teas is impressive.",
    "The seating area is spacious and comfortable.",
    "The coffee is consistently good; I've never been disappointed.",
    "The staff is friendly and attentive.",
    "The pastries are overpriced for the quality.",
    "The coffee shop is always busy, which is a testament to its popularity.",
    "The atmosphere is vibrant and lively.",
    "The prices are reasonable for the quality of the coffee.",
    "The baristas are skilled and passionate about their craft.",
    "The outdoor seating area is a nice touch.",
    "The coffee is too strong for my taste.",
    "The service was slow, and my order was incorrect.",
    "The decor is charming and eclectic.",
    "The coffee shop has a cozy, rustic vibe.",
    "The staff is efficient and professional.",
    "The pastries are a bit stale.",
    "The coffee shop is a bit out of the way, but it's worth the trip.",
    "The prices are a bit high, but the quality of the coffee justifies it.",
    "The ambiance is warm and inviting.",
    "The staff is knowledgeable about their products.",
    "The coffee shop is always clean and well-maintained.",
    "The music is too loud for my liking.",
    "The prices are competitive, and the portions are generous.",
    "The baristas are friendly and helpful.",
    "The coffee is consistently delicious.",
    "The wait time for my order was minimal.",
    "The atmosphere is cozy and intimate.",
    "The pastries are fresh and tasty.",
    "The coffee shop is conveniently located near public transportation.",
    "The wifi connection is fast and reliable.",
    "The prices are affordable, and the portions are generous.",
    "The staff is friendly and attentive.",
    "The coffee shop is always bustling with activity.",
    "The music selection is diverse and enjoyable.",
    "The prices are a bit high, but the quality of the coffee justifies it.",
    "The decor is modern and stylish, with comfortable seating.",
    "The coffee shop has a relaxed, laid-back atmosphere.",
    "The staff is welcoming and accommodating.",
    "The pastries are delicious, especially the croissants.",
    "The coffee shop is a bit crowded, but the vibe is lively.",
    "The prices are a bit high, but the quality of the coffee is unmatched.",
    "The ambiance is cozy and inviting.",
    "The staff is knowledgeable about the menu.",
    "The coffee shop is always clean and well-maintained.",
    "The music is too loud, making it hard to concentrate.",
    "The prices are reasonable, considering the quality of the coffee.",
    "The baristas are friendly and approachable.",
    "The wait time for orders is minimal, even during peak hours, thanks.",
    "The decor is modern and stylish.",
    "The coffee shop has a cozy atmosphere, perfect for catching up with friends.",
    "The staff is welcoming and knowledgeable about their products.",
    "The pastries are delicious, especially the chocolate croissants.",
    "The coffee shop is usually crowded, but it adds to the lively atmosphere.",
    "The prices are reasonable for the quality of the coffee and service.",
    "The ambiance is warm and inviting.",
    "The staff is knowledgeable about the menu.",
    "The coffee shop is always clean and well-maintained.",
    "The music is too loud, making it hard to concentrate.",
    "The prices are reasonable, considering the quality of the coffee.",
    "The baristas are friendly and approachable.",
    "The wait time for orders is minimal, even during peak hours, thanks."
],
"Sentiment": [
    "Positive",  # Love the cozy ambiance and tasty coffee.
    "Negative",  # The service was slow and the latte was too bitter for my liking.
    "Positive",  # Great place to relax and enjoy a cup of coffee.
    "Negative",  # Disappointed with the small portion sizes and overpriced drinks.
    "Positive",  # The barista was friendly and made excellent recommendations.
    "Positive",  # The coffee shop has a great atmosphere and friendly staff.
    "Positive",  # The pastries are delicious and the coffee is top-notch.
    "Negative",  # The wait time for drinks was unbearable and the staff seemed disorganized.
    "Positive",  # Cozy place with a wide selection of drinks and a welcoming environment.
    "Negative",  # The prices are too high for the quality of the coffee and service.
    "Positive",  # I always look forward to visiting this coffee shop. The coffee is consistently good.
    "Negative",  # The cleanliness of the coffee shop leaves much to be desired.
    "Positive",  # The staff is always courteous and the service is efficient.
    "Negative",  # Poor quality coffee and unprofessional staff.
    "Positive",  # The coffee shop has a cozy and comfortable atmosphere.
    "Negative",  # The lack of parking around the coffee shop is a major inconvenience.
    "Negative",  # The coffee shop music is too loud and distracting.
    "Positive",  # The price of the coffee is reasonable for the quality and taste.
    "Positive",  # The baristas are very knowledgeable and provide great recommendations.
    "Negative",  # The limited seating at the coffee shop can be frustrating during peak hours.
    "Positive",  # The ambiance here is so welcoming and relaxing.
    "Negative",  # I was disappointed with the coffee; it tasted stale.
    "Negative",  # The staff seemed overwhelmed and unfriendly.
    "Negative",  # The selection of pastries is amazing, but the coffee was mediocre.
    "Positive",  # The prices are a bit high, but the quality of the coffee makes up for it.
    "Negative",  # The seating area is comfortable, but it gets crowded quickly.
    "Positive",  # I love the variety of drinks offered here.
    "Positive",  # The customer service was excellent; the staff was very attentive.
    "Positive",  # The coffee here is absolutely delicious.
    "Negative",  # The wait time for my order was longer than expected.
    "Positive",  # The atmosphere is lively and energetic.
    "Positive",  # The prices are reasonable, and the portion sizes are generous.
    "Positive",  # The barista recommended a new drink to me, and it was fantastic.
    "Positive",  # The decor is modern and stylish.
    "Positive",  # I found a quiet corner to work in; it's a great place to study.
    "Positive",  # The staff is always smiling and welcoming.
    "Positive",  # The coffee shop is conveniently located near my office.
    "Positive",  # The wifi connection is reliable, which is great for remote work.
    "Positive",  # I appreciate the eco-friendly practices of this coffee shop.
    "Positive",  # The pastries are always fresh and delicious.
    "Negative",  # The coffee tastes burnt; it's disappointing.
    "Negative",  # The music is too loud; it's hard to have a conversation.
    "Negative",  # The prices are a bit steep for what you get.
    "Positive",  # The staff is knowledgeable and helpful.
    "Positive",  # The coffee shop has a cozy atmosphere; it feels like home.
    "Negative",  # The service was slow, but the coffee was worth the wait.
    "Positive",  # The selection of teas is impressive.
    "Positive",  # The seating area is spacious and comfortable.
    "Positive",  # The coffee is consistently good; I've never been disappointed.
    "Positive",  # The staff is friendly and attentive.
    "Negative",  # The pastries are overpriced for the quality.
    "Positive",  # The coffee shop is always busy, which is a testament to its popularity.
    "Positive",  # The atmosphere is vibrant and lively.
    "Positive",  # The prices are reasonable for the quality of the coffee.
    "Positive",  # The baristas are skilled and passionate about their craft.
    "Positive",  # The outdoor seating area is a nice touch.
    "Negative",  # The coffee is too strong for my taste.
    "Negative",  # The service was slow, and my order was incorrect.
    "Positive",  # The decor is charming and eclectic.
    "Positive",  # The coffee shop has a cozy, rustic vibe.
    "Positive",  # The staff is efficient and professional.
    "Negative",  # The pastries are a bit stale.
    "Positive",  # The coffee shop is a bit out of the way, but it's worth the trip.
    "Positive",  # The prices are a bit high, but the quality of the coffee justifies it.
    "Positive",  # The ambiance is warm and inviting.
    "Positive",  # The staff is knowledgeable about their products.
    "Positive",  # The coffee shop is always clean and well-maintained.
    "Negative",  # The music is too loud for my liking.
    "Positive",  # The prices are competitive, and the portions are generous.
    "Positive",  # The baristas are friendly and helpful.
    "Positive",  # The coffee is consistently delicious.
    "Positive",  # The wait time for my order was minimal.
    "Positive",  # The atmosphere is cozy and intimate.
    "Positive",  # The pastries are fresh and tasty.
    "Positive",  # The coffee shop is conveniently located near public transportation.
    "Positive",  # The wifi connection is fast and reliable.
    "Positive",  # The prices are affordable, and the portions are generous.
    "Positive",  # The staff is friendly and attentive.
    "Positive",  # The coffee shop is always bustling with activity.
    "Positive",  # The music selection is diverse and enjoyable.
    "Positive",  # The prices are a bit high, but the quality of the coffee justifies it.
    "Positive",  # The decor is modern and stylish, with comfortable seating.
    "Positive",  # The coffee shop has a relaxed, laid-back atmosphere.
    "Positive",  # The staff is welcoming and accommodating.
    "Positive",  # The pastries are delicious, especially the croissants.
    "Positive",  # The coffee shop is a bit crowded, but the vibe is lively.
    "Positive",  # The prices are a bit high, but the quality of the coffee is unmatched.
    "Positive",  # The ambiance is cozy and inviting.
    "Positive",  # The staff is knowledgeable about the menu.
    "Positive",  # The coffee shop is always clean and well-maintained.
    "Negative",  # The music is too loud, making it hard to concentrate.
    "Positive",  # The prices are reasonable, considering the quality of the coffee.
    "Positive",  # The baristas are friendly and approachable.
    "Positive",  # The wait time for orders is minimal, even during peak hours, thanks.
    "Positive",  # The decor is modern and stylish.
    "Positive",  # The coffee shop has a cozy atmosphere, perfect for catching up with friends.
    "Positive",  # The staff is welcoming and knowledgeable about their products.
    "Positive",  # The pastries are delicious, especially the chocolate croissants.
    "Positive",  # The coffee shop is usually crowded, but it adds to the lively atmosphere.
    "Positive",  # The prices are reasonable for the quality of the coffee and service.
    "Positive",  # The ambiance is warm and inviting.
    "Positive",  # The staff is knowledgeable about the menu.
    "Positive",  # The coffee shop is always clean and well-maintained.
    "Negative",  # The music is too loud, making it hard to concentrate.
    "Positive",  # The prices are reasonable, considering the quality of the coffee.
    "Positive",  # The baristas are friendly and approachable.
    "Positive",  # The wait time for orders is minimal, even during peak hours, thanks.
]
}

# Sentiments
sentiments = reviews_data['Sentiment']

# Data Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews_data['Review'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# Training the classifiers
nb_classifier = MultinomialNB()
svm_classifier = SVC(kernel='linear')
lr_classifier = LogisticRegression(max_iter=1000)

nb_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
lr_classifier.fit(X_train, y_train)

# Making predictions
nb_preds = nb_classifier.predict(X_test)
svm_preds = svm_classifier.predict(X_test)
lr_preds = lr_classifier.predict(X_test)

# Evaluating the models
nb_accuracy = accuracy_score(y_test, nb_preds)
nb_precision = precision_score(y_test, nb_preds, average='weighted', zero_division=0)
nb_recall = recall_score(y_test, nb_preds, average='weighted', zero_division=0)

svm_accuracy = accuracy_score(y_test, svm_preds)
svm_precision = precision_score(y_test, svm_preds, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, svm_preds, average='weighted', zero_division=0)

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_precision = precision_score(y_test, lr_preds, average='weighted', zero_division=0)
lr_recall = recall_score(y_test, lr_preds, average='weighted', zero_division=0)

# Displaying the evaluation metrics
print("Multinomial Naive Bayes Classifier:")
print(f"Accuracy: {nb_accuracy}")
print(f"Precision: {nb_precision}")
print(f"Recall: {nb_recall}\n")

print("Support Vector Classifier:")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}\n")

print("Logistic Regression Classifier:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")

# Accepting a new review or message from the user
new_review = input("\nEnter a new review or message: ")

# Vectorizing the new review/message
new_review_vectorized = vectorizer.transform([new_review])

# Predicting the sentiment of the new review/message using all three classifiers
nb_prediction = nb_classifier.predict(new_review_vectorized)
svm_prediction = svm_classifier.predict(new_review_vectorized)
lr_prediction = lr_classifier.predict(new_review_vectorized)

# Displaying the predicted sentiment
print("\nPredicted Sentiment:")
print("Multinomial Naive Bayes Classifier:", nb_prediction[0])
print("Support Vector Classifier:", svm_prediction[0])
print("Logistic Regression Classifier:", lr_prediction[0])