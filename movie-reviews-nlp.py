import nltk
import random
from nltk.corpus import movie_reviews
import pprint
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")

all_words = []
for w in movie_reviews.words():
    if (w not in stop_words) and (w not in string.punctuation):
        all_words.append(w.lower())
        
all_words = nltk.FreqDist(all_words)
all_words.most_common(20)

feature_words = list(all_words.keys())[:5000]
print(feature_words[:10])

def find_features(document):
    words = set(document)
    feature = {}
    for w in feature_words:
        feature[w] = (w in words)
    return feature

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
            ]
random.shuffle(documents)

print(documents[0])

feature_sets = [(find_features(rev), category) for (rev, category) in documents]
print(feature_sets[0])
len(feature_sets)

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)

## Testing it's accuracy
print("Naive bayes classifier accuracy percentage : ", (nltk.classify.accuracy(classifier, testing_set))*100)

review_spirit = '''
Spirited Away' is the first Miyazaki I have seen, but from this stupendous film I can tell he is a master storyteller. A hallmark of a good storyteller is making the audience empathise or pull them into the shoes of the central character. Miyazaki does this brilliantly in 'Spirited Away'. During the first fifteen minutes we have no idea what is going on. Neither does the main character Chihiro. We discover the world as Chihiro does and it's truly amazing to watch. But Miyazaki doesn't seem to treat this world as something amazing. The world is filmed just like our workaday world would. The inhabitants of the world go about their daily business as usual as full with apathy as us normal folks. Places and buildings are not greeted by towering establishing shots and majestic music. The fact that this place is amazing doesn't seem to concern Miyazaki.
 
What do however, are the characters. Miyazaki lingers upon the characters as if they were actors. He infixes his animated actors with such subtleties that I have never seen, even from animation giants Pixar. Twenty minutes into this film and I completely forgot these were animated characters; I started to care for them like they were living and breathing. Miyazaki treats the modest achievements of Chihiro with unashamed bombast. The uplifting scene where she cleanses the River God is accompanied by stirring music and is as exciting as watching gladiatorial combatants fight. Of course, by giving the audience developed characters to care about, the action and conflicts will always be more exciting, terrifying and uplifting than normal, generic action scenes. 
'''
print(review_spirit)

words = word_tokenize(review_spirit)
words = find_features(words)

classifier.classify(words)