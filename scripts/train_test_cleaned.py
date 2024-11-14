### Importing dependencies.

import os, math, operator, csv, random, pickle,re

import gc

from nltk.tokenize import TweetTokenizer

from unidecode import unidecode

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

### DEfining columns, submission file, and special characters that need to be removed.

TEXT_COLUMN = 'comment_text'
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

#defining paths

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
data_folder = "../input/jigsaw-toxic-comment-classification-challenge/"
pretrained_folder = "../input/"
train_filepath = data_folder + "train.csv.zip"
test_filepath = data_folder + "test.csv.zip"

#path to a submission
submission_path =  data_folder + "submission.csv"

#paths to pretrained dictionaries
hyphens_filepath = "../input/cleaning-dictionaries/hyphens_dictionary.bin"
misspellings_filepath = "../input/cleaning-dictionaries/misspellings_all_dictionary.bin"
merged_filepath = "../input/cleaning-dictionaries/merged_all_dictionary.bin"

hyphens_dict = misspellings_dict = merged_dict = {}
with open(hyphens_filepath, mode='rb') as file: hyphens_dict = pickle.load(file)
with open(misspellings_filepath, mode='rb') as file: misspellings_dict = pickle.load(file)
with open(merged_filepath, mode='rb') as file: merged_dict = pickle.load(file)
    
print(len(hyphens_dict))
print(len(misspellings_dict))
print(len(merged_dict)) 

training_samples_count = 149571
validation_samples_count = 10000

length_threshold = 20000 #We are going to truncate a comment if its length > threshold
word_count_threshold = 900 #We are going to truncate a comment if it has more words than our threshold
words_limit = 310000

#We will filter all characters except alphabet characters and some punctuation
valid_characters = " " + "@$" + "'!?-" + "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
valid_characters_ext = valid_characters + "abcdefghijklmnopqrstuvwxyz".upper()
valid_set = set(x for x in valid_characters)
valid_set_ext = set(x for x in valid_characters_ext)

#List of some words that often appear in toxic comments
#Sorry about the level of toxicity in it!
toxic_words = ["poop", "crap", "prick", "twat", "wikipedia", "wiki", "hahahahaha", "lol", "bastard", "sluts", "slut", "douchebag", "douche", "blowjob", "nigga", "dumb", "jerk", "wanker", "wank", "penis", "motherfucker", "fucker", "fuk", "fucking", "fucked", "fuck", "bullshit", "shit", "stupid", "bitches", "bitch", "suck", "cunt", "dick", "cocks", "cock", "die", "kill", "gay", "jewish", "jews", "jew", "niggers", "nigger", "faggot", "fag", "asshole"]
astericks_words = [('mother****ers', 'motherfuckers'), ('motherf*cking', 'motherfucking'), ('mother****er', 'motherfucker'), ('motherf*cker', 'motherfucker'), ('bullsh*t', 'bullshit'), ('f**cking', 'fucking'), ('f*ucking', 'fucking'), ('fu*cking', 'fucking'), ('****ing', 'fucking'), ('a**hole', 'asshole'), ('assh*le', 'asshole'), ('f******', 'fucking'), ('f*****g', 'fucking'), ('f***ing', 'fucking'), ('f**king', 'fucking'), ('f*cking', 'fucking'), ('fu**ing', 'fucking'), ('fu*king', 'fucking'), ('fuc*ers', 'fuckers'), ('f*****', 'fucking'), ('f***ed', 'fucked'), ('f**ker', 'fucker'), ('f*cked', 'fucked'), ('f*cker', 'fucker'), ('f*ckin', 'fucking'), ('fu*ker', 'fucker'), ('fuc**n', 'fucking'), ('ni**as', 'niggas'), ('b**ch', 'bitch'), ('b*tch', 'bitch'), ('c*unt', 'cunt'), ('f**ks', 'fucks'), ('f*ing', 'fucking'), ('ni**a', 'nigga'), ('c*ck', 'cock'), ('c*nt', 'cunt'), ('cr*p', 'crap'), ('d*ck', 'dick'), ('f***', 'fuck'), ('f**k', 'fuck'), ('f*ck', 'fuck'), ('fc*k', 'fuck'), ('fu**', 'fuck'), ('fu*k', 'fuck'), ('s***', 'shit'), ('s**t', 'shit'), ('sh**', 'shit'), ('sh*t', 'shit'), ('tw*t', 'twat')]
fasttext_misspelings = {"'n'balls": 'balls', "-nazi's": 'nazis', 'adminabuse': 'admin abuse', "admins's": 'admins', 'arsewipe': 'arse wipe', 'assfack': 'asshole', 'assholifity': 'asshole', 'assholivity': 'asshole', 'asshoul': 'asshole', 'asssholeee': 'asshole', 'belizeans': 'mexicans', "blowing's": 'blowing', 'bolivians': 'mexicans', 'celtofascists': 'fascists', 'censorshipmeisters': 'censor', 'chileans': 'mexicans', 'clerofascist': 'fascist', 'cowcrap': 'crap', 'crapity': 'crap', "d'idiots": 'idiots', 'deminazi': 'nazi', 'dftt': "don't feed the troll", 'dildohs': 'dildo', 'dramawhores': 'drama whores', 'edophiles': 'pedophiles', 'eurocommunist': 'communist', 'faggotkike': 'faggot', 'fantard': 'retard', 'fascismnazism': 'fascism', 'fascistisized': 'fascist', 'favremother': 'mother', 'fuxxxin': 'fucking', "g'damn": 'goddamn', 'harassmentat': 'harassment', 'harrasingme': 'harassing me', 'herfuc': 'motherfucker', 'hilterism': 'fascism', 'hitlerians': 'nazis', 'hitlerites': 'nazis', 'hubrises': 'pricks', 'idiotizing': 'idiotic', 'inadvandals': 'vandals', "jackass's": 'jackass', 'jiggabo': 'nigga', 'jizzballs': 'jizz balls', 'jmbass': 'dumbass', 'lejittament': 'legitimate', "m'igger": 'nigger', "m'iggers": 'niggers', 'motherfacking': 'motherfucker', 'motherfuckenkiwi': 'motherfucker', 'muthafuggas': 'niggas', 'nazisms': 'nazis', 'netsnipenigger': 'nigger', 'niggercock': 'nigger', 'niggerspic': 'nigger', 'nignog': 'nigga', 'niqqass': 'niggas', "non-nazi's": 'not a nazi', 'panamanians': 'mexicans', 'pedidiots': 'idiots', 'picohitlers': 'hitler', 'pidiots': 'idiots', 'poopia': 'poop', 'poopsies': 'poop', 'presumingly': 'obviously', 'propagandaanddisinformation': 'propaganda and disinformation', 'propagandaministerium': 'propaganda', 'puertoricans': 'mexicans', 'puertorricans': 'mexicans', 'pussiest': 'pussies', 'pussyitis': 'pussy', 'rayaridiculous': 'ridiculous', 'redfascists': 'fascists', 'retardzzzuuufff': 'retard', "revertin'im": 'reverting', 'scumstreona': 'scums', 'southamericans': 'mexicans', 'strasserism': 'fascism', 'stuptarded': 'retarded', "t'nonsense": 'nonsense', "threatt's": 'threat', 'titoists': 'communists', 'twatbags': 'douchebags', 'youbollocks': 'you bollocks'}
acronym_words = {} #{"btw":"by the way", "yo": "you", "u": "you", "r": "are", "ur": "your", "ima": "i am going to", "imma": "i am going to", "i'ma":"i am going to", "cos":"because", "coz":"because", "stfu": "shut the fuck up", "wat": "what"}

### Various pattern matchings and pattern replacements

cont_patterns = [
    (r'(W|w)on\'t', r'will not'),
    (r'(C|c)an\'t', r'can not'),
    (r'(I|i)\'m', r'i am'),
    (r'(A|a)in\'t', r'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def split_word(word, toxic_words):
    if word == "":
        return ""
    
    lower = word.lower()
    for toxic_word in toxic_words:
        start = lower.find(toxic_word)
        if start >= 0:
            end = start + len(toxic_word)
            result = " ".join([word[0:start], word[start:end], split_word(word[end:], toxic_words)])
            return result.replace("  ", " ").strip()
    return word

tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
def word_tokenize(sentence):
    sentence = sentence.replace("$", "s")
    sentence = sentence.replace("@", "a")    
    sentence = sentence.replace("!", " ! ")
    sentence = sentence.replace("?", " ? ")
    
    return tknzr.tokenize(sentence)

def replace_url(word):
    if "http://" in word or "www." in word or "https://" in word or "wikipedia.org" in word:
        return ""
    return word

def normalize_by_dictionary(normalized_word, dictionary):
    result = []
    for word in normalized_word.split():
        if word == word.upper():
            if word.lower() in dictionary:
                result.append(dictionary[word.lower()].upper())
            else:
                result.append(word)
        else:
            if word.lower() in dictionary:
                result.append(dictionary[word.lower()])
            else:
                result.append(word)
    
    return " ".join(result)

### Normalizing comments with spacy

from spacy.symbols import nsubj, VERB, dobj
import spacy
nlp = spacy.load('en')

def normalize_comment(comment):
    comment = unidecode(comment)
    comment = comment[:length_threshold]
    
    normalized_words = []
    
    for w in astericks_words:
        if w[0] in comment:
            comment = comment.replace(w[0], w[1])
        if w[0].upper() in comment:
            comment = comment.replace(w[0].upper(), w[1].upper())
    
    for word in word_tokenize(comment):
        #for (pattern, repl) in patterns:
        #    word = re.sub(pattern, repl, word)

        if word == "." or word == ",":
            normalized_words.append(word)
            continue
        
        word = replace_url(word)
        if word.count(".") == 1:
            word = word.replace(".", " ")
        filtered_word = "".join([x for x in word if x in valid_set])
                    
        #Kind of hack: for every word check if it has a toxic word as a part of it
        #If so, split this word by swear and non-swear part.
        normalized_word = split_word(filtered_word, toxic_words)
        normalized_word = normalize_by_dictionary(normalized_word, hyphens_dict)
        normalized_word = normalize_by_dictionary(normalized_word, merged_dict)
        normalized_word = normalize_by_dictionary(normalized_word, misspellings_dict)
        normalized_word = normalize_by_dictionary(normalized_word, fasttext_misspelings)
        normalized_word = normalize_by_dictionary(normalized_word, acronym_words)

        normalized_words.append(normalized_word)
        
    normalized_comment = " ".join(normalized_words)
    
    result = []
    for word in normalized_comment.split():
        if word.upper() == word:
            result.append(word)
        else:
            result.append(word.lower())
    
    #apparently, people on wikipedia love to talk about sockpuppets :-)
    result = " ".join(result)
    if "sock puppet" in result:
        result = result.replace("sock puppet", "sockpuppet")
    
    if "SOCK PUPPET" in result:
        result = result.replace("SOCK PUPPET", "SOCKPUPPET")
    
    return result

### Funcion that reads and cleans the texts.

def read_data_files(train_filepath, test_filepath):
    #read train data
    train = pd.read_csv(train_filepath)


    labels = train[categories].values
    
    #read test data
    test = pd.read_csv(test_filepath)

    test_comments = test["comment_text"].fillna("_na_").values

    #normalize comments
    np_normalize = np.vectorize(normalize_comment)
    comments = train["comment_text"].fillna("_na_").values
    normalized_comments = np_normalize(comments)
    del comments
    gc.collect()

    
    comments = test["comment_text"].fillna("_na_").values
    normalized_test_comments = np_normalize(test_comments)
    del comments
    gc.collect()
       

    print('Shape of data tensor:', normalized_comments.shape)
    print('Shape of label tensor:', labels.shape)
    print('Shape of test data tensor:', normalized_test_comments.shape)
    
    return (labels, normalized_comments, normalized_test_comments)

### Finally we run everything.

labels, x_train, x_test = read_data_files(train_filepath, test_filepath) 

### And save teh new cleaned data into files.

np.save("lables", labels)
np.save("x_train", x_train)
np.save("x_test", x_test)