import re
import nltk

dict_accents = {
    'è': 'e',
    'é': 'e',
    'à': 'a',
    'ì': 'i',
    'ò': 'o',
    'ó': 'o',
    'ù': 'u',
}

dict_punctuation = {
    '.': ' ',
    "'": ' ',
    '-': ' ',
    '_': ' ',
    ';': ' ',
}


def remove_accents(tweet):
    new_tweet = []
    for c in tweet:
        if c in dict_accents:
            new_tweet.append(dict_accents[c])
        else:
            new_tweet.append(c)
    return "".join(new_tweet)


def remove_punctuation(tweet):
    new_tweet = []
    for c in tweet:
        if c in dict_punctuation:
            new_tweet.append(dict_punctuation[c])
        else:
            new_tweet.append(c)
    return "".join(new_tweet)


def only_one_unique_letter(token):
    """
    Returns True if the token contains only one unique letter (eg. aaaaa)
    """
    if len(set(token)) == 1:
        return True
    return False


def preprocess_tweet(tweet):

    tweet = tweet.lower()
    tweet = re.sub('https://t\.co/\w+', ' ', tweet)  # removes the links
    tweet = re.sub('\d+', '', tweet)
    tweet = remove_accents(tweet)
    tweet = remove_punctuation(tweet)
    tokens = nltk.word_tokenize(tweet)

    return " ".join([token for token in tokens if not only_one_unique_letter(token)])


