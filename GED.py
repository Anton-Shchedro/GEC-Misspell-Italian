import re
from collections import defaultdict
from tqdm import tqdm
from utils import xnor, split_in_sentence
from EliCoDe_GED import elicode_in

import spacy

nlp = spacy.load("it_core_news_lg")


def each_word(text):
    # split string in words and non-alphabetical sharacters
    words = re.split(r'\s|([^\w])', text)
    return [w for w in words if not w is None and w != '']


def n_gram(tokens, n):
    # create list of n-grams from list of tokens
    # ['a','b','c','a'] -> ['a b', 'b c', 'c d', 'd a']
    if len(tokens) < n:
        l = [''] * (n - len(tokens))
        return tokens + l
    l = []
    for i in range(len(tokens) - n + 1):
        l.append(' '.join(tokens[i:i + n]))
    return l


def two_gram(text):
    lis = each_word(text)
    return n_gram(lis, 2)


def n_gram_BOW(text, n, bow):
    # From a sentence extract n-gram and update bag of n-grams (in case of n=1 Bag of Words)
    lis = each_word(text)
    for token in n_gram(lis, n):
        bow[token] += 1


def create_BOW_corpus(df, n, colum_name='text'):
    # Create a bag of n-grams and for each text in dataframe update it.
    bow = defaultdict(lambda: 0)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        n_gram_BOW(row[colum_name].lower(), n, bow)
    return sorted(bow.items(), key=lambda item: item[1], reverse=True)  # sorted in descending order by absolute
    # frequency in corpus


def in_BOW(gramm, bow):
    # check if n-gram in BOW
    # return True if in BOW
    for k,_ in bow:
        if gramm == k:
            return True
    return False


def zero_search(n_gramm, bow):
    # check list of n-grams in bows. Results are in order of input list.
    result = []
    pbar = tqdm(total=len(n_gramm), unit="gramm", unit_scale = False, position=0)
    for gramm in n_gramm:
        result.append(in_BOW(gramm.lower(), bow))
        pbar.update(1)
    pbar.close()
    return result


def df_text_to_lemma(df, colum_name='text'):
    l_df = df.copy()
    for index, row in tqdm(l_df.iterrows(), total = len(l_df)):
        doc = nlp(row[colum_name].lower())
        row[colum_name] = " ".join([token.lemma_ for token in doc])
    return l_df


def replace_apostrophe(text):
    return text.replace("’", "'").replace("‘", "'")


def gt_bio(gt_text, text, print_difference=False):
    check = []
    i = 0
    j = 0
    is_extra_space = False
    extra_word = ''
    while i < len(gt_text) or j < len(text):
        if i < len(gt_text):
            left = replace_apostrophe(gt_text[i])
        else:
            left = "_"
        if j < len(text):
            right = replace_apostrophe(text[j])
            if is_extra_space:
                right = extra_word + right
        else:
            right = "_"
        if left == right:
            if left == "_" and (i >= len(gt_text) or j >= len(text)):
                continue
            correct = True
            if is_extra_space:
                correct = False
            is_extra_space = False
        else:
            correct = False
            is_extra_space = False # False in most cases, if True will overwrite in specific case
            if (j+2)<len(text) and (i+1)<len(gt_text) and replace_apostrophe(gt_text[i+1]) == replace_apostrophe(text[j+2]):
                check.append(correct) # skip 1 entry of text_words as it's false
                right = right + ' _ ' + replace_apostrophe(text[j+1])
                j+=1
            elif (j)<len(text) and (i+1)<len(gt_text) and replace_apostrophe(gt_text[i+1]) == replace_apostrophe(text[j]):
                j-=1
                correct = True
            elif (j+1)<len(text) and left.startswith(right) and left.startswith(right + replace_apostrophe(text[j+1])): # pa ro la
                is_extra_space = True
                extra_word = right
                i -= 1
            elif (j+1)<len(text) and not left.startswith(right) and left.startswith(replace_apostrophe(text[j+1])): # _ pa rola
                i -= 1
            elif (j+1)<len(text) and left.startswith(right) and text[j+1] == ' ' and left.startswith(right + replace_apostrophe(text[j+2])): # pa _ ro la
                extra_word = right
                check.append(correct)
                j+=1
                i -= 1
            elif (j+2)<len(text) and not left.startswith(right) and (replace_apostrophe(text[j+1]) in left or replace_apostrophe(text[j+2]) in left): # I ~ ro la
                i -= 1
                is_extra_space = True
                extra_word = right
            elif (j+1)<len(text) and (i+1)<len(gt_text) and replace_apostrophe(gt_text[i+1]) == replace_apostrophe(text[j+1]):
                pass
        if print_difference:
            print(i,"|", j,"|", left, "|", right, "|", correct)
        i += 1
        j += 1
        check.append(correct)
    return check


def evaluate(check, result):
    catch = 0
    extra = 0
    miss = 0
    b = xnor(check, result)
    for i in range(len(b)):
        if b[i] == True:
            if result[i] == False:
                catch += 1
            else:
                pass
        else:
            if result[i] == False:
                extra+=1
            else:
                miss+=1
    return catch, extra, miss


def evaluate_text(gt_text: list[str], text: list[str], result: list[bin]):
    # gt_text - list of tokens in Ground True text
    # text - list of tokens in OCR text
    # result - list of binnary of GED predictions
    check = gt_bio(gt_text, text)
    catch, extra, miss = evaluate(check, result)
    return catch, extra, miss


def proceed_zero_search(text_2_gramm, corpus):
    result = zero_search(text_2_gramm, corpus)
    result.insert(0, True)
    r_or = []
    for i in range(len(result) - 1):
        r_or.append(result[i] or result[i + 1])
    return r_or


def two_gramm_corpus(text, corpus):
    # predict errors by zero_search function from corpus bow (or bag of n-grams)
    text_2_gramm = two_gram(text)
    return proceed_zero_search(text_2_gramm, corpus)


def two_gramm_lema_corpus(text, corpus):
    # predict errors by zero_search function from corpus bow (or bag of n-grams) with lemma
    doc = nlp(text.lower())
    text = " ".join([token.lemma_ for token in doc])
    return two_gramm_corpus(text, corpus)


def two_gramm_s_lema_corpus(text, corpus):
    # predict errors by zero_search function from corpus bow (or bag of n-grams) with lemma
    doc = nlp(text.lower())
    lis = [token.lemma_ for token in doc]
    text_2_gramm = n_gram(lis, 2)
    return proceed_zero_search(text_2_gramm, corpus)


def create_s_BOW_corpus(df, n, column_name='text'):
    pbar = tqdm(total=len(df), unit="rows", unit_scale = True, position = 0)
    # create BOW with spacy tokenizer
    bow = defaultdict(lambda: 0)
    list_of_texts = df[column_name].tolist()
    for doc in nlp.pipe(list_of_texts):
        sentence_tokens = [[token.text for token in sent] for sent in doc.sents]
        for sentence in sentence_tokens:
            for token in n_gram(sentence, n):
                bow[token] += 1
        pbar.update(1)
    pbar.close()
    return sorted(bow.items(), key=lambda item: item[1], reverse=True)  # sorted in descending order by absolute


def elicode_spacy_in(text):
    # detect errors in text by using EliCoDe
    doc = nlp(text)
    txt = ""
    for sent in doc.sents:
        for token in sent:
            if token.text == "" or token.text == " ":
                continue
            txt = txt + token.text + ' O O O\n'
        # txt = txt + '\n'
    words, result = elicode_in(txt)
    return words, result


def elidoce_spacy_pred(text):
    # Split text in sentences and detect errors in text by using EliCoDe
    # Split is necessary for big texts!
    sentences = split_in_sentence(text)
    results = []
    to_text_words = []
    i = 0
    j = 0
    sent = ""
    for s in sentences:
        if j < 4:
            if j == 0:
                sent = s
            else:
                sent = sent + "\n\n" + s
            j += 1
        else:
            j = 0
            sent = sent + "\n\n" + s
            words, res = elicode_spacy_in(sent)
            results.append(res)
            to_text_words.append(words)
            #clear_output(wait=True)
            #print(i)
            # print(words)
            # print("=================")
            sent = ""
        i += 1

    if j > 0:
        words, res = elicode_spacy_in(sent)
        results.append(res)
        to_text_words.append(words)

    result = []
    for lis in results:
        result = result + lis
    text_words = []
    for words in to_text_words:
        text_words = text_words + words
    return result, text_words


def label_ner(text, exc=None):
    # replace entities in text with Labels (using Spacy NER)
    # if label of entity is in exc (exceptions), entity will not be replaced.
    if exc is None:
        exc = []
    doc = nlp(text)
    if len(exc) > 0:
        return " ".join([t.text if not t.ent_type_ or (t.ent_type_ in exc) else t.ent_type_ for t in doc])
    return " ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc])


def create_s_BOW_corpus_ner(df, n, exc=None):
    # create BOW with spacy tokenizer and NER recognition and replacement to its labels (see label_ner function)
    if exc is None:
        exc = []
    pbar = tqdm(total=len(df), unit="rows", unit_scale = True, position = 0)
    bow = defaultdict(lambda: 0)
    list_of_texts = df['text'].tolist()
    for doc in nlp.pipe(list_of_texts):
        if len(exc) > 0:
            sentence_tokens = [[t.text if not t.ent_type_ or (t.ent_type_ in exc) else t.ent_type_ for t in sent] for sent in doc.sents]
        else:
            sentence_tokens = [[t.text if not t.ent_type_ else t.ent_type_ for t in sent] for sent in doc.sents]
        for sentence in sentence_tokens:
            for token in n_gram(sentence, n):
                bow[token] += 1
        pbar.update(1)
    pbar.close()
    return sorted(bow.items(), key=lambda item: item[1], reverse=True)  # sorted in descending order by absolute


def each_word_s(text):
    # create list of tokens from text by spacy tokenizer
    words = []
    doc = nlp(text)
    for w in doc:
        words.append(w.text)
    return words


def two_gram_s(text):
    #create list of 2-grams with spacy tokenizer
    lis = each_word_s(text)
    return n_gram(lis, 2)


def two_gramm_corpus_ner(text, corpus):
    # predict errors by zero_search function from corpus bow (or bag of n-grams) with spacy tokenizer
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ != "MISC":
            ents.append((ent.start, ent.end))
    text_2_gramm = two_gram_s(text)
    return proceed_zero_search(text_2_gramm, corpus)