import re
import numpy as np

def split_in_sentence(old_text):
    """
    split text in sentences (does not split if 'N.1234' or similar)
    """
    return(re.split(r"(?<=\.\s)(?![0-9])", old_text))

def find(s, ch):
    """
    find index of char ch in string s
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]

def ranges(myarray):
    """
    transform array of indexes in list of tuples of consecutive numbers
    ex: [1,2,3,5,8,9] -> [(1,3),5,(8,9)]
    """
    if myarray == []: return []
    sequences = np.split(myarray, np.array(np.where(np.diff(myarray) > 1)[0]) + 1)
    l = []
    for s in sequences:
        if len(s) > 1:
            l.append((np.min(s), np.max(s)))
        else:
            l.append(s[0])
    return l

def remove_spaces(old, sub_space):
    """
    remove chars by index in 'sub_space' from string 'old'
    """
    j = 0
    sub_index = len(old)
    for i in sub_space:
        if i < sub_index:
            sub_index = i-1
        old = old[:(i-j)] + old[(i-j+1):]
        j += 1
    return old

def remove_space(old, index):
    """
    remove chars by index in 'sub_space' from string 'old'
    """
    return old[:(index)] + old[(index+1):]

def insert_space(old, index):
    """
    insert char ' ' by index in 'sub_space' to string 'old'
    """
    return old[:(index)] + ' ' + old[(index):]

def replace_word(old, index, lenght, sub_new):
    """
    replace substring from 'old' string starting by 'index' and with 'lenght' lenght by 'sub_new'
    """
    changed_start =  index
    changed_end = changed_start + lenght
    return old[:changed_start] + sub_new + old[changed_end:]

def levenshtein(s1, s2):
    """
    calculate levenshtein distance between strings s1 and s2
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def xnor(a,b):
    result = []
    if len(a) < len(b):
        for i in range(len(b)-len(a)):
            a.append(False)
    elif len(a) > len(b):
        for i in range(len(a)-len(b)):
            b.append(False)
    for i in range(len(a)):
        result.append(a[i] == b[i])
    return result
    #return ((a and b) or ((not a) and (not b)))


def calc_prf05(row, name):
    #calculate Precision, Recall and F0.5 for each text (one row of df)
    try:
        c = name+"_catch" #TP
        e = name+"_extra" #FP
        m = name+"_miss" #FN
        if c!=0 or e!=0:
            p = float(row[c]) / float(row[c] + row[e])
        else:
            p = 1
        if c!=0 or m!=0:
            r = float(row[c]) / float(row[c] + row[m])
        else:
            r=1
        if p==0 and r==0:
            f05 = 0
        else:
            f05 = 1.25*((p * r) / ((0.25*p) + r))
    except ZeroDivisionError:
        p = 0.0
        r = 0.0
        f05 = 0.0
    return p,r,f05


def df_calc_prf05(df, name):
    # calculate Precision, Recall and F0.5 as a sum of TP, FP and FN in whole df
    c = name+"_catch" #TP
    e = name+"_extra" #FP
    m = name+"_miss" #FN
    p = df[c].sum() / (df[c].sum() + df[e].sum())
    r = df[c].sum() / (df[c].sum() + df[m].sum())
    f05 = 1.25*((p * r) / ((0.25*p) + r)) #in theory in whole df prob. that TP == 0 is low
    return p,r,f05


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