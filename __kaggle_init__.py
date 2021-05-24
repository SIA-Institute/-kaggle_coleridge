# Social Impact Analytics Institute Kaggle Coleridge competition initialization file


# Kaggle cleaning function
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

#-------------

# <BEN'S CODE>

def walk_directory(directory):
    import os
    
    iterator = 0
    file_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('.'):
                pass
            else:
                pathway = os.path.join(root, file)
                pathway = os.path.realpath(pathway)# added from Jeanna for Windows to read
                file_list.append(pathway)
                iterator += 1
                
    return file_list


def determine_os():
    import platform
    system = platform.system()
    if system == 'Darwin':
        os_system = 'Mac'
    elif system == 'Windows':
        os_system = 'Windows'
        
    return os_system


def walk_mac_directory_json(directory):
    import os
    iterator = 0
    file_list = []

    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.startswith('.'):
                pass
            elif file.endswith('json'):
                pathway = os.path.join(root, file)
                file_list.append(pathway)
            else:
                pass
            
    return file_list


def walk_pc_directory_json(directory):
    import os
    
    file_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('.'):
                pass
            elif file.endswith('json'):
                pathway = os.path.realpath(pathway)# added from Jeanna for Windows to read
                file_list.append(pathway)
            else:
                pass
                
                
    return file_list

# </BEN'S CODE>

#---------------

# <ELISA'S CODE>

#Imports and downloads
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import re
from nltk import word_tokenize
import en_core_web_sm
import numpy as np
nlp = en_core_web_sm.load()

#Helper functions for pattern search
def combine(sentence): #replaces all pos tags with named entities where possible
    doc = nlp(sentence)
    ner_words, labels, starts = [], [], []
    for ent in doc.ents: 
        ner_words.append(ent.text)
        labels.append(ent.label_)
        starts.append(ent.start_char)
    pos_words, pos_tags = two_list_spacy_pos_tags(sentence)
    for i in range(len(ner_words)): #replace pos words with entities if possible
        split_ner_word = spacy_pos_words(ner_words[i])
        len_ner = len(split_ner_word)
        ind = sublist_index(split_ner_word, pos_words)
        for j in range(len_ner):
            pos_words.pop(ind)
            pos_tags.pop(ind)
        pos_words.insert(ind, ner_words[i])
        pos_tags.insert(ind, labels[i])
    return pos_words, pos_tags


def two_list_spacy_pos_tags(sentence): #returns ([words], [pos tags])
    tagged_sentence = nlp(sentence)
    return ([str(word) for word in tagged_sentence], [str(word.tag_) for word in tagged_sentence])


def spacy_pos_words(sentence): #tokenizes words based on spacy pos tagging
    tagged_sentence = nlp(sentence)
    return [str(word) for word in tagged_sentence]


def sublist_index(sublist,l): #returns index of the first component of a sublist in a list
    for ind in (i for i,e in enumerate(l) if e == sublist[0]):
        if l[ind:ind + len(sublist)] == sublist: return ind
    return False


def init_enc_dict(): #creates dictionary relating a pos/ner in all_tags to a character in alpha
    all_tags = ['WORK_OF_ART', 'CARDINAL', 'QUANTITY', 'LANGUAGE', 'ORDINAL', 'PERCENT', 'PRODUCT',
                'PERSON', 'MONEY', 'EVENT', 'PRP\$', 'NNPS', 'TIME', 'DATE', 'NORP', '-LRB-', '-RRB-',
                'POS', 'SYM', 'NNS', 'NNP', 'WRB', 'PDT', 'JJS', 'JJR', 'NFP', 'VBD', 'RBS', 'RBR',
                'WDT', 'VBN', 'WP\$', 'AFX', 'HYPH', 'NIL', 'ADD', 'GW', 'XX', 'BES', 'HVS', 'PRP',
                'VBP', 'VBZ', 'VBG', 'LAW', 'LOC', 'GPE', 'SP', 'ORG', 'FAC', 'UH', 'FW', 'IN', 'DT',
                'NN', 'WP', "''", 'RP', 'MD', 'CD', 'EX', 'CC', 'RB', 'VB', 'TO', 'LS', '``', '""',
                '--', 'JJ', ',', ':', '\.', '\?', '\$', '#'] #includes NER and POS
    alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~`@<>;=áàäåāèé'
    #list of characters that will serve as replacements for components of all_tags
    tag_dict = {}
    for i in range(len(all_tags)):
        tag_dict[all_tags[i]] = alpha[i]
    return tag_dict


def enc(x): #encrypts input using tag_dict
    tag_dict = init_enc_dict()
    for i in tag_dict.keys():
        x = re.sub('_' + i + '_', tag_dict[i], x)
    return x


#Pattern Search
'''
Overview:
Returns words corresponding to pos/ner pattern in sentence containing search word.

Parameters/How to use:
text - string; search_terms - list of word(s); pattern - regex pattern with _pos/ner_ i.e.
'(_PERSON_(((_,__CC_)|_CC_|_,_)_PERSON_)+)'; break_words - list of word(s) that limit searching
distance; distance - how many ner/pos to look behind and in front of search term for (assuming no
break_words or sentence break); search_pos - list of possible pos/ner for search_words.

How it works:
The text is tokenized into sentences and the code looks for each of the search terms
in each sentence. If a search term is present and is the correct pos (dictated by search_pos),
then the function creates a list of ner/pos tags for each word/entinty in the sentence. The
function then determines the range in which to look for the pattern by determining the break word
closest to the word from the front and the back within the distance parameter. The code then encrypts
the pattern and the sentence's pos tags and uses regex to determine if the pattern is present in the
sentence within the distance previously determined by the break_words and/or distance parameter. If
it is, the function adds the words corresponding to the pattern to a list to return after all
sentences have been checked for the search_terms.
'''

def pattern_search(text, search_terms, pattern, break_words = [], distance = 5, search_pos = 'none'):
# code currently assumes valid pos or ner tags have been inputted
    sentences = sent_tokenize(text)
    all_finds = []

    for sentence in sentences:
        for search_term in search_terms:
            if re.search(r'\b' + search_term.lower() + r'\b', sentence.lower()): #if search_word in sentence
                pos_words, pos_tags = combine(sentence) #combine NER with POS

                search_term = spacy_pos_words(search_term.lower())
                lower_pos_words = [i.lower() for i in pos_words]
                lower_word_arr = np.array(lower_pos_words) #array of all words or enitities
                word_ind = sublist_index(search_term, lower_pos_words)
                if search_pos != 'none' and pos_tags[word_ind] not in search_pos: print('yeer')
                
                break_ind = [] #indices of all break_words
                for i in break_words:
                    if i.lower() in lower_word_arr: break_ind += list(np.where(lower_word_arr == i.lower())[0])

                break_ind.sort()
                stop_before, stop_after = max(0, word_ind - distance), min(word_ind + distance, len(pos_words))
                x = 0

                while x < len(break_ind) and word_ind > break_ind[x]:
                    stop_before = break_ind[x] + 1 #index of break word closest to search_word preceding + 1
                    x += 1
                x = len(break_ind) - 1
                while  x >= 0 and word_ind < break_ind[x]:
                    stop_after = break_ind[x] #index of break word closest to search_word following
                    x -= 1
                
                enc_pos = enc('_' + '__'.join(pos_tags) + '_')
                indices = [(m.start(0), m.end(0)) for m in re.finditer(enc(pattern), enc_pos)]
                chosen = []
                
                for i in indices:
                    if i[0] < stop_after and i[0] > stop_before:
                        chosen = i
                        break
                if chosen != []:
                    words = ' '.join(pos_words[chosen[0]:chosen[1]])
                    if words.lower() == ' '.join(search_term).lower(): continue
                    all_finds.append(words)

    return all_finds if all_finds!= '' else None

def get_citations(text):
    x = re.finditer('\((\w| |&|,)+, (19|20)\d\d(,.*?\)|;.*?\)|\))', text)
    all_citations = []
    for i in x:
        all_citations.append(text[i.start(): i.end()])
    return all_citations

# </ELISA'S CODE>
