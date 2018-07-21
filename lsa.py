from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy
import os
import string

english_stoptokens = set(stopwords.words('english') + list(string.punctuation))
wordnet_lemmatizer = WordNetLemmatizer()

document_lengths = {}
all_lemmas = set()
all_lemma_counts = {}

# returns the text contents of a document (currently, all lowercased)
def read_document(directory_name, filename):
    full_filename = os.path.join(directory_name,filename)
    print("Reading %s..." % full_filename)
    with open(full_filename,'r',encoding="utf-8") as filehandler:
        content = filehandler.read()
    return content.lower() # .lower() converts everything to lowercase

def increment(counts_dict,word):
    if word in counts_dict:
        counts_dict[word] += 1
    else:
        counts_dict[word] = 1

# gives a dictionary of lemmas with how many documents they occur in
def document_frequencies(_lemma_list,_all_lemma_counts):
    dfs = {}
    for lemma in _lemma_list:
        for document in _all_lemma_counts:
            document_lemma_counts = _all_lemma_counts[document]
            if document_lemma_counts.get(lemma,0) > 0: # if lemma occurs in document >0 times
                increment(dfs,lemma)
    return dfs

def tf_idf(_term_count,_document_length,_term_df,_number_of_documents):
    tf = _term_count/_document_length
    idf = numpy.log(_number_of_documents/_term_df)
    return tf*idf

document_list = os.listdir('documents')

for document in document_list:
    content = read_document('documents',document)
    tokenized = word_tokenize(content)
    document_lengths[document] = len(tokenized)
    print(document + " contains %d total tokens." % len(tokenized))
    
    lemmas = set()
    lemma_counts = {}
    for token in tokenized:
        lemmatized = wordnet_lemmatizer.lemmatize(token)
        if lemmatized not in english_stoptokens:
            lemmas.add(lemmatized)
            increment(lemma_counts,lemmatized)

    all_lemmas.update(lemmas)
    all_lemma_counts[document] = lemma_counts
    print(document + " contains %d distinct non-stop lemmas.\n" % len(lemmas))

print("In all, there are %d distinct non-stop lemmas in your corpus." % len(all_lemmas))
lemma_list = list(all_lemmas)
all_dfs = document_frequencies(lemma_list,all_lemma_counts)

# this will be a grid of how many times each word appears in each document
raw_counts = numpy.zeros((len(document_list),len(lemma_list)),dtype=int)
# this is a grid of tf-idf scores
tf_idf_counts = numpy.zeros((len(document_list),len(lemma_list)),dtype=float)
# you don't need both, i'm giving both and you can pick what works best
for i in range(len(document_list)):
    document = document_list[i]
    for j in range(len(lemma_list)):
        lemma = lemma_list[j]
        raw_count = all_lemma_counts[document].get(lemma,0) # the number of times that lemma occurs inside document
        raw_counts[i,j] = raw_count

        document_length = document_lengths[document]
        term_df = all_dfs[lemma]
        tf_idf_count = tf_idf(raw_count,document_length,term_df,len(document_list))
        tf_idf_counts[i,j] = tf_idf_count

