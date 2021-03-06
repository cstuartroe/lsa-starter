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

# term frequency-inverse document frequency, a shmancy statistic measuring a word's "importance" in a corpus
def tf_idf(_term_count,_document_length,_term_df,_number_of_documents):
    tf = _term_count/_document_length
    idf = numpy.log(_number_of_documents/_term_df)
    return tf*idf

# this one's tough to explain. read the readme.
def reduced_column_count(_sigma,_min_coverage):
    _sigma = list(_sigma)
    assert(_min_coverage > 0 and _min_coverage <= 1)
    for i in range(1,len(_sigma)+1):
        coverage = sum(_sigma[:i])/sum(sigma)
        if coverage >= _min_coverage:
            return i

# this is a naive way to categorize words - just see in which dimension they score the highest
# if you're really trying to group words by topic you might want to do more advanced vector stuff, like k-means sorting
def categorize_words(_v,_lemma_list):
    categories = [[] for i in range(v.shape[0])]
    for i in range(v.shape[1]):
        column = list(numpy.absolute(v[:,i]))
        category = column.index(max(column))
        categories[category].append(_lemma_list[i])
    return categories

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
            lemmas.add(lemmatized) # adds the new lemma to the set, if it's not already included
            increment(lemma_counts,lemmatized)

    all_lemmas.update(lemmas) # adds any new words from current document to the set of all lemmas
    all_lemma_counts[document] = lemma_counts
    print(document + " contains %d distinct non-stop lemmas.\n" % len(lemmas))

print("In all, there are %d distinct non-stop lemmas in your corpus." % len(all_lemmas))
lemma_list = list(all_lemmas) # converting from a set (unordered) to a list (ordered)
all_dfs = document_frequencies(lemma_list,all_lemma_counts) # a dictionary giving how many documents each word appears in

# this will be a grid of how many times each word appears in each document
raw_counts = numpy.zeros((len(document_list),len(lemma_list)),dtype=int)
# this is a grid of tf-idf scores
tf_idf_counts = numpy.zeros((len(document_list),len(lemma_list)),dtype=float)
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

u, sigma, v = numpy.linalg.svd(tf_idf_counts)
min_coverage = .9 # a proportion of the variance you want to keep. anywhere .75-.95 is reasonable.
rcc = reduced_column_count(sigma,min_coverage)
print("To keep %d%% percent coverage, %d columns are retained." % (round(min_coverage*100),rcc))
u = u[:,:rcc]
sigma = sigma[:rcc]
v = v[:rcc,:]

approx = numpy.round(numpy.matmul(numpy.matmul(u,numpy.diag(sigma)),v),decimals=6)
tf_idf_counts = numpy.round(tf_idf_counts,decimals=6)

word_categories = categorize_words(v,lemma_list)
for category in word_categories:
    print(category[:10])
