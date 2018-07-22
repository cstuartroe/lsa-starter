# lsa-starter

I made this starter code so Colin could learn how to do latent semantic analysis.

It makes use of [NTLK](https://www.nltk.org/) and [NumPy](http://www.numpy.org/).

## how to get started

Install Python 3 and download this repository. Use a command line to navigate to the root folder of this project, 
and run `pip install -r requirements.txt`. That should install numpy and nltk.

Try out the code by entering `python lsa.py` or `python synonymy.py`. They'll both give you printouts of what's going on.

## lsa.py

This is a script to perform basic latent semantic analysis. 
I put together a small (~20,000 word) corpus of four texts each in three categories - 
[NPR Health articles](https://www.npr.org/sections/health/), 
[taco recipe blog posts](https://pinchofyum.com/?s=taco), 
and [30 Rock episode transcripts](https://www.springfieldspringfield.co.uk/episode_scripts.php?tv-show=30-rock) -
to test its functionality.

When you run lsa.py, you should see a printout for each document that looks something like:

```
Reading documents\Cauliflower Tacos.txt...
Cauliflower Tacos.txt contains 531 total tokens.
Cauliflower Tacos.txt contains 168 distinct non-stop lemmas.
```

The first number is the number of tokens in the document, which includes words and punctuation marks. 
It's broadly similar to a word count.
The second number is the number of unique words, not including punctuation, 
stopwords (common, mostly meaningless words like "be" and prepositions),
or inflected forms of words (runs, ran, running).
I've included that statistic because, as the script is currently written, that's the input to the LSA algorithm.

Then you'll see something like 

```
To keep 90% percent coverage, 10 columns are retained.
['unused', 'annnnd', 'mash', 'tender', 'though', 'top', 'ready', 'drenching', 'zero', 'top-knot-rocking']
[]
['12', 'th', '2020', 'director', 'society', 'for-profit', 'acquisition', 'dental', 'doctor', 'lemon']
['good', 'enchilada', 'maybe', 'tostada', 'love', 'trifecta', 'true', 'rice', 'spicy', 'embarrassing']
['avocado', 'pan', 'soon', 'coming', 'sing', 'chunk', 'onion', 'promise', 'melissa', 'recommend']
['lawmaker', 'legislative', 'range', 'code', 'successfully', 'write', 'bountiful', 'alliance', 'deployed', 'choice']
[]
['minnesota', 'scooped', 'watson', 'estate', 'plus-size', 'actuary', 'facebook', 'consider', 'criminal', 'may']
['reviewed', '2,000', 'michael', 'enriching', 'treat', 'lindell', 'expanded', 'swarm', 'costlier', 'merola']
['guaranteed', 'lightning', 'pervert', 'object', 'gotten', '307', 'reinforce', 'network', 'answered', 'sense']
```

Each row there corresponds to a single category or words, according to a pretty blunt method of analyzing the LSA output.
Exactly which words you see will vary every time you run the script, due to Python's pseudorandom set hashing.

Let's look inside the script.

### Imports

```
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy
import os
import string
```

As you can see, it imports specific parts of nltk. `word_tokenize` splits up text into individual words and punctuation,
`stopwords` is a package that contains common, unimportant words for several languages,
and WordNetLemmatizer turns inflected forms of words like "running" or "dogs" into uninflected forms like "run" and "dog".
It's these three imports that are at work when you see how many tokens and unique lemmas each document has.

Numpy is a library that contains a lot of linear algebra functions like 
[singular value decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition),
an algorithms that distills high-dimensional datasets into their "principal components", kinda like finding
a line of best fit in multidimensional space. SVD is at the heart of LSA.

os and string are native python libraries. os performs basic functions for working with files and folders, 
and string has a few useful tools for working with strings of text.

```
english_stoptokens = set(stopwords.words('english') + list(string.punctuation))
wordnet_lemmatizer = WordNetLemmatizer()

document_lengths = {}
all_lemmas = set()
all_lemma_counts = {}
```

These lines are basically setting everything up. `english_stoptokens` is the full set of tokens I've chosen to filter out, 
including stopwords and punctuation. `wordnet_lemmatizer` is an instance of `WordNetLemmatizer`.

`document_lengths` and `all_lemma_counts` are dictionaries and `all_lemmas` is a set. 
In Python, dictionaries and sets are two types of unordered data structures. Sets, like mathematical sets,
just contain items without duplicates and in no particular order. Dictionaries contain pairs of "keys" and "values",
and you use the keys to look up corresponding values.

So for instance, `document_lengths` is going to end up looking like

`{'Anna Howard Shaw Day.txt': 4444, 'Beef Tacos.txt': 600, ...}`

and will be used to look up token count for documents by name. `all_lemmas` will look like 

`{'anytime', 'npr', 'mechanism', 'four', ...}`

and contain all the lemmas we're keeping track of for LSA. `all_lemma_counts` will be a dictionary whose
keys are document names and values are dictionaries of words and counts. Like this:

`{'Anna Howard Shaw Day.txt': {'anytime':10, 'npr':0, ...}, 'Beef Tacos.txt': {'anytime':15, 'npr':12, ...}, ...}`

(I made up those numbers. If a recipe for beef tacos mentions NPR twelve times I'm gonna look into who's sponsoring them.)

### helper functions

`read_document` returns the contents of a file, all lowercased. 
`increment` looks up a term in a dictionary of word counts, either adding one to the existing count if the word's already listed,
otherwise creating a new entry and setting its count equal to one.
Hopefully both of those functions are pretty readable.

`document_frequencies` looks up to see how many documents a word occurs inside at least once. 
Its output will be a dictionary that looks like `{'funding': 3, 'entire': 1, 'exactly': 4, ...}`.

`tf_idf` performs a term frequency-inverse dcument frequency calculation for a particular term. 
TF-IDF is the most common statistic to use for LSA, because for whatever reasons it works the best.
The actual math it's doing is pretty simple, but if you're curious why TF-IDF is calculated,
visit [the wikipedia page](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). The basic idea is weighting
words according to how unique and important their appearance is to a document.

I guess I passed off the explanation of `reduced_column_count` to this README. To understand what it's doing you need
to understand the output of singular value decomposition. The purpose of SVD is to take data in an arbitrary vector space and transform
it into a set of new dimensions from most significant possible to least significant. In this way, you can throw out a number of
less significant dimensions and model the shape of the original data very accurately with a smaller number of dimensions.
The implication for LSA is that you can model your big, messy TF-IDF data in a simpler dimensional space, which makes it
easier to work with and throws out some statistical noise. This process of approximation is also what leads to synonyms
or related words getting lumped together.

SVD is pretty tough to explain in text, so here are some great youtube videos:

[Change of basis](https://www.youtube.com/watch?v=P2LTAUO1TdA) - 3Blue1Brown is great and he has a lot of videos about linear algebra.
Change of basis is the general process of describing data in a new set of dimensions. SVD just performs a change of basis so that
the new dimensions are from most significant possible, to least significant.

[SVD](https://www.youtube.com/watch?v=P5mlg91as1c) - Here's a Stanford video about SVD specifically.

SVD essentially takes one matrix and cracks it into three different matrices, which when multiplied together produce the original 
again. The middle of the three matrices, called `sigma`, is full of zeros except a series of numbers along the diagonal, corresponding
to the new dimensions in order of most to least significant. Those numbers tell you how significant each dimension is. If the sum of
the first ten values is 90% as much as the sum of all the values, that tells you that throwing out all but the first ten dimensions
will allow you to keep 90% of the information in the original data.

So, what `reduced_column_count` does is add up all the values in `sigma` and figure out how many you need to keep to retain a 
desired level of variability. 

`categorize_words` splits words into categories based on which of the new dimensions they are furthest from 0 in. This is not a very
good way of categorizing words - you could look into [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
depending on your specific needs.

### the main code

The `for document in document_list:` loop reads each document, tokenizes and lemmatizes it, counts the non-stop tokens, then updates
`all_lemmas` and `all_lemma_counts` so that `all_lemmas` gets any new lemmas added to it and `all_lemma_counts` gets an entry
with that document's lemma counts.

The `for i in range(len(document_list)):` loop goes through every document and fills in the matrix `raw_counts` with just the 
number of times each non-stop lemma occurs in each document and the matrix `tf_idf_counts` with the TF-IDF scores for each
lemma and document. 
The way the script is written right now, `raw_counts` doesn't actually get used, and `tf_idf_counts` is used for LSA.

The last block performs SVD, chucks however many dimensions it can to retain the desired amount of original information, 
and performs that basic word categorization.

## synonymy.py

This script just shows of some of the basic functions of NLTK's WordNet.

The first block shows off various properties and functions of synsets, which are sets of senses for a given word. A "synset" in
WordNet is a unit that has a single definition and meaning but might have several associated synonymous words. Hopefully the
printouts make it pretty clear what's going on.

`naive_similarity` and `max_similarity` both give a similarity score between 0 and 1 for two words. In WordNet, semantic similarity
is measured between synsets, so a synset has to be picked out for each word provided. What's naive about `naive_similarity`
is that it just picks out whichever synset happens to be listed first, while `max_similarity` looks through every meaning for
each of the two words and picks out the pair of definitions with the highest similarity. If two words share any synset,
their similarity score will be 1. Again, hopefully the printouts help clarify what's going on.
