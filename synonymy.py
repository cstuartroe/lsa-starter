from nltk.corpus import wordnet

# this just shows some featurs of wordnet - synonym sets, definitions, different senses of words, examples sentences
for synset in wordnet.synsets('dog'):
    print([str(lemma.name()) for lemma in synset.lemmas()])
    print(synset.definition())
    if len(synset.examples()) > 0:
        print(synset.examples()[0])
    print()

similarity_pairs = [('dog','cat'),('dog','lemonade'),('small','little'),('small','large')]

# this measures the similarity of the first definitions for each word
def naive_similarity(word0,word1):
    synset0 = wordnet.synsets(pair[0])[0]
    synset1 = wordnet.synsets(pair[1])[0]
    defn0 = synset0.definition()
    defn1 = synset1.definition()
    similarity = synset0.path_similarity(synset1)
    
    return similarity, defn0, defn1

# this finds the pair of definitions with the highest similarity
def max_similarity(word0,word1):
    highest_similarity = 0
    most_similar_defn0 = ""
    most_similar_defn1 = ""
    for synset0 in wordnet.synsets(word0):
        for synset1 in wordnet.synsets(word1):
            similarity = synset0.path_similarity(synset1)
            if (similarity is not None) and (similarity > highest_similarity):
                highest_similarity = similarity
                most_similar_defn0 = synset0.definition()
                most_similar_defn1 = synset1.definition()
                
    return highest_similarity, most_similar_defn0, most_similar_defn1

# just to print it nicely
def similarity_display(similarity, defn0, defn1):
    print(pair[0] + ": " + defn0)
    print(pair[1] + ": " + defn1)
    print("These terms have a similarity score of " + str(similarity))    

# illustrating the difference between naive and max similarity
for pair in similarity_pairs:
    print("Naive Similarity")
    similarity_display(*naive_similarity(pair[0],pair[1]))
    print("Maximum Similarity")
    similarity_display(*max_similarity(pair[0],pair[1]))
    print()
