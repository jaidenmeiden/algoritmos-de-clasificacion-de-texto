#!/usr/bin/env python
# coding: utf-8

# # Entrenando un Modelo Markoviano Latente (HMM)

# ## Corpus de español: 
# 
# * AnCora | Github: https://github.com/UniversalDependencies/UD_Spanish-AnCora
# 
# * usamos el conllu parser para leer el corpus: https://pypi.org/project/conllu/
# 
# * Etiquetas Universal POS (Documentación): https://universaldependencies.org/u/pos/

# In[9]:


#@title dependencias previas
get_ipython().system('pip install conllu')
get_ipython().system('git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora.git')


# In[10]:


#@title leyendo el corpus AnCora
# Función que permite la data en formato conllu
from conllu import parse_incr 
wordList = []
data_file = open("UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
for tokenlist in parse_incr(data_file):
    print(tokenlist.serialize())


# In[14]:


#@title Estructura de los tokens etiquetados del corpus
tokenlist[1]


# In[12]:


tokenlist[1]['form']+'|'+tokenlist[1]['upos']


# ## Entrenamiento del modelo - Calculo de conteos:
# Lo siguiente es un elemento diccionario que se refiere a cuantas veces aparece un ``tag``
# * tags (tags) `tagCountDict`: $C(tag)$
# 
# Lo siguiente es un elemento diccionario que son las emisiones que se refiere a cuantas veces dado un ``tag`` le corresponde un ``word``
# * emisiones (word|tag) `emissionProbDict`: $C(word|tag)$
# 
# Lo siguiente es un elemento diccionario que son las transiciones que se refiere a cuantas veces dado un ``tag`` previo le corresponde un ``tag`` en la posición siguiente
# * transiciones (tag|prevtag) `transitionDict`: $C(tag|prevtag)$

# In[19]:


tagCountDict = {} 
emissionDict = {}
transitionDict = {}

tagtype = 'upos'
data_file = open("UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")

# Calculando conteos (pre-probabilidades)
for tokenlist in parse_incr(data_file):
    prevtag = None #Tag previo para el primer token
    for token in tokenlist:

        # C(tag)
        tag = token[tagtype]
        if tag in tagCountDict.keys():
            tagCountDict[tag] += 1
        else:
            tagCountDict[tag] = 1

        # C(word|tag) -> probabilidades emision
        wordtag = token['form'].lower()+'|'+token[tagtype] # (word|tag)
        if wordtag in emissionDict.keys():
            emissionDict[wordtag] = emissionDict[wordtag] + 1
        else:
            emissionDict[wordtag] = 1

        #  C(tag|tag_previo) -> probabilidades transición
        if prevtag is None:
            prevtag = tag
            continue #Salta a la siguiente iteración

        transitiontags = tag+'|'+prevtag
        if transitiontags in transitionDict.keys():
            transitionDict[transitiontags] = transitionDict[transitiontags] + 1
        else:
            transitionDict[transitiontags] = 1
        prevtag = tag
    

#tagCountDict
#emissionDict
#transitionDict


# ## Entrenamiento del modelo - calculo de probabilidades
# * probabilidades de transición:
# $$P(tag|prevtag) = \frac{C(prevtag, tag)}{C(prevtag)}$$
# 
# * probabilidades de emisión:
#  $$P(word|tag) = \frac{C(word|tag)}{C(tag)}$$

# In[23]:


transitionProbDict = {} # matriz A
emissionProbDict = {} # matriz B

# transition Probabilities 
for key in transitionDict.keys():
    tag, prevtag = key.split('|')
    if tagCountDict[prevtag]>0:
        transitionProbDict[key] = transitionDict[key]/(tagCountDict[prevtag])
    else:
        print(key)

# emission Probabilities 
for key in emissionDict.keys():
    word, tag = key.split('|')
    if emissionDict[key]>0:
        emissionProbDict[key] = emissionDict[key]/tagCountDict[tag]
    else:
        print(key)


# In[24]:


transitionProbDict


# In[25]:


transitionProbDict['ADJ|ADJ']


# In[ ]:


emissionProbDict


# In[ ]:


emissionProbDict['poderío|NOUN']


# ## Guardar parámetros del modelo

# In[ ]:


import numpy as np
np.save('transitionHMM.npy', transitionProbDict)
np.save('emissionHMM.npy', emissionProbDict)
transitionProbdict = np.load('transitionHMM.npy', allow_pickle='TRUE').item()
transitionProbDict['ADJ|ADJ']


# In[ ]:





# In[26]:





# In[27]:


emissionProbDict['poderío|NOUN']


# ## Guardar parámetros del modelo

# In[28]:


import numpy as np
np.save('transitionHMM.npy', transitionProbDict)
np.save('emissionHMM.npy', emissionProbDict)
transitionProbdict = np.load('transitionHMM.npy', allow_pickle='TRUE').item()
transitionProbDict['ADJ|ADJ']


# In[ ]:




