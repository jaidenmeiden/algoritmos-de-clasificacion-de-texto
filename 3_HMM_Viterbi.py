#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np


# In[34]:


# instalacion de dependencias previas
get_ipython().system('pip install conllu')
get_ipython().system('git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora.git')


# # Carga del modelo HMM previamente entrenado

# Cargamos las probabilidades del modelo HMM

# In[35]:


transitionProbdict = np.load('transitionHMM.npy', allow_pickle='TRUE').item()
emissionProbdict = np.load('emissionHMM.npy', allow_pickle='TRUE').item()


# Identificamos las categorias gramaticales 'upos' unicas en el corpus.
# 
# Obtenemos las llaves de la colección de probabilidades de emisión con `emissionProbdict.keys()` y creamos un bucle recorriendo la lista de llaves `[k for k in emissionProbdict.keys()` y de cada llave obtenida captuarmos unicamente la categoría gramatical `k.split('|')[1]` en la segunda posición de la llave. Para que no nos muestre categorías repetidas aplicamos la función `set()`, donde nos debe mostrar **17 registros** según la convención internacional.

# In[36]:


stateSet = set([k.split('|')[1] for k in emissionProbdict.keys()])
stateSet


# Enumeramos las categorias con números para asignar a las columnas (Asignamos un número entero) de la matriz de Viterbi.

# In[37]:


tagStateDict = {}
for i, state in enumerate(sorted(stateSet)):
    tagStateDict[state] = i
tagStateDict


# # Distribucion inicial de estados latentes
# 
# Calculamos distribución inicial de estados

# In[38]:


from conllu import parse_incr 
wordList = []
data_file = open("UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")


# En `initTagStateProb` (Guarda los `\rho_i^{(0)}`, que son los **rhos** del estado **i** en el momento **0**) es donde guardamos la probabilidad de que encuentre una categoría gramatical al principio de una frase en el corpus.

# In[39]:


initTagStateProb = {} # \rho_i^{(0)}
count = 0 # cuenta la longitud del corpus
for tokenlist in parse_incr(data_file):
    count += 1
    tag = tokenlist[0]['upos']
    if tag in initTagStateProb.keys():
        initTagStateProb[tag] += 1
    else:
        initTagStateProb[tag] = 1

for key in initTagStateProb.keys():
    initTagStateProb[key] /= count

initTagStateProb


# Verificamos que la suma de las probabilidades es 1 (100%)
# 
# En la forma **NO** elegante, sumamos los valores de la colección creando un blucle con el que recorremos la colleción por sus llaves creando una lista con todas las probabilidades, la cual convertimos a un arreglo de `numpy` para aplicar la función `sum()`

# In[40]:


np.array([initTagStateProb[k] for k in initTagStateProb.keys()]).sum()


# En una forma más simple y eficiente, sumamos los valores de la colección accediendo directemante a la lista de valores de la colección con `list(initTagStateProb.values())`.

# In[41]:


np.array(list(initTagStateProb.values())).sum()


# En la forma muy eficiente, simplemente sumamos la lista de valores, sin utilizar `numpy`.

# In[42]:


sum(initTagStateProb.values())


# # Construcción del algoritmo de Viterbi
# 
# 
# 
# 
# 

# Dada una secuencia de palabras $\{p_1, p_2, \dots, p_n \}$, y un conjunto de categorias gramaticales dadas por la convención `upos`, se considera la matriz de probabilidades de Viterbi así:
# 
# $$
# \begin{array}{c c}
# \begin{array}{c c c c}
# \text{ADJ} \\
# \text{ADV}\\
# \text{PRON} \\
# \vdots \\
# {}
# \end{array} 
# &
# \left[
# \begin{array}{c c c c}
# \nu_1(\text{ADJ}) & \nu_2(\text{ADJ}) & \dots  & \nu_n(\text{ADJ})\\
# \nu_1(\text{ADV}) & \nu_2(\text{ADV}) & \dots  & \nu_n(\text{ADV})\\ 
# \nu_1(\text{PRON}) & \nu_2(\text{PRON}) & \dots  & \nu_n(\text{PRON})\\
# \vdots & \vdots & \dots & \vdots \\ \hdashline
# p_1 & p_2 & \dots & p_n 
# \end{array}
# \right] 
# \end{array}
# $$
# 
# Donde las probabilidades de la primera columna (para una categoria $i$) están dadas por: 
# 
# $$
# \nu_1(i) = \underbrace{\rho_i^{(0)}}_{\text{probabilidad inicial}} \times \underbrace{P(p_1 \vert i)}_{\text{emisión}}
# $$
# 
# luego, para la segunda columna (dada una categoria $j$) serán: 
# 
# $$
# \nu_2(j) = \max_i \{ \nu_1(i) \times \underbrace{P(j \vert i)}_{\text{transición}} \times \underbrace{P(p_2 \vert j)}_{\text{emisión}} \}
# $$
# 
# así, en general las probabilidades para la columna $t$ estarán dadas por: 
# 
# $$
# \nu_{t}(j) = \max_i \{ \overbrace{\nu_{t-1}(i)}^{\text{estado anterior}} \times \underbrace{P(j \vert i)}_{\text{transición}} \times \underbrace{P(p_t \vert j)}_{\text{emisión}} \}
# $$
# 
# ### Debemos importar la librería NLTK, ya que debemos tokenizar

# In[43]:


import nltk
nltk.download('punkt') # cargamos el paquete 'punkt' de NLTK
from nltk import word_tokenize # importamos el tokenizador de palabras


# Construimos la función `ViterbiMatrix` a la cual le pasamos la secuencia de palabras (Este `string` lo tenemos que tokenizar), la matriz de transición `A`, las probabilidades de emisión `B`, el diccionario de categorias con números para asignar a las columnas `tagStateDict` y la probabilidad de que encuentre una categoría gramatical al principio de una frase en el corpus `initTagStateProb`.

# In[44]:


# (secuencia, A, B, tagStateDict, initTagStateProb)
# Eacribimos la función con valores predeterminados, pero podrian enviarse otras matrices
def ViterbiTags(secuencia, 
                transitionProbdict=transitionProbdict, 
                emissionProbdict=emissionProbdict, 
                tagStateDict=tagStateDict, 
                initTagStateProb=initTagStateProb):
    
    # Tokenizamos la secuencia     
    seq = word_tokenize(secuencia)
    # Inicializamos la matrix de Viterbi, la cual inicia en cero
    viterbiProb = np.zeros((17, len(seq)))  # upos tiene 17 categorias
    
    # inicialización primera columna
    for tag in tagStateDict.keys():
        tag_row = tagStateDict[tag]
        word_tag = seq[0].lower() + '|' + tag
        if word_tag in emissionProbdict.keys():
            viterbiProb[tag_row, 0] = initTagStateProb[tag] * emissionProbdict[word_tag]
            
    for col in range(1, len(seq)):
        for tag_actual in tagStateDict.keys():
            tag_row = tagStateDict[tag_actual]
            word_tag = seq[col].lower() + '|' + tag_actual
            if word_tag in emissionProbdict.keys():
                possible_probs = []
                for tag_prev in tagStateDict.keys():
                    tag_prev_row = tagStateDict[tag_prev]
                    tag_prev_tag = tag_actual + '|' + tag_prev
                    if tag_prev_tag in transitionProbdict.keys():
                        if viterbiProb[tag_prev_row, col-1] > 0:
                            possible_probs.append(
                                viterbiProb[tag_prev_row, col-1] * transitionProbdict[tag_prev_tag] * emissionProbdict[word_tag])
                viterbiProb[tag_row, col] = max(possible_probs)
    
    # contruccion de secuencia de tags
    etiquetas = []
    for i, p in enumerate(seq):
        for tag in tagStateDict.keys():
            # Buscamos en que fila esta la máxima probabilidad de todas las posibles filas
            if tagStateDict[tag] == np.argmax(viterbiProb[:, i]):
                #print(tagStateDict[tag], np.argmax(viterbiProb[:, i]))
                etiquetas.append((p, tag))
                
    return etiquetas

ViterbiTags('el mundo es pequeño')


# In[45]:


ViterbiTags('estos instrumentos han de rasgar')


# # Entrenamiento directo de HMM con NLTK
# 
# * clase en python (NLTK) de HMM: https://www.nltk.org/_modules/nltk/tag/hmm.html
# 
# `@title` ejemplo con el Corpus Treebank en ingles

# In[ ]:





# # Carga del modelo HMM previamente entrenado
# 
# `@title` estructura de la data de entrenamiento

# In[ ]:





# `@title` HMM pre-construido en NLTK

# In[ ]:





# In[ ]:





# `@title` training accuracy

# In[ ]:





# ## Ejercicio de práctica
# 
# **Objetivo:** Entrena un HMM usando la clase `hmm.HiddenMarkovModelTrainer()` sobre el dataset `UD_Spanish_AnCora`.

# 1. **Pre-procesamiento:** En el ejemplo anterior usamos el dataset en ingles `treebank`, el cual viene con una estructura diferente a la de `AnCora`, en esta parte escribe código para transformar la estructura de `AnCora` de manera que quede igual al `treebank` que usamos así:
# 
# $$\left[ \left[ (\text{'El'}, \text{'DET'}), (\dots), \dots\right], \left[\dots \right] \right]$$

# In[46]:


# desarrolla tu código aquí 


# 2. **Entrenamiento:** Una vez que el dataset esta con la estructura correcta, utiliza la clase `hmm.HiddenMarkovModelTrainer()` para entrenar con el $80 \%$ del dataset como conjunto de `entrenamiento` y $20 \%$ para el conjunto de `test`.
# 
# **Ayuda:** Para la separacion entre conjuntos de entrenamiento y test, puedes usar la funcion de Scikit Learn: 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 
# En este punto el curso de Machine Learning con Scikit Learn es un buen complemento para entender mejor las funcionalidades de Scikit Learn: https://platzi.com/cursos/scikitlearn-ml/ 

# In[47]:


# desarrolla tu código aquí


# 3. **Validación del modelo:** Un vez entrenado el `tagger`, calcula el rendimiento del modelo (usando `tagger.evaluate()`) para los conjuntos de `entrenamiento` y `test`.
# 
# 

# In[48]:


#desarrolla tu código aquí

