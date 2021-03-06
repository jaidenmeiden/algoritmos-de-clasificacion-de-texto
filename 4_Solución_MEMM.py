# -*- coding: utf-8 -*-
"""4_Solución_MEMM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19laX7a78t9vQ_RkAKjis4H3s_9tS0tNn

## Construcción de un modelo markoviano de máxima entropía
"""

get_ipython().system('pip install conllu')
get_ipython().system('pip install stanza')
get_ipython().system('git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora.git')

"""### Entrenamiento del modelo - cálculo de conteos

Para este modelo consideramos el cálculo de las probabilidades: 

$$P(t_i | w_i, t_{i-1}) =\frac{C(w_i, t_i, t_{i-1})}{C(w_i, t_{i-1})} $$

* `uniqueFeatureDict` $C(tag|word,prevtag) = C(w_i, t_i, t_{i-1})$
* `contextDict` $C(word,prevtag) = C(w_i, t_{i-1})$

En este caso cuando consideremos el primer elemento de una frase $w_0$, no existirá un elemento anterior $w_{-1}$ y por lo tanto, tampoco una etiqueta previa $t_{-1}$, podemos modelar este problema asumiendo que existe una etiqueta "None", para estos casos: 

$$P(t_0|w_0,t_{-1}) = P(t_0|w_0,\text{"None"})$$
"""

from conllu import parse_incr

wordcorpus = 'form'
tagtype = 'upos'
data_file = open("UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")

uniqueFeatureDict = {}
contextDict = {}

# Calculando conteos (pre-probabilidades)
for tokenlist in parse_incr(data_file):
  prevtag = "None"
  for token in tokenlist:
    tag = token[tagtype]
    word = token[wordcorpus].lower()

    #C(tag,word,prevtag)
    c1 = tag + '(,)' + word + '(,)' + prevtag
    if c1 in uniqueFeatureDict.keys():
      uniqueFeatureDict[c1] += 1
    else:
      uniqueFeatureDict[c1] = 1

    #C(word|prevtag)  
    c2 = word + '(,)' + prevtag
    if c2 in contextDict.keys():
      contextDict[c2] += 1
    else:
      contextDict[c2] = 1
    prevtag=tag

"""### Entrenamiento del modelo - cálculo de probabilidades

$$P(t_i|w_i, t_{i-1}) = \frac{C(t_i, w_i, t_{i-1})}{C(w_i, t_{i-1})}$$
"""

posteriorProbDict = {}

for key in uniqueFeatureDict.keys():
  prob = key.split('(,)')
  posteriorProbDict[prob[0] + '(|)' + prob[1] + '(,)' + prob[2]] = uniqueFeatureDict[key]/contextDict[prob[1] + '(,)' + prob[2]]

# Aquí verificamos que todas las probabilidades 
# por cada contexto 'word,prevtag' suman 1.0
 
for base_context in contextDict.keys():
  sumprob = 0
  items = 0
  
  for key in posteriorProbDict.keys():
      if key.split('(|)')[1] == base_context:
        sumprob += posteriorProbDict[key]
        items += 1
  if sumprob != 1:
    print("La combinación '" + base_context + "'", "de " + str(items), "solo suma: " + str(sumprob))

"""### Distribución inicial de estados latentes"""

# identificamos las categorias gramaticales 'upos' unicas en el corpus
stateSet = set([k.split('(,)')[1] for k in contextDict.keys()])
stateSet.remove("None")
stateSet

# enumeramos las categorias con numeros para asignar a 
# las columnas de la matriz de Viterbi
tagStateDict = {}
for i, state in enumerate(sorted(stateSet)):
    tagStateDict[state] = i
tagStateDict

data_file = open("UD_Spanish-AnCora/es_ancora-ud-train.conllu", "r", encoding="utf-8")

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

sum(initTagStateProb.values())

"""### Construcción del algoritmo de Viterbi

Dada una secuencia de palabras $\{p_1, p_2, \dots, p_n \}$, y un conjunto de categorias gramaticales dadas por la convención `upos`, se considera la matriz de probabilidades de Viterbi así:

$$
\begin{array}{c c}
\begin{array}{c c c c}
\text{ADJ} \\
\text{ADV}\\
\text{PRON} \\
\vdots \\
{}
\end{array} 
&
\left[
\begin{array}{c c c c}
\nu_1(\text{ADJ}) & \nu_2(\text{ADJ}) & \dots  & \nu_n(\text{ADJ})\\
\nu_1(\text{ADV}) & \nu_2(\text{ADV}) & \dots  & \nu_n(\text{ADV})\\ 
\nu_1(\text{PRON}) & \nu_2(\text{PRON}) & \dots  & \nu_n(\text{PRON})\\
\vdots & \vdots & \dots & \vdots \\ \hdashline
p_1 & p_2 & \dots & p_n 
\end{array}
\right] 
\end{array}
$$

Donde las probabilidades de Viterbi en la primera columna (para una categoria $i$) están dadas por: 

$$
\nu_1(i) = \underbrace{\rho_i^{(0)}}_{\text{probabilidad inicial}} \times P(i \vert p_1, \text{"None"})
$$

y para las siguientes columnas: 

$$
\nu_{t}(j) = \max_i \{ \overbrace{\nu_{t-1}(i)}^{\text{estado anterior}} \times P(j \vert p_t, i) \}
$$

"""

import numpy as np 
import stanza
stanza.download('es')
nlp = stanza.Pipeline('es', processors='tokenize')

def ViterbiMatrix(
    secuencia, 
    posteriorProbDict = posteriorProbDict, 
    initTagStateProb = initTagStateProb):
  
  doc = nlp(secuencia)
  if len(doc.sentences) > 1:
    raise ValueError('secuencia must be a string!')
  seq = [word.text for word in doc.sentences[0].words]
  viterbiProb = np.zeros((17, len(seq)))
  
  # inicialización primera columna
  for tag in tagStateDict.keys():
    tag_row = tagStateDict[tag]
    key = tag + '(|)' + seq[0].lower() + '(,)' + "None"
    try:
      viterbiProb[tag_row, 0] = initTagStateProb[tag] * posteriorProbDict[key]
    except: 
      pass
  
  # computo de las siguientes columnas
  for col in range(1, len(seq)):
    for tag in tagStateDict.keys():
      tag_row = tagStateDict[tag]
      possible_probs = []
      for prevtag in tagStateDict.keys(): 
        prevtag_row = tagStateDict[prevtag]
        key = tag + '(|)' + seq[col].lower() + '(,)' + prevtag
        try:
          possible_probs.append(
              viterbiProb[prevtag_row, col-1] * posteriorProbDict[key])
        except:
          possible_probs.append(0)
      viterbiProb[tag_row, col] = max(possible_probs)

  return viterbiProb

ViterbiMatrix('el mundo es pequeño')

def ViterbiTags(
    secuencia, 
    posteriorProbDict = posteriorProbDict, 
    initTagStateProb = initTagStateProb):
  
  doc = nlp(secuencia)
  if len(doc.sentences) > 1:
    raise ValueError('secuencia must be a string!')
  seq = [word.text for word in doc.sentences[0].words]
  viterbiProb = np.zeros((17, len(seq)))
  
  # inicialización primera columna
  for tag in tagStateDict.keys():
    tag_row = tagStateDict[tag]
    key = tag + '(|)' + seq[0].lower() + '(,)' + "None"
    try:
      viterbiProb[tag_row, 0] = initTagStateProb[tag] * posteriorProbDict[key]
    except: 
      pass
  
  # computo de las siguientes columnas
  for col in range(1, len(seq)):
    for tag in tagStateDict.keys():
      tag_row = tagStateDict[tag]
      possible_probs = []
      for prevtag in tagStateDict.keys(): 
        prevtag_row = tagStateDict[prevtag]
        key = tag + '(|)' + seq[col].lower() + '(,)' + prevtag
        try:
          possible_probs.append(
              viterbiProb[prevtag_row, col-1] * posteriorProbDict[key])
        except:
          possible_probs.append(0)
      viterbiProb[tag_row, col] = max(possible_probs)

  # contruccion de secuencia de tags
  etiquetas = []
  for i, p in enumerate(seq):
    for tag in tagStateDict.keys():
      if tagStateDict[tag] == np.argmax(viterbiProb[:, i]):
        etiquetas.append((p, tag))


  return etiquetas

ViterbiTags('el mundo es pequeño')

ViterbiTags('estos instrumentos han de rasgar')

"""## ¿ Siguientes Pasos ? 

El modelo construido, aunque es la base de un MEMM, no explota todo el potencial del concepto  que estos modelos representan, en nuestro caso sencillo consideramos solo un **feature** para predecir la categoría gramatical: $<w_i, t_{i-1}>$. Es decir, las probabilidades de una cierta etiqueta $t_i$ dada una observación $<w_i, t_{i-1}>$ se calculan contando eventos donde se observe que $<w_i, t_{i-1}>$ sucede simultáneamente con $t_i$. 

La generalización de esto (donde puedo considerar multiples observaciones o **features**, y a partir de estos inferir la categoría gramatical) se hace construyendo las llamadas **feature-functions**, donde estas funciones toman valores de 0 o 1, cuando se cumplan las condiciones de la observación o feature en cuestion. En general podemos considerar una **feature-function** como : 

$$f_a(t, o) = f_a(\text{tag}, \text{observation}) = 
\begin{cases}
  1 , & \text{se cumple condición } a \\
  0, & \text{en caso contrario}
\end{cases}
$$

donde la condición $a$ es una relacion entre los valores que tome $\text{tag}$ y $\text{context}$, por ejemplo:

$$f_a(t, o) = f_a(\text{tag}, \text{observation}) = 
\begin{cases}
  1 , & (t_i, t_{i-1}) = \text{('VERB', 'ADJ')} \\
  0, & \text{en caso contrario}
\end{cases}
$$

Al considerar varias funciones, y por lo tanto varios features observables, consideramos una combinacion lineal de estos por medio de un coeficiente que multiplique a cada función: 

$$
\theta_1 f_1(t, o) + \theta_2 f_2(t, o) + \dots
$$

donde los coeficientes indicarán cuales features son más relevantes y por lo tanto pesan más para la decisión del resultado del modelo. De esta manera los coeficientes $\theta_j$ se vuelven parámetros del modelo que deben ser optimizados (esto puede realizarse con cualquier técnica de optimizacion como el Gradiente Descendente). Ahora, las probabilidades que pueden obtener usando un softmax sobre estas combinaciones lineales de features: 

$$
P = \prod_i \frac{\exp{\left(\sum_j \theta_j f_j(t_i, o)\right)}}{\sum_{t'}\exp{\left(\sum_j \theta_j f_j(t', o)\right)}}
$$

Así, lo que buscamos con el algoritmo de optimización es encontrar los parámetros $\theta_j$ que maximizan la probabilidad anterior. En NLTK encontramos la implementación completa de un clasificador de máxima entropia que no esta restringido a relaciones markovianas: https://www.nltk.org/_modules/nltk/classify/maxent.html

Un segmento resumido de la clase en python que implementa este clasificador en NLTK lo encuentras así: 

```
class MaxentClassifier(ClassifierI):

    def __init__(self, encoding, weights, logarithmic=True):
        self._encoding = encoding
        self._weights = weights
        self._logarithmic = logarithmic
        assert encoding.length() == len(weights)

    def labels(self):
        return self._encoding.labels()

    def set_weights(self, new_weights):
        self._weights = new_weights
        assert self._encoding.length() == len(new_weights)


    def weights(self):
        return self._weights

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        ### ...

        # Normalize the dictionary to give a probability distribution
        return DictionaryProbDist(prob_dict, log=self._logarithmic, normalize=True)

    @classmethod
    def train(
        cls,
        train_toks,
        algorithm=None,
        trace=3,
        encoding=None,
        labels=None,
        gaussian_prior_sigma=0,
        **cutoffs
    ):
     ### ......
```

Donde te das cuenta de la forma que tienen las clases en NLTK que implementan clasificadores generales. Aquí vemos que la clase `MaxentClassifier` es una subclase de una más general `ClassifierI` la cual representa el proceso de clasificación general de categoría única (es decir, que a cada data-point le corresponda solo una categoria), también que esta clase depende de definir un `encoding`
 y unos pesos `weights` : 

```
class MaxentClassifier(ClassifierI):

    def __init__(self, encoding, weights, logarithmic=True):
```

los pesos corresponden a los parámetros $\theta_i$. Y el encoding es el que corresponde a las funciones $f_a(t, o)$ que dan como resultado valores binarios $1$ o $0$.

La documentación de NLTK te puede dar mas detalles de esta implementación: https://www.nltk.org/api/nltk.classify.html

Finalmente, un ejemplo completo de uso y mejora de un modelo de máxima entropía, se puede encontrar en este fork para tenerlo como referencia y poder jugar y aprender con él: 

https://github.com/pachocamacho1990/nltk-maxent-pos-tagger

El cual fue desarrollado originalmente por Arne Neumann (https://github.com/arne-cl) basado en los fueatures propuestos por Ratnaparki en 1996 para la tarea de etiquetado por categorias gramaticales.

"""

