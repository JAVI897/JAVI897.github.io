---
title: Laplacian Score
date: 2020-12-5 11:58:47 +07:00
tags: [feature selection]
---

Los métodos de selección de variables se pueden clasificar en dos grandes grupos; 'wrapper methods' y 'filter methods'. Los métodos 'wrapper' evalúan las variables usando un algoritmo ya sea de regresión o de clasificación, en cambio, los métodos 'filter' examinan las *propiedades* de las variables para evaluar las variables. Laplacian Score (LS) es un método de selección de variables de tipo 'filter'. Fue propuesto en 2005 por He, X., Deng, C. y Partha, N. Para cada variable $x_{.j}$ se calcula su *Laplacian Score* de forma que este valor refleja la capacidad de preservar la geometría local que tiene la variable $x_{.j}$. LS asume que dos observaciones estarán relacionadas o pertenecerán a la misma clase si se encuentran *cerca*. Para modelar la geometría local se construirá un grafo de los vecinos más cercanos. LS le asignará más importancia a aquellas variables que mejor preserven esta geometría representada mediante el grafo de los vecinos más cercanos.

### Algoritmo

#### Construcción del grafo

La construcción del grafo de los vecinos más cercanos consta de dos partes; enlazar los vértices y asignar los pesos a cada enlace. En la versión no supervisada de Laplacian Score no se tiene en cuenta la clase de la observación, en cambio, en la versión supervisada <mark>se conectan solo aquellos nodos que comparten la misma etiqueta de clase</mark>.

##### Matriz de adyacencia

Sea $G = (V, E)$  donde $V$ es el conjunto de nodos que contiene las observaciones $x_i$ y $E$ es el conjunto de enlaces entre nodos vecinos. Existen dos variantes muy populares para construir este grafo;

###### **$\epsilon$-neighborhood graph**

En este caso se conecta a cada observación con sus vecinos y se consideran como vecinos a todas aquellas observaciones que se encuentren a un radio $\epsilon$.

$$N_{Eps}(x_i) = \{x_j \in D : d(x_j, x_i) \leq \epsilon  \}$$

*Ventajas:* Mantiene la geometría local cuando los datos están dispersos. *Desventajas:* Muchas veces produce un grafo con diversas componentes conexas. Una forma de solucionar este problema es escoger el mínimo valor de $\epsilon$ que genere un grafo conexo [1].

###### k-vecinos más cercanos

Al igual que antes, se conecta cada observación con sus vecinos y se considera como vecino a los $k$ más cercanos según la distancia euclídea.

$$N(x_i) = \{x_j \in D : d(x_j, x_i) \leq d(x_{L}, x_i )\}$$

Donde $x_L$ es la observación $k$ más cercana de $x_i$. *Ventajas:* Suele producir grafos conexos. *Desventajas:* Muchas veces se pierde la geometría local cuando los datos están dispersos.

##### Matriz de pesos

Las relaciones entre observaciones o nodos se refleja en los pesos de los enlaces de $G$, estos indican la similitud entre dos observaciones. En el artículo original de Laplacian Score se propone usar el kernel laplaciano como función de ponderación, sin embargo, existen otras funciones kernel muy populares;

###### Heat Kernel (o Laplacian Kernel)

Sea $s_{ij}$ el peso del enlace que une las observaciones vecinas $x_i$ y $x_j$. Definimos $ s_{ij}$ como;

$$s_{ij} = \left\{ \begin{array}{ccccc} e^{- \frac{ \lVert x_i - x_j \rVert ^2 }{\sigma}} & \text{si } x_i \text{ está conectado con }x_j  \\ 0 & \text{en otro caso} \\ \end{array} \right. $$

Donde $\sigma$ es un hiperparámetro que mide la varianza de la función.

###### Simple-minded

Sea $ s_{ij}$ el peso del enlace que une las observaciones vecinas $x_i$ y $x_j$. Definimos $s_{ij}$ como;

$$s_{ij} = \left\{ \begin{array}{ccccc} 1 & \text{si } x_i \text{ está conectado con }x_j  \\ 0 & \text{en otro caso} \\ \end{array} \right. $$

#### Cálculo del Laplacian Score

La idea en la que se basa Laplacian Score, de forma intuitiva, es que una variable $f_r$ será significativa si consigue que dos observaciones estén cerca si y solo si estas están conectadas en el grafo anterior. O sea, queremos una variable que minimice la siguiente expresión;

$$\phi(f_r) = \sum_{i,j} s_{ij} (f_{ri} - f_{rj})^2$$

Donde $f_r$ es un vector de dimensiones $n \times 1$ que contiene los valores de la variable $r$ para todas las $n$ observaciones. Fijémonos en las sutilezas de $\phi(f_r)$. Si las observaciones $x_i$ y $ x_j$ **no** son vecinas, $ s_{ij}$ será igual a $0$ y no ocurrirá nada. En cambio, si $x_i$ y $x_j$ son vecinas, $ s_{ij}$ tomará un valor elevado y por tanto, la expresión a minimizar será $(f_{ri} - f_{rj})^2$, o sea, la distancia euclídea al cuadrado entre el valor de la variable $r$ de ambas observaciones. Así, cuanto menor sea $\phi(f_r)$ mejor estaremos preservando la geometría local (la vecindad de las observaciones). En resumen, una buena variable será aquella para la que cuánto mayor sea $s_{ij}$ menor sea $(f_{ri} - f_{rj})$.

Pero eso no es todo, nos gustaría que además de preservar las localidades, la variable tuviese máxima varianza. Por tanto, el criterio sería minimizar la siguiente expresión;

$$L_r = \frac{\phi(f_r)}{Var(f_r)} = \frac{ \sum_{i,j} s_{ij} (f_{ri} - f_{rj})^2}{Var(f_r)}$$

Podemos reescribir el criterio $L_r$ de forma más conveniente, pero, para ello debemos introducir la matriz Laplaciana (ya explicada en [Spectral Clustering](https://javi897.github.io/Spectral_clustering/)).

------

##### Matriz Laplaciana

Existe todo un campo en las matemáticas dedicado a estudiar las propiedades de las matrices Laplacianas, así que me centraré en explicar únicamente algunas propiedades sobre esta que son importantes para el algoritmo Laplacian Score.

La matriz Laplaciana se define como;

$$L = D - S$$

Donde $S$ es la matriz de pesos del grafo de dimensiones $n \times n$. Y donde $D$ es una matriz diagonal de dimensiones $n \times n$. Los elementos de la diagonal de esta se definen como;

$$d_i = \sum_j^J s_{ij}$$

O sea, la suma de todos los pesos de los enlaces del vértice $i$. 

###### Propiedades

**1.**  $L$ es una matriz simétrica semidefinida positiva 

**2.** El menor valor propio de $L$ es $0$ y su vector propio asociado es un vector de unos

**3.** $L$ tiene valores propios reales y no negativos y estos están ordenados de forma ascendente; $0 = \lambda_1 \leq \lambda_2 \leq \cdots\leq\lambda_n$. De forma que, cuando hablemos de los primeros $K$ vectores propios de $L$ nos referiremos a los $K$ vectores propios asociados a los $K$ menores valores propios

**4.** Sea $G$ un grafo no-dirigido con pesos positivos. La multiplicidad del valor propio $0$ de $L$ es igual al número de componentes conexas del grafo. O sea, que si nuestro grafo tiene dos componentes conexas, tendremos dos valores propios iguales a $0$

**5.**  Para cualquier vector $f \in \mathbb{R}^n$ se cumple que;

$$f^tLf= \frac{1}{2} \sum_{i, j = 1}^n s_{ij} (f_i - f_j)^2$$

------

##### Reescribiendo $L_r$ 📃

Ahora, podemos reescribir la función $L_r$ en términos de la matriz laplaciana. Para ello, vamos a ir por partes (Jack the Ripper, 1888);

------

**Numerador de la fracción**

$$\phi(f_r) = \sum_{i,j} s_{ij} (f_{ri} - f_{rj})^2 = \sum_{ij} [ s_{ij}(f_{ri}^2 + f_{rj}^2 - 2f_{ri}f_{rj})] =$$

$$= \sum_{ij} s_{ij} f_{ri}^2 + \sum_{ji} s_{ij} f_{rj}^2 - 2 \sum_{ij} s_{ij}f_{ri}f_{rj} =$$

$$= \sum_{i} d_i f_{ri}^2 + \sum_{j} d_j f_{rj}^2 - 2 \sum_{ij}s_{ij}f_{ri}f_{rj} =$$

$$= 2 \sum_{i} d_i f_{ri}^2 - 2 \sum_{ij} s_{ij}f_{ri}f_{rj} =$$

$$= 2(\sum_{i} d_if_{ri}^2 - \sum_{ij} s_{ij}f_{ri}f_{rj}) =$$

$$= 2 (f_r^tDf_r - f_r^tSf_r) =$$

$$= 2(f_r^t (D- S)f_r) =$$

$$= 2(f_r^t L f_r)$$

De donde se demuestra la quinta propiedad de la matriz Laplaciana. Y por tanto, el numerador quedará reescrito como;

$$\phi(f_r) = 2(f_r^t L f_r)$$

------

**Denominador de la fracción**

La varianza de una variable aleatoria $a$ puede escribirse como:

$$Var(a) = \int_{M} (a - \mu)^2 dP(a), \; \mu = \int_M a dP(a)$$

donde $M$ es el *manifold* en el que se encuentran los datos (la superficie), $\mu$ es la esperanza de $a$ y $dP$ es una medida de probabilidad. Mediante la teoría espectral de grafos, $dP$ puede calcularse usando la matriz diagonal $D$. Por tanto, la varianza ponderada de la muestra se calculará como;

$$Var(f_r) = \sum_i (f_{ri} - \mu_r)^2D_{ii}$$

$$\mu_r = \sum_i (f_{ri} \frac{D_{ii}}{\sum_i D_{ii}}) = \frac{1}{\sum_i D_{ii}}(\sum_if_{ri}D_{ii}) = \frac{f_r^tD1}{1^tD1}$$

Donde $D_{ii}$ es el elemento $i$ de la diagonal de la matriz $D$, o sea, la suma de la i-ésima fila de la matriz $S$;

$$D_{ii} = \sum_{j}^{J} s_{ij}$$

Sea $\tilde{f_r}$ la variable $f_r$ centrada, o sea;

$$\tilde{f_r} = f_r - \mu_r1 = f_r - \frac{f_r^tD1}{1^tD1}1$$

Podemos reescribir $Var(f_r)$ como;

$$Var(f_r) = \sum_i \tilde{f_{ri}}^2D_{ii} = \tilde{f_r}^t D \tilde{f_r}$$

------

Finalmente, podemos reescribir $L_r$ como;

$$L_r = \frac{2(f_r^t L f_r)}{\tilde{f_r}^t D \tilde{f_r}}$$

Y como $2$ es un valor constante, lo podemos quitar de la expresión;

$$L_r = \frac{f_r^t L f_r}{\tilde{f_r}^t D \tilde{f_r}} $$

Dada la segunda propiedad de la matriz laplaciana; *El menor valor propio de $L$ es $0$ y su vector propio asociado es un vector de unos*, se cumple que;

$$L1 = 0 1 \rightarrow L1 = 0$$

$$L \; \alpha1 = 0 \; 1 \rightarrow L1 = 0 \; \forall \alpha \in \mathbb{R}^n$$

Por tanto; $f_r^tLf^r = \tilde{f_r}^tL\tilde{f_r}$. Esto es porque;

$$\tilde{f_r}^tL\tilde{f_r} = (f_r^t - \mu_r^t1^t)L(f_r - \mu_r1)=$$

$$\require{cancel} = (f_r^tL - \cancelto{0}{\mu_r^t1^tL})(Lf_r - \cancelto{0}{L\mu_r1}) = (f_r^tL)(Lf_r) = f_r^tLf_r$$

Y podemos reescribir $L_r$ como;

$$L_r = \frac{\tilde{f_r}^t L \tilde{f_r}}{\tilde{f_r}^t D \tilde{f_r}} $$

#### Referencias

- [1] Varini, Claudio & Degenhard, Andreas & Nattkemper, Tim. (2005). ISOLLE: Locally linear embedding with geodesic distance. 331-342. 10.1007/11564126_34.
- [2] He, X., Deng, C. y Partha, N. (2005). Laplacian Score for Feature Selection. Advances in Neural Information Processing Systems; 18:507-514.