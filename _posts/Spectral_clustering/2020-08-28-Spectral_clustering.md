---
title: Spectral Clustering
date: 2020-08-28 11:58:47 +07:00
tags: [clustering]
---

Spectral Clustering, que bien podría parecer el nombre de un libro de Stephen King, es un método de clustering que se ha vuelto muy popular en los últimos años. Y que muchas veces funciona mucho mejor que los clásicos algoritmos como K-medias.

### Intuición

El objetivo último del clustering es encontrar grupos en los datos tal que las observaciones dentro de un mismo grupo sean similares y que observaciones entre diferentes grupos sean disímiles. Usando las similitudes entre puntos podemos definir un grafo que contenga esta información;

$$G = (V, E)$$

Donde $V$ es el conjunto de nodos que contiene las observaciones $x_i$ y $E$ es el conjunto de enlaces entre nodos. Cada enlace tiene un peso que hace referencia a la similitud/distancia entre dos observaciones. Existen diferentes propuestas de grafos que son muy usadas en el algoritmo  Spectral Clustering;

------

###### $\epsilon$-neighborhood graph

En este caso se conecta a cada observación con sus vecinos y se consideran como vecinos a todas aquellas observaciones que se encuentren a un radio $\epsilon$.

$$N_{Eps}(x_i) = \{x_j \in D : d(x_j, x_i) \leq \epsilon  \}$$

###### k-vecinos más cercanos

Al igual que antes, se conecta cada observación con sus vecinos y se considera como vecino a los $k$ más cercanos.

$$N(x_i) = \{x_j \in D : d(x_j, x_i) \leq d(x_{L}, x_i )\}$$

Donde $x_L$ es la observación $k$ más cercana de $x_i$. 

###### Fully connected graph

En este se conectan todas las observaciones y se pondera cada enlace según la similitud entre observaciones. Una función para representar la similitud entre dos observaciones puede ser el kernel gaussiano ya comentado en [Kernel Functions](https://javi897.github.io/Kernels/);

$$s(x, y) = e^{- \frac{\lVert x- y \rVert^2}{2 \sigma^2}} = e^{-\gamma \lVert x- y \rVert^2}$$

------

Queremos que los nodos de un subgrafo sean similares entre ellos y que los nodos del otro subgrafo también lo sean. Si el grafo es conexo, debemos seleccionar un conjunto de enlaces para cortar el grafo. Por tanto, parecería razonable que los pesos de estos enlaces sumasen lo menos posible (considerando que los pesos representan la similitud entre dos observaciones). Entonces, podemos formular el objetivo del clustering en términos de grafos como; <mark>encontrar particiones del grafo tal que los enlaces entre estas particiones tengan el menor peso posible</mark>. En adelante, formularemos el problema de encontrar $k=2$ clusters por facilidad.  En el último aparatado se explica cómo se puede emplear el algoritmo para encontrar más de 2 clusters.

<center><img src="\assets\img\graph-cut.PNG" style="zoom:85%;" /></center>

**Fuente figura**: Adamic, Glance. The political blogosphere and the 2004 US election: divided they blog. 36-43 (2005).

### The perfect cut ✂️ 💫 

Bien, según lo anterior, para desconectar el grafo deberemos eliminar aquellos enlaces que unan ambos subgrafos y estos enlaces deberán tener un peso muy bajo así nos aseguramos que ambos subgrafos son disimiles. 

Esto nos permite formular el problema anterior como; **minimizar el sumatorio de pesos de los enlaces que unen ambos conjuntos ($\bar{A}, A$)**. Para dos conjuntos disjuntos de vértices; $A, \bar{A}  \subset V $ donde $\bar{A} = V - A$. 

$$\text{min} \; \frac{1}{2} \sum_{i\in A, j \in \bar{A}} w_{ij} $$

Se añade el factor $\frac{1}{2}$ para que no se cuente cada enlace por dos. Definimos $cut(A, \bar{A} )$ como el peso de todos los enlaces que unen ambos conjuntos de vértices;

$$cut(A, \bar{A}) = \frac{1}{2}\sum_{i\in A, j \in \bar{A}} w_{ij}$$

Por tanto, el problema anterior se transforma en minimizar $cut(A, \bar{A})$.

##### Problema con esta aproximación

Existe un problema con este criterio ya que puede llevarnos a soluciones en las que el número de vértices de $A$ sea muy diferente al de $\bar{A}$ ya que en la función a minimizar no exigimos que ambos conjuntos de vértices tangan un cardinal similar.

<center><img src="\assets\img\spc2.PNG" style="zoom:85%;" /></center>

**Fuente figura**: Jure Leskovec. Community Detection: Spectral Clustering. CS224W: Analysis of Networks

### RatioCut

Para evitarnos el problema descrito vamos a cambiar de función a minimizar y consideraremos RatioCut (Hagen and Kahng, 1992). 

$$ratiocut = \frac{1}{2} (\frac{cut(A, \bar{A})}{ \mid A \mid} + \frac{cut(A, \bar{A})}{ \mid \bar{A} \mid} )$$

Al minimizar esta función estamos exigiendo que $A$ y $\bar{A}$ sean de tamaños similares. En RatioCut el tamaño de los conjuntos de vértices se mide como el número de vértices de cada uno. Otras propuestas como Ncut (Shi and Malik, 2000) proponen medir el tamaño de un conjunto como la suma de pesos de cada conjunto.

### Reescribiendo 📃

Vamos a reescribir el problema de forma más conveniente (para la optimización).

Sea $f \in \mathbb{R}^n$. Definimos $f_i$ como;

$$f_i = \left\{ \begin{array}{ccccc}  \sqrt{\frac{\mid \bar{A} \mid}{\mid A \mid}} & \text{si } i \in A \\   - \sqrt{\frac{\mid A \mid}{\mid \bar{A} \mid}} & \text{si } i \in \bar{A} \\ \end{array} \right.$$

De cierta forma estamos asignando un valor positivo a los vértices que pertenezcan a $A$ y un valor negativo a los que pertenezcan a $\bar{A}$.

Ahora, minimizar la función $\frac{1}{2} \sum_{i, j = 1}^n w_{ij} (f_i - f_j)^2$ será equivalente a minimizar $ratiocut$.

$$\frac{1}{2} \sum_{i, j = 1}^n w_{ij} (f_i - f_j)^2 = $$

$$= \frac{1}{2} \sum_{i\in A, j \in \bar{A}} w_{ij} ( \sqrt{\frac{\mid \bar{A} \mid}{\mid A \mid}} + \sqrt{\frac{\mid A \mid}{\mid \bar{A} \mid}} )^2 + \frac{1}{2} \sum_{i\in \bar{A}, j \in A} w_{ij} ( -\sqrt{\frac{\mid A \mid}{\mid \bar{A} \mid}} - \sqrt{\frac{\mid \bar{A} \mid}{\mid A \mid}} )^2 =$$

$$= cut(A, \bar{A})(\frac{\mid \bar{A} \mid}{\mid A \mid} + \frac{\mid A \mid}{\mid \bar{A} \mid} + 2) = cut(A, \bar{A})(\frac{\mid A \mid + \mid \bar{A} \mid}{\mid A \mid} + \frac{\mid A \mid + \mid \bar{A} \mid}{\mid \bar{A} \mid} ) =$$

$$= \mid V \mid ratiocut(A, \bar{A})$$

Así que minimizando $\frac{1}{2} \sum_{i, j = 1}^n w_{ij} (f_i - f_j)^2$ estaremos minimizando $ratiocut$😏. Pero todavía podemos reescribirlo mejor, para ello primero debemos introducir la matriz Laplaciana.

##### Matriz Laplaciana

Existe todo un campo en las matemáticas dedicado a estudiar las propiedades de las matrices Laplacianas, así que me centraré en explicar únicamente algunas propiedades sobre esta que son importantes para el Spectral Clustering.

La matriz Laplaciana se define como;

$$L = D - W$$

Donde $W$ es la matriz de pesos del grafo de dimensiones $n \times n$. Y donde $D$ es una matriz diagonal de dimensiones $n \times n$. Los elementos de la diagonal de esta se definen como;

$$d_i = \sum_j^J w_{ij}$$

O sea, la suma de todos los pesos de los enlaces del vértice $i$. Es decir, la suma de las similitudes de los vecinos de $x_i$ con $x_i$.

###### Propiedades

**1.**  $L$ es una matriz simétrica semidefinida positiva 

**2.** El menor valor propio de $L$ es $0$ y su vector propio asociado es un vector de unos

**3.** $L$ tiene valores propios reales y no negativos y estos están ordenados de forma ascendente; $0 = \lambda_1 \leq \lambda_2 \leq \cdots\leq\lambda_n$. De forma que, cuando hablemos de los primeros $K$ vectores propios de $L$ nos referiremos a los $K$ vectores propios asociados a los $K$ menores valores propios

**4.** Sea $G$ un grafo no-dirigido con pesos positivos. La multiplicidad del valor propio $0$ de $L$ es igual al número de componentes conexas del grafo. O sea, que si nuestro grafo tiene dos componentes conexas, tendremos dos valores propios iguales a $0$

Y la propiedad que más nos interesa; para cualquier vector $f \in \mathbb{R}^n$ se cumple que;

$$f^tLf= \frac{1}{2} \sum_{i, j = 1}^n w_{ij} (f_i - f_j)^2 = \mid V \mid ratiocut(A, \bar{A})$$

###### Demostración

$$f^tLf = f^t (D - W)f = f^tDf - f^tWf = \sum_i d_if_i^2 - \sum_{ij} w_{ij}f_i f_j =$$

$$= \frac{1}{2} (2 \sum_i d_i f_i^2 - 2 \sum_{ij} w_{ij} f_i f_j) =$$

$$= \frac{1}{2} (\sum_i d_i f_i^2 + \sum_j d_j f_j^2 - 2 \sum_{ij} w_{ij} f_i f_j) =$$

$$= \frac{1}{2}(\sum_i f_i^2 \sum_j w_{ij} + \sum_j f_j^2 \sum_i w_{ij} - 2 \sum_{ij} w_{ij}f_if_j) = $$

$$= \frac{1}{2} (\sum_{ij}w_{ij}f_i^2 + \sum_{ij} w_{ij} f_j^2 - 2 \sum_{ij} w_{ij} f_i f_j) = $$

$$= \frac{1}{2} (\sum_{i j} w_{ij} (f_i - f_j)^2)$$

Por lo tanto, podemos formular el problema de minimizar RatioCut como;

$$\underset{f}{\text{min}} \; f^tLf$$

Donde $L$ es conocida y $f$ es el vector que queremos minimizar. Ya nos hemos encontrado con esta expresión a minimizar en PCA. Sin embargo, allí contábamos con una restricción. Sin una restricción en el problema la solución será cero, así que vamos a añadir la restricción de que el módulo de $f$ sea 1.

$$\underset{f}{\text{min}} \; f^tLf$$

$$s.t \;\; f^tf = 1$$

Realmente la restricción que teníamos era que si $x_i \in A$ el valor de $f_i$ era $\sqrt{\frac{\mid \bar{A} \mid}{\mid A \mid}}$ o $-\sqrt{\frac{\mid A \mid}{\mid \bar{A} \mid}}$ si $x_i \in \bar{A}$. Pero no se puede solucionar el problema de forma eficiente usando esta restricción. Por tanto, en vez de usar esa restricción usamos $f^tf = 1$. Los valores de $f$ no serán iguales a los descritos en $f_i$, sin embargo, serán positivos si la observación pertenece a $A$ y negativos si pertenece a $\bar{A}$

Como ya vimos en PCA, la solución a este problema de optimización son los vectores propios asociados a los menores valores propios de $L$. Pero como ya hemos visto en las propiedades de $L$, esta tiene siempre un valor propio igual a $0$, así que <mark>la solución de $f$ será el vector propio de $L$ asociado al menor valor propio distinto de $0$.</mark>

Una vez calculado el vector $f$. Asignaremos una observación a $A$ o $\bar{A}$ según si $f_i$ es positivo o negativo;

$$\left\{ \begin{array}{ccccc}  x_i \in A & \text{si } f_i \geq 0 \\   x_i \in \bar{A} & \text{si } f_i < 0 \\ \end{array} \right.$$

### ¿Y si k > 2?

La intuición del algoritmo explicado se basaba en cortar el grafo en dos partes. Pero, ¿qué ocurre si queremos más de dos clusters? La extensión de Spectral Clustering para un $k > 2$ no tiene una teoría tan desarrollada como para $k = 2$ y es más heurística. 

Con $k = 2$ escogíamos solo un vector propio de $L$; $f$ y asignábamos una observación a un cluster u a otro basándonos en el signo de $f_i$. Si queremos crear $k$ clusters, lo que haremos ahora será escoger $k - 1$ vectores propios de $L$ asociados a los menores valores propios distintos de $0$. Y sobre estos $k - 1$ vectores propios se ejecuta un algoritmo de clustering cualquiera (k-means, k-medoides, gaussian mixture...).



#### Referencias

- [1] <a href="https://www.youtube.com/watch?v=V680Ev0MNvs&t=3702s" target="_blank" rel="noopener">Ali Ghodsi, Lec 5: LLE, Spectral Clustering</a>
- [2] <a href="https://www.youtube.com/watch?v=DW3lSYltfzo" target="_blank" rel="noopener">Ali Ghodsi, Lec 6: Spectral Clustering, Laplacian Eigenmap, MVU</a>
- [3] <a href="http://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf" target="_blank" rel="noopener">Ulrike von Luxburg. A Tutorial on Spectral Clustering (2007)</a>

