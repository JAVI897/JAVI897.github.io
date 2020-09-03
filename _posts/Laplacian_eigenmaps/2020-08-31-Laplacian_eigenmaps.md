---
title: Laplacian eigenmaps
date: 2020-08-31 11:58:47 +07:00
modified: 2020-09-03 11:58:49 +07:00
tags: [dimensionality reduction]
---

Laplacian eigenmaps es otro algoritmo de reducción de la dimensión no-lineal. Fue propuesto en 2003 por Mikhail Belkin y Partha Niyogi. Es un algoritmo que posee muchas similitudes con el algoritmo Spectral Clustering ya que ambos se basan en la matriz Laplaciana como luego veremos. También mantiene un gran parecido con LEE, de hecho, se puede interpretar el algoritmo LEE (Roweis y Saul, 2000) como una versión de Laplacian Eigenmaps. 

### Intuición

El problema de la reducción de la dimensión se puede formular de forma muy simple como; dado un conjunto de $k$ observaciones $x_i, \dots, x_k$ en un espacio $\mathbb{R}^l$, encontrar un conjunto de puntos $y_1, \dots y_k$ en un espacio $\mathbb{R}^m$ donde $m \ll l $.

Se han propuesto multitud de algoritmos para solucionar este problema. MDS, por ejemplo, encuentra estos *embeddings* de forma que se mantengan las distancias euclídeas entre observaciones. LLE, por otra parte, reconstruye los datos en la nueva dimensión de forma que se preserva la geometría local de las observaciones.

Laplacian eigenmaps reconstruirá las observaciones $x_i$ de forma que <mark>se minimice la distancia entre observaciones vecinas y se maximice la distancia entre observaciones que no son vecinas.</mark> Esta vecindad de una observación se obtiene creando un grafo de los vecinos más cercanos ya que Laplacian Eigenmaps al igual que otros algoritmos como t-SNE, ISOMAP, UMAP o LLE modela el manifold usando un grafo de los vecinos más cercanos. Sin embargo, la diferencia entre estos algoritmos y Laplacian Eigenmaps es que este último hace uso de la matriz Laplaciana; $L$. Usando la descomposición espectral de $L$ y alguna de sus propiedades podemos encontrar un *embedding* óptimo que preserva la estructura del manifold [1].

### Algoritmo

#### Construcción del grafo

La construcción del grafo de los vecinos más cercanos consta de dos partes; enlazar los vértices y asignar los pesos a cada enlace. 

##### Matriz de adyacencia

Sea $G = (V, E)$  donde $V$ es el conjunto de nodos que contiene las observaciones $x_i$ y $E$ es el conjunto de enlaces entre nodos vecinos. Existen dos variantes muy populares en Laplacian Eigenmaps para construir este grafo;

###### **$\epsilon$-neighborhood graph**

En este caso se conecta a cada observación con sus vecinos y se consideran como vecinos a todas aquellas observaciones que se encuentren a un radio $\epsilon$.

$$N_{Eps}(x_i) = \{x_j \in D : d(x_j, x_i) \leq \epsilon  \}$$

*Ventajas:* Mantiene la geometría local cuando los datos están dispersos. *Desventajas:* Muchas veces produce un grafo con diversas componentes conexas. Una forma de solucionar este problema es escoger el mínimo valor de $\epsilon$ que genere un grafo conexo [2].

###### k-vecinos más cercanos

Al igual que antes, se conecta cada observación con sus vecinos y se considera como vecino a los $k$ más cercanos según la distancia euclídea.

$$N(x_i) = \{x_j \in D : d(x_j, x_i) \leq d(x_{L}, x_i )\}$$

Donde $x_L$ es la observación $k$ más cercana de $x_i$. *Ventajas:* Suele producir grafos conexos. *Desventajas:* Muchas veces se pierde la geometría local cuando los datos están dispersos.

##### Matriz de pesos

Las relaciones entre observaciones o nodos se refleja en los pesos de los enlaces de $G$, estos indican la similitud entre dos observaciones. Al igual que en el apartado anterior, también existen dos variantes muy populares para escoger los pesos de los enlaces.

###### Heat Kernel (o Laplacian Kernel)

Sea $w_{ij}$ el peso del enlace que une las observaciones vecinas $x_i$ y $x_j$. Definimos $w_{ij}$ como;

$$w_{ij} = \left\{ \begin{array}{ccccc} e^{- \frac{ \lVert x_i - x_j \rVert ^2 }{\sigma}} & \text{si } x_i \text{ está conectado con }x_j  \\ 0 & \text{en otro caso} \\ \end{array} \right. $$

Donde $\sigma$ es un hiperparámetro que mide la varianza de la función.

###### Simple-minded

Sea $w_{ij}$ el peso del enlace que une las observaciones vecinas $x_i$ y $x_j$. Definimos $w_{ij}$ como;

$$w_{ij} = \left\{ \begin{array}{ccccc} 1 & \text{si } x_i \text{ está conectado con }x_j  \\ 0 & \text{en otro caso} \\ \end{array} \right. $$

#### Reducción de la dimensión

En adelante vamos a suponer que el grafo $G$ es conexo y vamos a considerar el problema de encontrar los *embeddings* de las observaciones $x_i$ en un espacio 1-dimensional (o sea, una línea). Para encontrar el *embedding* $y = (y_1, \dots , y_k)^t$ un criterio razonable sería minimizar la siguiente función;

$$\phi(y) = \sum_{i,j} w_{ij} (y_i - y_j)^2$$

Fijémonos en las sutilezas de $\phi(y)$. Si las observaciones $y_i$ y $y_j$ **no** son vecinas, $w_{ij}$ será igual a $0$ y no ocurrirá nada. En cambio, si $y_i$ y $y_j$ son vecinas, $w_{ij}$ tomará un valor elevado y por tanto, la expresión a minimizar será $(y_i - y_j)^2$, o sea, la distancia euclídea al cuadrado entre ambas observaciones. Así, minimizando $\phi(y)$ nos aseguramos de que las observaciones vecinas aparezcan cerca en el espacio m-dimensional. 

Podemos reescribir la función $\phi(y)$ de forma más conveniente para la optimización, pero, para ello debemos introducir la matriz Laplaciana (ya explicada en [Spectral Clustering](https://javi897.github.io/Spectral_clustering/)).

------



##### Matriz Laplaciana

Existe todo un campo en las matemáticas dedicado a estudiar las propiedades de las matrices Laplacianas, así que me centraré en explicar únicamente algunas propiedades sobre esta que son importantes para el algoritmo Laplacian Eigenmaps.

La matriz Laplaciana se define como;

$$L = D - W$$

Donde $W$ es la matriz de pesos del grafo de dimensiones $n \times n$. Y donde $D$ es una matriz diagonal de dimensiones $n \times n$. Los elementos de la diagonal de esta se definen como;

$$d_i = \sum_j^J w_{ij}$$

O sea, la suma de todos los pesos de los enlaces del vértice $i$. 

###### Propiedades

**1.**  $L$ es una matriz simétrica semidefinida positiva 

**2.** El menor valor propio de $L$ es $0$ y su vector propio asociado es un vector de unos

**3.** $L$ tiene valores propios reales y no negativos y estos están ordenados de forma ascendente; $0 = \lambda_1 \leq \lambda_2 \leq \cdots\leq\lambda_n$. De forma que, cuando hablemos de los primeros $K$ vectores propios de $L$ nos referiremos a los $K$ vectores propios asociados a los $K$ menores valores propios

**4.** Sea $G$ un grafo no-dirigido con pesos positivos. La multiplicidad del valor propio $0$ de $L$ es igual al número de componentes conexas del grafo. O sea, que si nuestro grafo tiene dos componentes conexas, tendremos dos valores propios iguales a $0$

**5.**  Para cualquier vector $f \in \mathbb{R}^n$ se cumple que;

$$f^tLf= \frac{1}{2} \sum_{i, j = 1}^n w_{ij} (f_i - f_j)^2$$

------

##### Reescribiendo $\phi(y)$ 📃

Ahora, podemos reescribir la función $\phi(y)$ en términos de la matriz laplaciana;

$$\phi(y) = \sum_{ij} w_{ij}(y_i - y_j)^2 = \sum_{ij} [ w_{ij}(y_i^2 + y_j^2 - 2y_iy_j)] =$$

$$= \sum_{ij} w_{ij} y_i^2 + \sum_{ji} w_{ij} y_j^2 - 2 \sum_{ij} w_{ij}y_iy_j =$$

$$= \sum_{i} d_i y_i^2 + \sum_{j} d_jy_j^2 - 2 \sum_{ij}w_{ij}y_iy_j =$$

$$= 2 \sum_{i} d_iy_i^2 - 2 \sum_{ij} w_{ij}y_iy_j =$$

$$= 2(\sum_{i} d_iy_i^2 - \sum_{ij} w_{ij}y_iy_j) =$$

$$= 2 (y^tDy - y^tWy) =$$

$$= 2(y^t (D- W)y) =$$

$$= 2(y^t L y)$$

De donde se demuestra la quinta propiedad de la matriz Laplaciana. Por tanto, podemos formular el problema de minimizar $\phi(y)$ como;

$$\underset{y}{\text{min}} \; y^t L y \rightarrow \underset{y}{\text{min}} \; \langle Ly, y \rangle $$

Para que la optimización tenga solución se añade la restricción de que $y^tDy = 1$ que hace que los *embeddings* sean invariantes a escalados arbitrarios. Normalmente se añade la restricción $y^ty = 1$.  Pero en este caso se usa la matriz $D$ ya que contiene información sobre lo importante que es cada vértice. Para resolver el anterior problema podemos usar los multiplicadores de Lagrange. 

$$\pi(y) = y^tLy - \lambda(y^tDy - 1)$$

$$\frac{\partial \pi}{\partial y} = Ly - \lambda Dy = 0$$

La solución a este problema de optimización se obtiene de los vectores propios asociados a los menores valores propios del problema de valor propio generalizado de la forma;

$$L y = \lambda Dy$$

En este caso habría una solución trivial que sería el vector propio asociado al valor propio 0. Para que no aparezca esta solución podemos añadir la siguiente restricción $y^tD1 = 0$ o podemos simplemente ignorar el primer vector propio.

##### Generalizando

Hasta ahora hemos considerado el problema de encontrar los *embeddings* en un espacio 1-dimensional. Vamos ahora a considerar el problema más general de encontrar los *embeddings* en un espacio que no sea 1-dimensional. Entonces, el vector $y$ se convierte en la matriz $Y$. La función $\phi(Y)$ a minimizar será;

$$\phi(Y) = \sum_{ij} w_{ij} \lVert y_i - y_j\rVert^2_2$$

Donde $\lVert y_i - y_j\rVert^2_2$ es la distancia euclídea entre dos observaciones de dimensiones $1 \times m$. Notése que antes también estábamos calculando la distancia euclídea pero al ser $y$ de dimensiones $1 \times 1$ podíamos no usar la notación de la norma euclídea. Podemos reescribir lo anterior de forma muy similar a como hemos hecho con $\phi(y)$;

$$\phi(Y) = \sum_{ij} w_{ij} \lVert y_i - y_j\rVert^2_2 = \sum_{ij} [ w_{ij}( \lVert y_i \rVert^2_2 + \lVert y_j \rVert^2_2 - 2y_i^ty_j)] = $$

$$= \sum_{ij} w_{ij} \lVert y_i \rVert^2_2 + \sum_{ji} w_{ij} \lVert y_j \rVert^2_2 - 2 \sum_{ij} w_{ij}y_i^ty_j = $$

$$= \sum_{i} d_i \lVert y_i^2 \rVert + \sum_{j} d_j \lVert y_j \rVert^2_2 - 2 \sum_{ij}w_{ij}y_i^ty_j =$$

$$= 2 \sum_{i} d_i \lVert y_i \rVert^2_2 - 2 \sum_{ij} w_{ij}y_i^ty_j $$

Para reescribir la última ecuación en forma matricial iremos por partes (Jack the Ripper, 1888);

1️⃣ $\sum_{ij} w_{ij}y_i^ty_j = Tr(YW Y^t)$

Para demostrar esta igualdad consideremos una matriz $Y$ de dimensiones $2 \times 2$. O sea, de dos observaciones en un espacio 2-dimensional (las observaciones son las columnas de $Y$). Entonces desarrollando $Tr(YWY^t)$ tenemos que;

$$Tr(YWY^t) = Tr(\left[ \begin{array}{ccc} y_{11} & y_{21}\\ y_{12} & y_{22} \end{array} \right] \left[ \begin{array}{ccc} w_{11} & w_{12}\\ w_{21} & w_{22} \end{array} \right] \left[ \begin{array}{ccc} y_{11} & y_{12}\\ y_{21} & y_{22} \end{array} \right]) = $$

$$ \scriptsize = Tr(\left[ \begin{array}{ccc} y_{11}(y_{11}w_{11} + y_{21}w_{21}) + y_{21}(y_{11}w_{12} + y_{21}w_{22}) & y_{12}(y_{11}w_{11} + y_{21}w_{21}) + y_{22}(y_{11}w_{12} + y_{21}w_{22}) \\ y_{11}(y_{12}w_{11} + y_{22}w_{21}) + y_{21}(y_{12}w_{12} + y_{22}w_{22}) & y_{12}(y_{12}w_{11} + y_{22}w_{21}) + y_{22}(y_{12}w_{12} + y_{22}w_{22}) \end{array} \right]) = $$

$$\scriptsize = y_{11}(y_{11}w_{11} + y_{21}w_{21}) + y_{21}(y_{11}w_{12} + y_{21}w_{22}) + y_{12}(y_{12}w_{11} + y_{22}w_{21}) + y_{22}(y_{12}w_{12} + y_{22}w_{22}) = $$

$$\scriptsize = y_{11}^2w_{11} + y_{11}y_{21}w_{21} + y_{21}y_{11}w_{12} + y_{21}^2w_{22} + y_{12}^2w_{11} + y_{12}y_{22}w_{21} + y_{22}y_{12}w_{12} + y_{22}^2w_{22}$$

Desarrollando ahora $\sum_{ij} w_{ij}y_i^ty_j$ tenemos que;

$$\sum_{ij} w_{ij}y_i^ty_j = \sum_{i=1}^2\sum_{j=1}^2 w_{ij}y_i^ty_j =$$

$$ = w_{11}y_1^ty_1 + w_{12}y_1^ty_2 + w_{21}y_2^ty_1 + w_{22}y_2^ty_2 =$$

$$\scriptsize = y_{11}^2w_{11} + y_{11}y_{21}w_{21} + y_{21}y_{11}w_{12} + y_{21}^2w_{22} + y_{12}^2w_{11} + y_{12}y_{22}w_{21} + y_{22}y_{12}w_{12} + y_{22}^2w_{22}$$

Por tanto, ambas expresiones son equivalentes ya que desarrollando ambas llegamos a la misma expresión.

2️⃣ $\sum_{i} d_i \lVert y_i \rVert_2^2 = Tr(Y D Y^t)$

Al igual que antes, para demostrar esta igualdad consideremos una matriz $Y$ de dimensiones $2 \times 2$. Entonces, desarrollando $Tr(YDY^t)$ tenemos que;

$$Tr(YDY^t) = Tr(\left[ \begin{array}{ccc} y_{11} & y_{21}\\ y_{12} & y_{22} \end{array} \right] \left[ \begin{array}{ccc} d_1 & 0\\ 0 & d_2 \end{array} \right] \left[ \begin{array}{ccc} y_{11} & y_{12}\\ y_{21} & y_{22} \end{array} \right]) = $$

$$ = Tr(\left[ \begin{array}{ccc} y_{11}d_1y_{11} + y_{21}d_2y_{21} & y_{11}d_1y_{12} + y_{21}d_2y_{22} \\  y_{12}d_1y_{11} + y_{22} d_2 y_{21} & y_{12}d_1y_{12} + y_{22}d_2y_{22}  \end{array} \right]) = $$

$$= y_{11}d_1y_{11} + y_{21}d_2y_{21} + y_{12}d_1y_{12} + y_{22}d_2y_{22} = $$

$$= d_1 y_{11}^2 + d_1 y_{12}^2 + d_2 y_{21}^2 + d_2 y_{22}^2 = d_1 (y_{11}^2 + y_{12}^2) + d_2 (y_{21}^2 + y_{22}^2) =$$

$$= d_1 \langle y_1, y_1 \rangle + d_2 \langle y_2, y_2 \rangle = d_1 \lVert y_1 \rVert_2^2 + d_2 \lVert y_2 \rVert^2_2 = $$

$$= \sum_{i=1}^2 d_i \lVert y_i \rVert_2^2$$

------

Entonces, tomando 1️⃣ y 2️⃣ podemos reescribir $\phi(Y)$ en forma matricial;

$$\phi(Y) = 2 Tr(YDY^t) - 2 Tr(Y W Y^t) = 2 ( Tr(YDY^t) - Tr(Y W Y^t) )$$

$$= 2 Tr(YDY^t - Y W Y^t) = 2 Tr(Y (D- W) Y^t ) = $$

$$= 2 Tr(Y L Y^t)$$

------

**Restricciones**

Se debe añadir la misma restricción a los *embeddings* que la vista para el caso 1-dimensional.

**Scaling factor** Multiplicar $y_i$ por un vector $c$ cualquiera no debería afectar a $\phi(Y)$. Por ello, se añade la restricción $YDY^t = I$

------

##### Optimización

 Queremos minimizar $\phi(Y)$ teniendo en cuenta además la anterior restricción. Podemos formular el problema como;

$$\underset{Y}{ \text{min} } \; Tr(YLY^t)$$

$$s.t.   YDY^t = I $$

La solución a este problema de optimización se obtiene de los vectores propios asociados a los menores valores propios del problema de valor propio generalizado de la forma;

$$Ly = \lambda Dy \rightarrow (D- W)y = \lambda Dy$$

Donde los valores propios se obtienen del polinomio característico; $det(L - \lambda D)$ y los valores propios se obtienen resolviendo el sistema; $(L - \lambda D)y = 0$ para cada valor propio.

Se llega a esta conclusión resolviendo los multiplicadores de Lagrange tal y como hemos hecho con $m=1$. 

Aunque este problema se puede formular como un problema de valores propios estándar modificando la anterior expresión;

$$Ly = \lambda D y$$

$$D^{-1}Ly = \lambda y$$

$$(D^{-1}D - D^{-1}W)y = \lambda y$$

$$(I - D^{-1}W)y = \lambda y$$

$$\tilde{L}y = \lambda y$$

Donde $\tilde{L} = I - D^{-1}W$, esta matriz se llama *Random Walk Normalized Laplacian*. 

Por tanto, la matriz $Y$ será una matriz donde las columnas serán los vectores propios asociados a los menores valores propios distintos de cero de la matriz $\tilde{L}$. Y escogeremos tantos vectores propios como dimensiones queramos en el nuevo espacio dimensional.

### Swiss Roll

Uno de los clásicos datasets que se usa para demostrar la eficacia de un algoritmo de reducción de la dimensión no-lineal es el *Swiss Roll* que ya comentamos en [ISOMAP](https://javi897.github.io/ISOMAP/). En el artículo original de Laplacian Eigenmaps, como no podía ser de otro modo, también se muestra un ejemplo de este dataset.

<center><img src="\assets\img\LE1.PNG" style="zoom:90%;" /></center>

**Fuente figura**: Belkin, Niyogi. *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation 15, 1373–1396 (2003)

En la siguiente figura se observa el *embedding* del *Swiss Roll* para diferentes valores de $t$ y de $N$, que son el valor del parámetro del heat kernel ( que nosotros hemos llamado $\sigma$ )  y el número de vecinos en el grafo de los k-vecinos más cercanos respectivamente. 

<center><img src="\assets\img\LE2.PNG" style="zoom:90%;" /></center>

**Fuente figura**: Belkin, Niyogi. *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation 15, 1373–1396 (2003)

El caso de $t = ∞$ ocurre cuando los pesos toman valores binarios. Se observa que para valores de $N$ pequeños se obtienen representaciones muy similares independientemente del hiperparámetro $t$. En cambio, según se aumenta $N$, valores pequeños de $t$ ofrecen una mejor representación.

#### Referencias

- [1] <a href="https://nbviewer.jupyter.org/github/drewwilimitis/Manifold-Learning/blob/master/Laplacian-Eigenmaps.ipynb" target="_blank" rel="noopener"> Drew Wilimitis. Laplacian Eigenmaps and Spectral Embedding</a>
- [2] Varini, Claudio & Degenhard, Andreas & Nattkemper, Tim. (2005). ISOLLE: Locally linear embedding with geodesic distance. 331-342. 10.1007/11564126_34.
- [3] Belkin, Niyogi. *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation 15, 1373–1396 (2003)
- [4] [Dr. Juan Orduz. On Laplacian Eigenmaps for Dimensionality Reduction ](https://juanitorduz.github.io/documents/orduz_pydata2018.pdf) 
- [5] Ángela Fernández Pascual (2010). *Advanced methods for dimensionality reduction and clustering: Laplacian Eigenmaps and Spectral Clustering*

