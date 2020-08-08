---
title: PCA y Kernel PCA
date: 2020-08-08 11:58:47 +07:00
tags: [dimensionality reduction]
---

El Análisis de Componentes Principales (PCA en adelante) es un método clásico de reducción de la dimensión que surge en el siglo XIX de la mano de Eugenio Beltrami (en 1873) y Camille  Jordan (en 1874) en el campo de la descomposición espectral. Sin embargo, su uso en la estadística se la debemos Pearson y Hotelling. Karl Pearson la describió en 1901 y más tarde Hotelling la desarrolló en 1933.

Kernel PCA es una extensión de PCA mucho más reciente que utiliza el "kernel trick" que también es muy usado en otros algoritmos como, por ejemplo, las máquinas de soporte vectorial.

### PCA

PCA transforma una matriz de datos de un espacio *d*-dimensional en otra que se encuentra en una dimensión *p* donde *p* es menor que *d*. El objetivo en PCA será preservar tanta varianza ($\sigma^2$) como sea posible en las nuevas dimensiones. 

Las nuevas dimensiones se llaman componentes principales. La proyección de las observaciones originales sobre las componentes se denomina scores.  Por ejemplo, en la siguiente imagen vemos unas observaciones en un espacio 2-dimensional. El vector *p* sería el nuevo subespacio lineal sobre el que se proyectarían las observaciones. Podemos definir los elementos de la variable latente $t_1$ (score en la primera dimensión):

$$t_{1i} = proy_{p_1}(x_{i}) = p_1^t \; x_i = \sum_{j=1}^d p_{1j} \; x_{ij} =  p_{11} \; x_{i1} + p_{12} \; x_{i2} + ... + p_{1d} \; x_{id}$$

<center><img src="\assets\img\pca3.PNG" style="zoom:99%;" /></center>

Sea $X$ una matriz de datos de dimensiones $k \times n$ (donde $k$ es el número de variables y $n$ el número de individuos) y $p$ un vector de dimensiones $k \times 1$. Podemos expresar los scores en la primera dimensión de forma matricial :

$$t_1 = p_1^t X$$

Ahora la pregunta es... ¿cómo encontramos a $p$? El vector $p$ debe cumplir una serie de criterios;

------

### 1. Subespacio que mejor se ajusta a la nube

Según Hotelling, las componentes habría que seleccionarlas como ejes ortogonales de forma que minimizasen <mark>las sumas de las distancias residuales (suma de cuadrados residual)</mark> de las observaciones originales a las proyectadas.

<center><img src="\assets\img\pca1.PNG" style="zoom:99%;" /></center>

$$\lVert x_i \rVert^2 = \lVert x_i^{*} \rVert^2 + \lVert r \rVert^2 $$

$$\underset{w}{\text{min}} \; \sum_{i=1}^{n} w_i\lVert r_i \rVert^2 = \underset{}{\text{min}} \; SCR = \underset{}{\text{min}} \; \sum_{i=1}^{n} e_i^te_i $$

Minimizar la suma de cuadrados residual implica maximizar la norma euclídea de $x_i^{*}$, podríamos haber formulado el problema como:

$$\underset{w}{\text{max}} \; \sum_{i=1}^{n} w_i\lVert x_i^* \rVert^2$$

Por ello, seleccionar las componentes de forma que se minimice la suma de cuadrados residual es equivalente a  maximizar la varianza de las observaciones al proyectarse sobre los ejes.

### 2. Subespacio que minimiza la deformación de la nube proyectada

Las distancias entre observaciones en la nube proyectada deben ser lo más parecidas posibles a las distancias existentes en los datos originales.

<center><img src="\assets\img\pca2.PNG" style="zoom:99%;" /></center>

Por tanto, deberemos minimizar la diferencia entre las distancias originales y las proyectadas.

$$ \underset{w}{\text{min}} \; \sum_{i, j} w_iw_j(d^2(i, j) - d^{*2}(i, j)) = $$

$$\underset{w}{\text{max}} \; \sum_{i, j} w_iw_j \; d^{*2}(i, j) = \underset{w}{\text{max}} \; \sum_{i} w_i \; d^{*2}(i, m)$$

Llegamos a la conclusión de que debemos maximizar las distancias de las observaciones proyectadas con respecto al centro de coordenadas. Por tanto, también en este caso el criterio conduce a seleccionar como ejes definitorios del subespacio (vectores *p*) los asociados a las combinaciones
lineales de las variables de varianza máxima.

------

### Maximizar la varianza de la nube proyectada

Ambos criterios nos llevan a la conclusión de que debemos encontrar una dirección $\vec{p}$ que maximice la varianza de la variable latente. Asumiendo que $X$ está centrada, la varianza de esta variable latente se define como;

$$\sigma_{t_1}^2 = var(t_1) = var(p_1^t X) = t_1 t_1^t = (p_1^tX)(p_1^tX)^t = p_1^tXX^tp_1 = p_1^tSp_1$$

Donde $S = XX^t$ es la matriz de covarianzas de dimensiones $k \times k$. Entonces, podemos formular el problema como:

$$\underset{p}{\text{max}} \; p_1^tSp_1$$

La varianza de *t* puede ser infinitamente grande aumentando la magnitud de *p*. Por tanto, para que la maximización tenga solución, el módulo de *p* debe ser 1:

$$\underset{p}{\text{max}} \; p_1^tSp_1$$

$$s.t. \;\;\; p_1^tp_1 = 1$$

Para resolver este problema de optimización se usan los multiplicadores de Lagrange:

$$L(p_1) = p_1^tSp_1 - \lambda (p_1^tp_1-1)$$

$$\frac{dL}{dp_1} = 2Sp_1 - 2\lambda p_1 = 0 \rightarrow 2 S p_1 = 2\lambda p_1 \rightarrow Sp_1 = \lambda p_1$$

Es decir, que $p_1$ es un vector propio de $S$ con valor propio $\lambda$. Más aun:

$$\underset{p}{\text{max}} \; p_1^tSp_1 = \underset{p}{\text{max}} \; p_1^t\lambda p_1 = \underset{p}{\text{max}} \;  p_1^t p_1 \lambda$$

Y como los vectores propios son ortogonales, el producto de la traspuesta de $p_1$ por $p_1$ será 1. Y tendremos:

$$\underset{p}{\text{max}} \;\lambda$$

Por tanto, <mark>$p_1$ será el vector propio asociado al mayor valor propio de $S$</mark>. Así, para encontrar las componentes principales solo tenemos que calcular $S$ y descomponer esta en vectores y valores propios.

### Algoritmo PCA

Existen diversos algoritmos para el Análisis de Componentes Principales; NIPALS, descomposición en vectores y valores propios, MDS... En este post presentaré el algoritmo basado en la descomposición en valores singulares de $X$ (asumo que el lector está familiarizado con la descomposición en valores singulares de una matriz).

Mediante la SVD se puede descomponer cualquier matriz en tres matrices. Sea $X$ una matriz centrada de dimensiones $d \times n$:

$$X = U \Sigma V^T$$

Donde:

-  $U$ es una matriz ortogonal cuyas columnas son los vectores propios asociados a los valores propios de $XX^t$ ortonormalizados
-  $V$ es una matriz ortogonal cuyas columnas son los vectores propios asociados a los valores propios de $X^tX$ ortonormalizados
- $\Sigma$ es una matriz diagonal que contiene los valores singulares de $X$, que no son más que la raíz cuadrada de los valores propios de  $XX^t$ o  $X^tX$

Una de las ventajas de la descomposición en valores singulares es que las matrices $U$ y $V$ tienen sus columnas ordenadas según los valores propios (de mayor a menor).

<center><img src="\assets\img\pca4.PNG" style="zoom:99%;" /></center>

##### Proyección

Como la matriz está centrada, y si recordamos, $XX^t$ es la matriz de covarianzas. Por tanto, la matriz $U$ contiene los vectores propios de la matriz de covarianzas, o sea, las componentes principales ($\vec{p}$). Entonces, las variables latentes, o sea, los scores, se calcularán como: 

$$T = U^tX$$

Donde la matriz $T$ es la matriz de scores de dimensiones $p \times n$. Si queremos reducir la dimensionalidad de $\mathbb{R}^d$ a $\mathbb{R}^p$, la matriz $U$ contendrá únicamente las primeras *p* columnas.

También podemos definir la matriz de scores como:

$$T = U^tX = U^t (U\Sigma V^t) = \Sigma V^t$$

Donde, al igual que antes, las matrices $V$ y $\Sigma$ contendrán los vectores propios (de $X^tX$ en este caso) y valores singulares asociados a los $p$ mayores valores singulares.  

##### Reconstrucción

Una de las ventajas de PCA frente a otros métodos es que podemos reconstruir la matriz de datos $X$:

$$X' = UT = U\Sigma V^t = XV\Sigma^{-1} \Sigma V^t = X V V^t$$

##### Proyección nuevas observaciones

Otra de las ventajas de PCA es que se pueden proyectar nuevas observaciones (centradas y escaladas según las originales) sobre el espacio creado de dimensión $p$:

$$t = U^tx = \Sigma^{-1}V^tX^tx = \Sigma^{-1}V^tX^tx$$

##### Reconstrucción nuevas observaciones

Al igual que se puede proyectar una observación nueva en el subespacio creado, también se puede reconstruir dicha observación:

$$x' = Ut = UU^tx = XV\Sigma^{-1} (XV\Sigma^{-1})^t x =$$

$$= XV\Sigma^{-2} V^tX^tx $$

Nota: no caigamos en la tentación de decir que $UU^t = I$ ya que no estamos considerando la matriz $U$ completa sino sus $p$ primera columnas.

### Kernel PCA

Hasta el momento todo era lineal, hemos asumido que nuestros datos se encontraban sobre un subespacio lineal. ¿Pero qué ocurre si nuestros datos se encuentran en un *sub-manifold* donde existe curvatura?  Por ejemplo, tomemos el conjunto de datos de la siguiente imagen. 

<center><img src="\assets\img\pca5.png" style="zoom:99%;" /></center>

La recta de la imagen se corresponde con la aplicación de PCA sobre los datos. Como se observa, PCA proyecta los datos sobre la dirección de máxima varianza y no se consigue separar entre la clase roja y la amarilla que aparecen superpuestas.

Kernel PCA respetará la estructura del *sub-manifold* al reducir la dimensión. Pero antes de nada debemos entender qué es un kernel.

------



##### Transformaciones no lineales

Las observaciones de la imagen anterior no son separables de forma lineal en $\mathbb{R}^2$. La idea detrás de las transformaciones con kernel es que si nuestros datos no son separables de forma lineal, podemos manipular nuestros datos y llevárnoslos a un espacio de una dimensión mayor donde las clases sí sean separables de forma lineal. Por ejemplo, tomemos la siguiente transformación (o homomorfismo):

$$x \rightarrow \phi(x)$$

Donde $x$ es una observación de la imagen anterior y $\phi(x)$ es un *mapping* de esa observación a un espacio $\mathbb{R}^3$. $\phi(x)$ se define como:

$$\phi(x) = \phi(\pmatrix{x_1 \cr x_2}) = \pmatrix{x_1^2 \cr x_2^2 \cr \sqrt{2}x_1x_2}$$

Si aplicamos esta transformación, las clases son separables de forma lineal en $\mathbb{R}^3$.

<center><img src="\assets\img\pca6.png" style="zoom:99%;" /></center>

Existen muchas funciones $\phi(x)$ diferentes que harán que los datos sean separables de forma lineal en un espacio de una dimensión mayor. Kernel PCA se basa en la idea de que muchos conjuntos de datos que no son separables de forma lineal en el espacio original, sí pueden serlo al proyectarlos en un espacio de una dimensión mayor. 

##### Funciones kernel

Un kernel es una función que toma dos vectores de datos en el espacio original $\mathbb{R}^p$ y devuelve el producto escalar de ambos vectores en un espacio $\mathbb{R}^m$ donde *m > p​*.

De forma más formal, sea $x, y \in X$ y sea $\phi: X \rightarrow \mathbb{R}^m$ entonces:

$$k(x, y) = \langle \phi(x), \phi(y) \rangle = \phi(x)^t\phi(y)$$

es un kernel.

##### Ejemplo de kernel

Supongamos que tenemos dos observaciones $x, y \in \mathbb{R}^2$ y les aplicamos a ambas observaciones la transformación $\phi(x)$ vista antes.

$$x = \pmatrix{x_1 \cr x_2} \xrightarrow{\phi} \phi(x) = \pmatrix{x_1^2 \cr x_2^2 \cr \sqrt{2}x_1x_2} $$

$$y = \pmatrix{y_1 \cr y_2} \xrightarrow{\phi} \phi(y) = \pmatrix{y_1^2 \cr y_2^2 \cr \sqrt{2}y_1y_2} $$

Si calculamos $\phi(x)^t\phi(y)$ estaremos calculando $k(x, y)$:

$$k(x, y) = \phi(x)^t\phi(y) = \pmatrix{x_1^2 \;\;\; x_2^2  \;\;\; \sqrt{2}x_1x_2} \pmatrix{y_1^2 \cr y_2^2 \cr \sqrt{2}y_1y_2} = $$

$$= x_1^2y_1^2 + x_2^2y_2^2+2x_1x_2y_1y_2$$

Bien, existe otra función que toma como entrada $x$ e $y$ y nos devuelve exactamente lo mismo que  $\phi(x)^t\phi(y)$:

$$k(x, y) = (\langle x, y \rangle)^2 = (\pmatrix{x_1 \;\;\; x_2}\pmatrix{y_1 \cr y_2})^2 = (x_1y_1 + x_2y_2)^2 = $$

$$= x_1^2y_1^2 + x_2^2y_2^2+2x_1x_2y_1y_2$$

Por tanto, tenemos la función $(\langle x, y \rangle)^2$ que calcula el producto escalar entre $x$ e $y$  y que a su vez puede calcular el producto escalar $\phi(x)^t\phi(y)$. Esta es la principal ventaja de los kernels. <mark>Podemos calcular  $\phi(x)^t\phi(y)$ sin necesidad de saber nada sobre $\phi(x)$ o $\phi(y)$, tan solo usando productos escalares entre las observaciones originales.</mark>

Existen muchas funciones kernel; *lineal, polynomial, radial basis function*... y cada una de ellas tiene su propio $\phi$.

------

Bien, ahora que ya sabemos qué es un kernel, podemos volver al tema que nos ocupa; Kernel PCA. 

En kernel PCA se proyectan los datos ($\mathbb{R}^d$) a un espacio $\mathbb{R}^m$ donde *d < m* y luego a un espacio menor que *m*. En síntesis, Kernel PCA toma nuestros datos y los proyecta en un espacio de una dimensión mayor donde se realiza PCA.

$$X \rightarrow L \rightarrow T$$

$$\mathbb{R}^d \rightarrow \mathbb{R}^m \rightarrow \mathbb{R}^l$$

El objetivo es reducir la dimensionalidad pasando por $\phi(X)$ sin necesidad de conocer $\phi(X)$. Para reducir la dimensión de $\mathbb{R}^m$ a $\mathbb{R}^l$ se aplica PCA. Definimos la matriz $\phi(X)$ como:

$$\phi(X) = \pmatrix{\phi(x_1), \;\;\; \phi(x_2),  ..., \phi(x_n)}$$

De dimensiones $m \times n$. Asumiendo que $\phi(X)$ es una matriz centrada, aplicando el algoritmo de la SVD;

$$\phi(X) = U \Sigma V^t$$

Donde $U$ es una matriz ortogonal cuyas columnas son los vectores propios asociados a los valores propios de $\phi(X)\phi(X)^t$. $U$ era la matriz que se usaba para proyectar los datos sobre un subespacio lineal en el algoritmo de la SVD. Sin embargo, calcular $U$ va a salirnos (computacionalmente hablando) muy costoso si *m* es muy grande ya que las dimensiones de $\phi(X)\phi(X)^t$ serán; $m \times m$.

Para evitarnos tener que calcular $U$, primero asumimos que tenemos una función kernel $k(x, y)$ . Usando esa función kernel podemos calcular la matriz $K=\phi(X)^t\phi(X)$ de forma eficiente sin tener que calcular $\phi(X)$ (como hemos visto con el ejemplo). Los elementos de esta matriz $K$ serán:

$$k_{i, j} = \langle \phi(x_i), \phi(x_j) \rangle = k(x_i, x_j)$$

Las dimensiones de $K$ serán; $n \times n$.  Pero recordemos que hemos asumido que la matriz $\phi(X)$ es una matriz centrada, de forma que;

$$\phi(x_i)' = \phi(x_i) - \frac{1}{m} \sum_{k=1}^{m} \phi(x_k)$$

Si queremos calcular la matriz $K$ sin necesidad de pasar por $\phi(X)$ deberemos encontrar una expresión que centre $K$ sin necesidad de conocer $\phi(X)$. Esto implica que los elementos de la matriz $K$ serán;

$$k_{i, j} = k(x_i, x_j) = (\phi(x_i)')^t\phi(x_j)' =$$

$$= ( \phi(x_i) - \frac{1}{m} \sum_{k=1}^{m} \phi(x_k) )^t ( \phi(x_j) - \frac{1}{m} \sum_{k=1}^{m} \phi(x_k)) =$$

$$= k(x_i, x_j) - \frac{1}{m}\sum_{k=1}^m k(x_i, x_k) - \frac{1}{m}\sum_{k=1}^m k(x_j, x_k) - \frac{1}{m^2} \sum_{l, k=1}^m k(x_l, x_k)$$

Si recordamos, en el algoritmo anterior podíamos calcular los scores usando únicamente $\Sigma$ y $V$.

$$T = \Sigma V^t$$

Donde;

- $V$ es una matriz ortogonal cuyas columnas son los vectores propios asociados a los valores propios de $\phi(X)^t\phi(X)$, o sea, los vectores propios de $K$
- $\Sigma$ es una matriz diagonal que contiene los valores singulares de $\phi(X)$, que no son más que la raíz cuadrada de los valores propios de $\phi(X)^t\phi(X)$  o $\phi(X)\phi(X)^t$ 

Descomponiendo la matriz $K$ en vectores y valores propios tenemos que:

$$K = V\Lambda V^t$$

Donde $V$ es la matriz de  vectores propios y $\Lambda$ la matriz diagonal de valores propios. Por tanto, $\sqrt{\Lambda} = \Sigma$. Así, descomponiendo en vectores y valores propios la matriz $K$ podemos calcular los scores $T$.

##### Pasos

**`Paso 1`** Calcular $K$ centrada usando la fórmula anterior

**`Paso 2`** Descomponer $K$ en vectores y valores propios. Sea $V$ la matriz de vectores propios de $K$ y sea $\Sigma$ una matriz diagonal de las raíces de los valores propios de $K$

**`Paso 3`** Escoger las $l$ primeras columnas de $\Sigma$ y $V$ asociados a los $l$ valores propios más grandes. Y calcular $T$ (scores) como $T = \Sigma V^t$ 

##### Reconstrucción

En PCA podíamos reconstruir los datos, en cambio, no podremos hacerlo en Kernel PCA;

$$X' = UT = U \Sigma V^t = \phi(X)V \Sigma^{-1}\Sigma V^t = \phi(X) V V^t $$

No podremos ya que se desconoce $\phi(X)$.

##### Proyección nuevas observaciones

Al igual que en PCA, en kernel PCA también podemos proyectar nuevas observaciones sobre el subespacio creado; 

$$t = U^t\phi(x) = \Sigma^{-1}V^t\phi(X)^t\phi(x)$$

Podremos ya que $\phi(X)^t\phi(x) = k(X, x)$

##### Reconstrucción nuevas observaciones

En PCA podíamos reconstruir observaciones nuevas, sin embargo, no podremos en kernel PCA.

$$x' = Ut = UU^t\phi(x) = \phi(X)V \Sigma^{-1} (\phi(X)V\Sigma^{-1})^t\phi(x)=$$

$$= \phi(X)V\Sigma^{-2}V^t\phi(X)^t\phi(x)$$

Se desconoce $\phi(X)$, por lo tanto, no se puede calcular la anterior expresión.

### Ejemplo Kernel rbf

<p float="middle">
	<img src="\assets\img\pca7.png" style="zoom:45%;" />
    <img src="\assets\img\pca8.png" style="zoom:45%;" />
</p>

Kernel PCA aplicado sobre el dataset visto en imágenes anteriores. Creado usando **sklearn.**

#### Referencias

- <a href="http://www.math.uwaterloo.ca/~aghodsib/courses/f10stat946/notes/lec6.pdf" rel="noopener">PDF: Lecture 6: Dual PCA, Kernel PCA (Ali Ghodsi) </a>
- <a href="https://www.youtube.com/watch?v=jeOEXCFK30M&t=2003s" rel="noopener">Video: Lecture 6: Dual PCA, Kernel PCA (Ali Ghodsi) </a>
- <a href="https://www.youtube.com/watch?v=HbDHohXPLnU" rel="noopener">David Thompson (Part 6): Nonlinear Dimensionality Reduction: KPCA</a>
- Data Science y redes complejas (Eloy Vicente Cestero y Alfonso Mateos Caballero)
- <a href="https://drewwilimitis.github.io/The-Kernel-Trick/" target="_blank" rel="noopener">The Kernel Trick in Support Vector Classification (Drew Wilimitis)</a>