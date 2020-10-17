---
title: Expectation Maximization
date: 2020-9-11 11:58:47 +07:00
tags: [optimization]
---



El algoritmo EM (Expectation Maximization) es un algoritmo muy popular que se usa para encontrar estimadores de máxima verosimilitud de parámetros de distribuciones o modelos que dependen de variables ocultas. Fue propuesto por Arthur Pentland Dempster, Nan Laird y Donald Rubin de la Royal Statistical Society en una publicación de 1977. En el mismo artículo se señala que el algoritmo ya se había usado en otras circunstancias por otros autores pero la publicación de 1977 generaliza el método.

### Mixtura de distribuciones

Las variables aleatorias se pueden considerar como muestras de una distribución. Pero a veces, estas variables aleatorias se extraen de una mezcla de distribuciones, por ejemplo, una distribución de Poisson y una normal.  Supongamos una muestra de individuos $X_1, ..., X_N$ donde cada individuo puede ser o bien una tentación o bien tener pareja. Asociada a cada observación tendremos una variable que nos dirá a qué grupo pertenece $z_i \in \\{T, P\\}$. Normalmente desconocemos si un individuo es tentación o tiene pareja (Estefanía, 2019), es por eso que la variable $z$ será una variable oculta. Por el teorema de la probabilidad total sabemos que;

$$p(x) = \sum_{k=1}^K p(x \vert z=k) ~ p(z=k) = \sum_{k=1}^K p(x\vert z=k) ~ \alpha_k$$

$$\text{s.t.} ~~~~ \sum_{k=1}^K \alpha_k = 1$$

Donde $\alpha_k$ nos indica la probabilidad de que $x_i$ pertenezca a la k-ésima distribución y $p(x\vert z=k)$ representa la distribución de $x_i$ asumiendo que pertenece a la distribución $k$.

Una mixtura de distribuciones para variables continuas se define de forma genérica como;

$$f(x; \Theta_1, \dots, \Theta_K) = \sum_{k=1}^K \alpha_k\, g_k(x; \Theta_k)$$

$$\text{s.t.} ~~~~ \sum_{k=1}^K \alpha_k = 1$$

### Máxima verosimilitud

Asumamos que tenemos una muestra de observaciones de tamaño $n$; $S = \\{x_1, \dots, x_n\\}$ y que conocemos la distribución de la que han sido obtenidas las observaciones pero desconocemos los parámetros de esta distribución. Una forma de estimar los parámetros de la distribución usando $S$ es mediante MLE (Maximum Likelihood Estimation). 

La idea en la que se basa MLE es la de maximizar la función de verosimilitud que se define como la función de densidad conjunta de todas las observaciones asumiendo que son independientes, esto es;

$$L(\Theta) = f(x_1, x_2, \dots, x_n \vert \Theta_1, \dots, \Theta_k) = \prod_{i=1}^n f(x_i \vert \Theta_1, \dots, \Theta_k)$$

$L(\Theta)$ es una función de $\Theta$ que nos dirá lo bien que ajusta el o los parámetros $\Theta$  (puede ser un vector) en la distribución.

<center><img src="\assets\img\MLE.png" style="zoom:85%;" /></center>

Así, el método MLE encuentra el vector $\Theta$ que maximiza la función $L(\Theta)$;

$$\widehat{\Theta} = \arg \max_{\Theta} L(\Theta)$$

Normalmente, por cuestiones de cálculo, se prefiere utilizar el logaritmo de la verosimilitud;

$$\ell(\Theta) = \log L(\Theta) = \log \prod_{i=1}^n f(x_i, \Theta) = \sum_{i=1}^n \log f(x_i, \Theta)$$

### Algoritmo

El algoritmo EM es un algoritmo iterativo que se usa para estimar los parámetros de distribuciones de probabilidad subyacentes en los datos. Consta de dos partes; 

Se inicializan los parámetros a valores aleatorios:

1️⃣ **Paso E (expectation step)**; se calcula la esperanza de las variables ocultas. La probabilidad de que $x$ haya sido obtenida por la distribución k-ésima (probabilidad a posteriori) usando los parámetros de la iteración $t$

2️⃣ **Paso M (maximization step)**; estimar los nuevos parámetros con los parámetros de la iteración anterior y usando las probabilidades a posteriori obtenidas en el paso E

Los pasos E y M se alternan hasta la convergencia, esto es, evaluamos el logaritmo de la verosimilitud tras el paso M y si este ha variado menos que un valor $\epsilon$ con respecto a la anterior iteración, se detiene el algoritmo. 

Para entender cómo funciona el algoritmo EM vamos a partir de un ejemplo. Estimaremos los parámetros de **dos distribuciones** que no tienen por qué ser de la misma familia. 

#### Paso E

La mixtura para dos distribuciones será;

$$f(x; \Theta_1, \Theta_2) = \alpha g_1(x;\theta_1) + (1-\alpha) g_2(x;\theta_2) $$

Donde $\alpha$ es la probabilidad de que $x$ pertenezca a la distribución $g_1$. A este parámetro se le denomina en la literatura *mixing probability* y se lo suele denotar con la letra $\pi $ [1]. El logaritmo de la verosimilitud de esta mixtura será;

$$\ell(\Theta_1, \Theta_2) = log \prod_{n=1}^{N} f(x_i; \Theta_1, \Theta_2) = $$

$$= log \prod_{n=1}^{N}  \alpha g_1(x;\theta_1) + (1-\alpha) g_2(x;\theta_2) = $$

$$= \sum_{i=1}^N log [\alpha g_1(x;\theta_1) + (1-\alpha) g_2(x;\theta_2)]$$

Podemos reescribir el logaritmo de la verosimilitud más fácilmente definiendo la función $\Delta_i$. $\Delta_i$ tomará el valor 1 si $x_i$ pertenece a la distribución 1 y 0 si $x_i$ pertenece a la distribución 2.

$$\Delta_i = \left\{ \begin{array}{ccccc} 1 & \text{si } x_i \in g_1 \\0 & \text{si } x_i \in g_2 \\ \end{array} \right.$$

La probabilidad de que $\Delta_i$ sea igual a 1 o 0 se define como;

$$\left\{ \begin{array}{ccccc} p(\Delta = 1) = \alpha \\ p(\Delta = 0) = 1 - \alpha & \\ \end{array} \right.$$

Por tanto, podemos reescribir $\ell(\Theta_1, \Theta_2)$ como;

$$\ell(\Theta_1, \Theta_2) = \left\{ \begin{array}{ccccc} \sum_{i=1}^N log[ \alpha g_1(x;\theta_1) ]& \text{si } \Delta_i = 1\\ \sum_{i=1}^N log[ (1 - \alpha) g_2(x;\theta_2) ] & \text{si } \Delta_i = 0 \\ \end{array} \right.$$

La expresión anterior se puede expresar también como;

$$\ell(\Theta_1, \Theta_2) = \sum_{i=1}^N [\Delta_i log(\alpha g_1(x;\theta_1)) + (1 - \Delta_i) log[(1-\alpha) g_2(x;\theta_2)]] $$

En la anterior expresión el parámetro que desconocemos es $\Delta_i$. A priori no sabemos si una observación pertenece a la distribución 1 o 2. Lo que haremos será estimar $\Delta_i$ con su esperanza;

$$Q(\Theta_1, \Theta_2) = \sum_{i=1}^N [\mathbb{E}[\Delta_i \vert X; \theta_1, \theta_2] log(\alpha g_1(x;\theta_1)) + $$

$$\mathbb{E}[ (1 - \Delta_i) \vert X; \theta_1, \theta_2] log[(1-\alpha) g_2(x;\theta_2)]]$$

$\Delta_i$ puede tomar dos valores; 1 o 0. Por tanto;

$$\mathbb{E}[\Delta_i \vert X ; \theta_1, \theta_2] = 1 \cdot p(\Delta_i = 1 \vert X ; \theta_1, \theta_2) + 0 \cdot p(\Delta_i = 0 \vert X; \theta_1, \theta_2) =$$

$$= p(\Delta_i = 1 \vert X; \theta_1, \theta_2) $$

Usando la regla de Bayes tenemos que;

$$p(\Delta_i = 1 \vert X ; \theta_1, \theta_2) = \frac{p(\Delta_i = 1)p(X \vert \Delta_i = 1 ; \theta_1, \theta_2)}{p(X; \theta_1, \theta_2)} = $$

$$= \frac{p(\Delta_i = 1)p(X \vert \Delta_i = 1 ; \theta_1, \theta_2)}{\sum_{j=0}^{J = 1} p(\Delta_i = j) p( X \vert \Delta_i = j ; \theta_1, \theta_2)} = $$

$$= \frac{\alpha g_1(X;\theta_1)}{\alpha g_1(x;\theta_1) + (1-\alpha) g_2(x;\theta_2)}$$

Sea $\hat{\gamma} = \mathbb{E}[\Delta_i \vert X ; \theta_1, \theta_2] = p(\Delta_i = 1 \vert X ; \theta_1, \theta_2) $. Podemos reescribir $Q(\Theta_1, \Theta_2)$ como;

$$Q(\Theta_1, \Theta_2) = \sum_{i=1}^N [ \hat\gamma_i log(\alpha g_1(x;\theta_1)) + $$

$$(1 -\hat\gamma_i) log[(1-\alpha) g_2(x;\theta_2)]]$$

Desarrollando $Q(\theta_1, \theta_2)$ tenemos que ;

$$Q(\Theta_1, \Theta_2) = \sum_{i=1}^N [ \hat\gamma_i log(\alpha) + \hat\gamma_i log(g_1(x;\theta_1)) + $$

$$(1 -\hat\gamma_i)log(1-\alpha) + (1 -\hat\gamma_i)log(g_2(x;\theta_2))]$$

#### Paso M

El paso M consiste en maximizar $Q$ usando alguna técnica de optimización;

$$\widehat{\Theta}_1, \widehat{\Theta}_2, \widehat{\alpha} = \arg \max_{\Theta_1, \Theta_2, \alpha} Q(\Theta_1, \Theta_2, \alpha)$$

Una forma de hacerlo sería obteniendo el gradiente de $Q$ e igualándolo a 0;

$$ \frac{\partial Q}{\partial \Theta_1} = \sum_{i=1}^n \Big[\frac{\widehat{\gamma}_i}{g_1(x_i;\Theta_1)} \frac{\partial g_1(x_i;\Theta_1)}{\partial \Theta_1} \Big] = 0$$

$$\frac{\partial Q}{\partial \Theta_2} = \sum_{i=1}^n \Big[\frac{1-\widehat{\gamma}_i}{g_2(x_i;\Theta_1)} \frac{\partial g_2(x_i;\Theta_2)}{\partial \Theta_2} \Big] = 0$$

 $$ \frac{\partial Q}{\partial \alpha} = \sum_{i=1}^n \Big[\widehat{\gamma}_i (\frac{1}{\alpha}) + (1-\widehat{\gamma}_i)(\frac{-1}{1-\alpha})\Big] = 0$$

Desarrollando la última ecuación;

$$\sum_{i=1}^n \Big[ \frac{ \widehat{\gamma}_i - \alpha}{ \alpha(1 - \alpha)} \Big] = 0 \rightarrow \sum_{i=1}^n \Big[ \frac{ \widehat{\gamma}_i}{ \alpha(1 - \alpha)} \Big] - n \frac{\alpha}{\alpha(1-\alpha)} = 0$$

 $$ \frac{1}{ \alpha(1 - \alpha)} \sum_{i=1}^n \widehat{\gamma}_i = \frac{n}{(1-\alpha)} \rightarrow  \sum_{i=1}^n \widehat{\gamma}_i = \frac{n \alpha (1-\alpha)}{(1-\alpha)}$$

$$ \implies \widehat{\alpha} = \frac{1}{n} \sum_{i=1}^n \widehat{\gamma}_i$$

Así que, $\alpha$ no es más que un promedio de todas las $\mathbb{E}[\Delta_i \vert X ; \theta_1, \theta_2]$. Al resolver las derivadas parciales de $Q$ respecto a $\theta_1$ y $\theta_2$ obtenemos los parámetros $\widehat{\theta_1}, \widehat{\theta_2}$ de la siguiente iteración.

### Algoritmo (resumen)

#### Paso E

Para cada observación $x_i$ se calcula la probabilidad a posteriori de que haya sido generada por la primera distribución usando los parámetros de la iteración $t$;

$$p(z_1 \vert x_i ; \theta_1(t), \theta_2(t)) = \frac{\alpha(t) g_1(X;\theta_1(t))}{\alpha(t) g_1(x;\theta_1(t)) + (1-\alpha(t)) g_2(x;\theta_2(t))}$$

#### Paso M

Obtener los parámetros para la siguiente iteración a partir de las derivadas parciales ya vistas.

#### Referencias

Para escribir este post me he basado principalmente en [1].

- [1] Ghojogh, Benyamin, Ghojogh, Aydin, Crowley, Mark, and Karray, Fakhri. Fitting a mixture distribution to data: Tutorial. arXiv preprint arXiv:1901.06708, 2019a
- [2] [Wikipedia: Algoritmo esperanza-maximización](https://es.wikipedia.org/wiki/Algoritmo_esperanza-maximizaci%C3%B3n)
- [3] [Wikipedia: Máxima verosimilitud](https://es.wikipedia.org/wiki/M%C3%A1xima_verosimilitud)

