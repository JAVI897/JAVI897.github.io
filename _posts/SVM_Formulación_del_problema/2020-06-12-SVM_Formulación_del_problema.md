---
title: SVM Formulación del problema desde un punto de vista 'real'
date: 2020-07-12 11:58:49 +07:00
---



Durante este post usaremos el siguiente conjunto de datos $ \pi = \{(x_i, y_i) | x_i \in \mathbb{R}^p, y_i \in \{Austrias, Borbones\} \}^m_{i=1}$ . Como representantes de la dinastía borbónica he escogido a mí trío favorito: Carlos IV, su esposa María Luisa de Parma y Manuel Godoy. Sí, Godoy no es ningún borbón, pero existen  razonables dudas sobre la parte que le ocupó a Carlos en los 14 hijos de María Luisa de Parma y si a esto le añadimos el razonable parecido de estos con Godoy junto a la promiscuidad y desatención de la pobre María Luisa por parte de su marido...

Como representantes de la dinastía de los austrias he escogido a Juana I *la Loca*, Carlos II *el Hechizado* y el príncipe don Carlos (primogénito de Felipe II). Creo que estos tres personajes representan a la perfección la demencia y los frutos de la consanguinidad que caracterizaron a la dinastía de los austrias. 

Respecto al poco conocido Carlos, creo que merece algunas palabras. El pobre nunca llegó a ser rey, murió a manos de su malvado padre (Felipe II) o esa es la versión que se nos ha intentado vender por parte de los británicos. En realidad, Carlos era un sádico con graves problemas mentales. Sus problemas se manifestaron bien pronto, cuando era bebé. En aquella época era habitual que la reina no amamantase a sus hijos sino que lo hiciesen las nodrizas, bien, a nuestro amiguito Carlos no le convencía ninguna teta ya que mordía los pezones de sus amas hasta llagarlos y estas terminaban muriendo debido a la sepsis provocada por las heridas. Intentaron que una cabra amamantase a Carlos pero él era más listo que sus amas; él quería carne humana. Su vida no duró mucho, murió a los 23 años tras un encierro (por voluntad de su malvado padre) que acrecentó su locura.

<img src="C:\Users\Usuario\Desktop\JAVI897.github.io\_posts\SVM_Formulación_del_problema\familia-real.png" style="zoom:20%;" />

Un dataset $D = \{(x_i, y_i) | x_i \in \mathbb{R}^p, y_i \in \{-1, 1\} \}^m_{i=1}$ linealmente separable puede clasificarse usando varios hiperplanos diferentes. Un modelo de clasifiación como el perceptrón encontrará miles de hiperplanos en cada ejecución. Las SVM en vez de encontrar un hiperplano cualquiera, encuentran *el* señor hiperplano, el que mejor separa entre los datos.



<p float="middle">
	<img src="https://github.com/JAVI897/JAVI897.github.io/blob/master/_posts/SVM_Formulaci%C3%B3n_del_problema/Diferentes-hiperplanos.png?raw=true" style="zoom:15%;" />
    <img src="https://github.com/JAVI897/JAVI897.github.io/blob/master/_posts/SVM_Formulaci%C3%B3n_del_problema/hiperplano-optimo.png?raw=true" style="zoom:15%;" />
</p>

