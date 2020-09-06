---
title: proyectos
permalink: /proyectos/
layout: page
excerpt: 
comments: 
---



# Manifold Learning

1️⃣ [LLE-and-its-variants](https://github.com/JAVI897/LLE-and-its-variants): Locally Linear Embedding es un método de reducción de la dimensión no-lineal. Fue propuesto por Roweis y Saul en el 2000. LLE presenta un enfoque radicalmente diferente a lo que se venía proponiendo en aquel entonces ya que no intentará preservar las distancias entre observaciones sino la estructura local.

En este repositorio se implementa la versión original de LLE y algunas de sus variantes más recientes en Python. La explicación sobre el algoritmo la puedes encontrar en; [Think Globally, Fit Locally; LLE](https://javi897.github.io/LLE/).

2️⃣ [Laplacian-Eigenmaps](https://github.com/JAVI897/Laplacian-Eigenmaps): Laplacian Eigenmaps (LE) es otro método de reducción de la dimensión no-lineal. Fue propuesto en 2003 por Mikhail Belkin y Partha Niyogi. LE construye los *embeddings* usando las propiedades de la matriz Laplaciana.

En este repositorio se implementa una versión de Laplacian Eigenmaps en Python. La descripción del algoritmo la puedes encontrar en; [Laplacian eigenmaps](https://javi897.github.io/Laplacian_eigenmaps/).

3️⃣ [Kernel-PCA](https://github.com/JAVI897/Kernel-PCA): Kernel PCA es una extensión de PCA mucho más reciente que utiliza el “kernel trick” que también es muy usado en otros algoritmos como, por ejemplo, las máquinas de soporte vectorial.

En este repositorio se implementa una versión de Kernel PCA con diversos kernels. La explicación sobre Kernel PCA la puedes encontrar en; [PCA y Kernel PCA](https://javi897.github.io/PCA-y-KernelPCA/) y en este otro post explico algunos de los kernels implementados; [Kernel Functions](https://javi897.github.io/Kernels/).

# ML-Metrics

ML-Metrics es una interfaz programada en *python* mediante el software *streamlit* para poder analizar el punto de corte óptimo de un modelo de clasificación.

La aplicación implementa diferentes curvas como las famosas curvas ROC o PRC para validar hasta 5 modelos diferentes. Los criterios para elegir el punto de corte implementados son: Youden, Distance_PRC, Distance ROC, Difference Recall-Precision, F-score y Difference Sensitivity-Specificity. Por otra parte, las métricas para evaluar el modelo de clasificación implementadas fueron: especificidad, sensibilidad, PPV, AUC, AP y la matriz de confusión.

Repositorio: <a href="https://github.com/JAVI897/ML-Metrics" target="_blank" rel="noopener">ML-Metrics</a>

# Deep-learning-projects

Este es un repositorio de código (en formato notebook normalmente) de proyectos relacionados con el deep learning usando datasets de kaggle u otros. 

| Proyecto                                           |                                                       Código | Temática                                     |
| -------------------------------------------------- | -----------------------------------------------------------: | -------------------------------------------- |
| Deep autoencoder for collaborative filtering       | [Deep-autoencoder](https://github.com/JAVI897/Deep-learning-projects/blob/master/Deep%20autoencoder%20for%20collaborative%20filtering/Deep_autoencoder_for_collaborative_filtering_notebook.ipynb) | Sistema de recomendación                     |
| Image classification with transfer learning        | [Transfer-learning](https://github.com/JAVI897/Deep-learning-projects/blob/master/Image%20classification%20with%20transfer%20learning/Image_classification_with_transfer_learning.ipynb) | Transfer learning                            |
| House price estimation from image and text feature | [Regression](https://github.com/JAVI897/Deep-learning-projects/blob/master/House%20price%20estimation%20from%20image%20and%20text%20feature/House_price_estimation_from_image_and_text_feature.ipynb) | Clasificación de imágenes y regresión        |
| VGG from scratch                                   | [Models](https://github.com/JAVI897/Deep-learning-projects/blob/master/VGG/VGG.ipynb) | Implementación de redes convolucionales      |
| Skin Cancer MNIST: HAM10000                        | [Image-classification](https://github.com/JAVI897/Deep-learning-projects/blob/master/Skin%20Cancer%20MNIST:%20HAM10000/SKIN_CANCER.ipynb) | Transfer learning, multiclass classification |
| Localization of Image Forgeries                    | [Image-localization](https://github.com/JAVI897/Deep-learning-projects/blob/master/Detecci%C3%B3n%20y%20localizaci%C3%B3n%20de%20falsificaci%C3%B3n%20de%20im%C3%A1genes/ManTraNet.ipynb) | Image localization                           |
| Faces segmentation                                 | [Segmentation](https://github.com/JAVI897/Deep-learning-projects/blob/master/Faces%20segmentation/Faces%20segmentation.ipynb) | Segmentación de rostros                      |
| Nails segmentation                                 | [Segmentation](https://github.com/JAVI897/Deep-learning-projects/blob/master/Nails%20segmentation/Nails%20segmentation.ipynb) | Segmentación de uñas                         |
| DCGAN on Simpsons                                  | [GAN](https://github.com/JAVI897/Deep-learning-projects/blob/master/DCGAN%20on%20Simpsons/DCGAN%20on%20Simpsons.ipynb) | Generative adversarial network               |

Repositorio: <a href="https://github.com/JAVI897/Deep-learning-projects" target="_blank" rel="noopener">Deep-learning-projects</a>

