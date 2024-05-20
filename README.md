# Machine Learning Module

For the Spanish version, click [here](#Módulo-de-Machine-Learning)

## **Repository Contents**

This repository contains my solution for the practice exercise of this module, where we were asked to tackle a realistic Machine Learning problem using an Airbnb dataset, following the methodology and best practices explained throughout the module.

**Machine Learning (ML)** is a branch of artificial intelligence (AI) that focuses on the development of algorithms and models that allow computers to learn and make predictions or decisions based on data. Unlike systems explicitly programmed to perform a specific task, machine learning systems identify patterns and relationships in the data and use these findings to improve their performance over time without constant human intervention.<br><br>

## **Module Contents**

## Regularization (Ridge and Lasso)
**Regularization** is a technique used to prevent overfitting in ML models. Two common approaches are:

- **Ridge (L2)**: Adds a penalty to the model coefficients to prevent them from becoming too large. Helps reduce variance and improve generalization.
- **Lasso (L1)**: Similar to Ridge but also performs automatic feature selection by forcing some coefficients to zero. Useful when reducing the number of features is desired.

## Feature Selection
Various methods exist for selecting relevant features:

- **Filter Methods**: Evaluate the relevance of each feature individually. Examples include F-Test and Mutual Information.
- **Wrapper Methods**: Perform feature selection based on classifier performance (e.g., Recursive Feature Elimination).
- **Embedded Methods**: Incorporate feature selection directly into the model training process (e.g., L1 regularization in Lasso).

## Decision Trees, Bagging, and Random Forest
- **Decision Trees**: Decision models based on tree structures that split data into branches based on features.
- **Bagging**: Technique that combines multiple models (e.g., trees) trained on random subsets of data to improve accuracy.
- **Random Forest**: A type of Bagging that uses multiple decision trees and combines their results for a more robust prediction.

## Boosting and Support Vector Machines (SVMs)
- **Boosting**: Combines several weak models to create a stronger one. Examples include AdaBoost and Gradient Boosting.
- **SVMs**: Classification algorithms that seek to find the optimal separation hyperplane between classes.

## Metrics
Common evaluation metrics include:
- Precision, Recall, F1-Score for classification.
- Mean Squared Error (MSE) for regression.<br><br><br><br>

# Módulo de Machine Learning

Para versión en inglés haz clic [aquí](#Machine-Learning-Module)

### **Contenido del repositorio**

Este repositorio contiene mi solución para la práctica de este módulo en la cual se nos pedía abordar un problema de Machine Learning realista sobre un DataSet de Airbnb, siguiendo la metodología y buenas prácticas explicadas a lo largo del módulo.

El **Machine Learning (ML)** es una rama de la inteligencia artificial (IA) que se centra en el desarrollo de algoritmos y modelos que permiten a las computadoras aprender y hacer predicciones o decisiones basadas en datos. A diferencia de los sistemas programados explícitamente para realizar una tarea específica, los sistemas de machine learning identifican patrones y relaciones en los datos y utilizan estos hallazgos para mejorar su rendimiento a lo largo del tiempo sin intervención humana constante.<br><br>

## **Contenido del modulo**

## Regularización (Ridge y Lasso)
La **regularización** es una técnica utilizada para evitar el sobreajuste en modelos de ML. Dos enfoques comunes son:

- **Ridge (L2)**: Agrega una penalización a los coeficientes del modelo para evitar que sean demasiado grandes. Ayuda a reducir la varianza y mejora la generalización.
- **Lasso (L1)**: Similar a Ridge, pero también realiza selección automática de características al forzar algunos coeficientes a cero. Útil cuando se desea reducir el número de características.

## Selección de Características
Existen varios métodos para seleccionar características relevantes:

- **Métodos de Filtrado**: Evalúan la relevancia de cada característica de forma individual. Ejemplos incluyen F-Test y Mutual Information.
- **Métodos Wrapper**: Realizan selección de características basada en el rendimiento de un clasificador (por ejemplo, Recursive Feature Elimination).
- **Métodos Embedded**: Incorporan la selección de características directamente en el proceso de entrenamiento del modelo (por ejemplo, L1 regularización en Lasso).

## Árboles de Decisión, Bagging y Random Forest
- **Árboles de Decisión**: Modelos de decisión basados en estructuras de árbol que dividen los datos en ramas según características.
- **Bagging**: Técnica que combina múltiples modelos (por ejemplo, árboles) entrenados en subconjuntos aleatorios de datos para mejorar la precisión.
- **Random Forest**: Un tipo de Bagging que utiliza múltiples árboles de decisión y combina sus resultados para obtener una predicción más robusta.

## Boosting y Support Vector Machines (SVMs)
- **Boosting**: Combina varios modelos débiles para crear un modelo más fuerte. Ejemplos incluyen AdaBoost y Gradient Boosting.
- **SVMs**: Son algoritmos de clasificación que buscan encontrar el hiperplano óptimo de separación entre clases. Utilizan funciones kernel para mapear datos a un espacio de características de mayor dimensión.

## Métricas
Las métricas evalúan el rendimiento del modelo. Algunas comunes son:
- Precisión, Recall, F1-Score para clasificación.
- Error cuadrático medio (MSE) para regresión.