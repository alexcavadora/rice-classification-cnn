# Detalle de los cambios

## Lenet
### Hipotesis 1
Se cambió la función de tanh a ReLU que en combinación con el cambio al pooling de average a MaxPooling, resultará en una mejor detección de bordes de los granos. El padding nos ayuda a preservar  la información de los bordes de la imagen de entrada. Le agregamos una capa de aprendizaje y dropout para prevenir el overfitting. Otro cambio pequeño fue el adaptive pooling, que permite entradas de cualquier tamaño de imagen. La nueva versión debería tener mayor accuracy, una convergencia más rápida, mayor generalización gracias al dropout, pero tiempo de entrenamiento podría ser más lento debido a la capa adicional. Si bien la tarea del arroz es relativamente simple y el modelo base es funcional, debido a esto mismo, al entrenarlo se aprecia overfitting.

### Hipotesis 2
lorem ipsum dolor

## VGG16

### Hipotesis 1
Se plantea que una versión reducida de VGGNet, con un número menor de filtros (de 8 a 64 en lugar de 64 a 512) y un dropout del 90 \% en las capas totalmente conectadas, puede mantener un buen desempeño al tiempo que disminuye la complejidad computacional y mejora la generalización del modelo frente al sobreajuste.

### Hipotesis 2
lorem ipsum dolor

## AlexNet

### Hipotesis 1
lorem ipsum dolor

### Hipotesis 2
lorem ipsum dolor
