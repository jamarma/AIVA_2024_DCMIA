# Detector de Casas Municipales en Imágenes Aéreas (DCMIA)

## Información

DCMIA es un sistema de detección de casas en imágenes aéreas de alta resolución basado en algoritmos de visión artificial. Dada una imagen, proporciona el número de casas y las coordenadas píxel de los rectángulos delimitadores de cada una de las casas.
Se ha desarrollado para satisfacer las necesidades de un ayuntamiento que busca conocer las casas que tiene su municipio.

![](https://github.com/jamarma/AIVA_2024_DCMIA/blob/main/docs/readme/example.png)

## Índice de contenidos

* [Uso en local](https://github.com/jamarma/AIVA_2024_DCMIA#uso-en-local)
    * [Prerrequisitos](https://github.com/jamarma/AIVA_2024_DCMIA#prerrequisitos)
    * [Instalación](https://github.com/jamarma/AIVA_2024_DCMIA#instalaci%C3%B3n)
    * [Ejecución](https://github.com/jamarma/AIVA_2024_DCMIA#ejecuci%C3%B3n)
        * [Inferencia](https://github.com/jamarma/AIVA_2024_DCMIA#inferencia)
        * [Evaluación](https://github.com/jamarma/AIVA_2024_DCMIA#evaluaci%C3%B3n)
    * [Entrenamiento](https://github.com/jamarma/AIVA_2024_DCMIA#entrenamiento)
        * [Preparación de los datos](https://github.com/jamarma/AIVA_2024_DCMIA#preparaci%C3%B3n-de-los-datos)
        * [Ejecución del entrenamiento](https://github.com/jamarma/AIVA_2024_DCMIA#ejecuci%C3%B3n-del-entrenamiento)
    * [Test unitarios](https://github.com/jamarma/AIVA_2024_DCMIA#test-unitarios)
* [Despliegue](https://github.com/jamarma/AIVA_2024_DCMIA#despliegue)

## Uso en local

En esta sección se muestran los pasos para instalar y ejecutar el sistema de detección de casas en local por línea de comandos.

### Prerrequisitos

Algoritmo testeado en Ubuntu 22.04:

* Python 3.10  
* torch 2.2.1  
* torchvision 0.17.1  
* CUDA 12.1 (opcional, también compatible con ejecución en CPU)

### Instalación

Clonar el repositorio.

```
git clone https://github.com/jamarma/AIVA_2024_DCMIA.git
```

Crea un _venv_ de Python e instala la versión de **torch** y **torchvision** indicada en [Prerrequisitos](https://github.com/jamarma/AIVA_2024_DCMIA#prerrequisitos).

A continuación, instala el resto de dependencias conforme a los siguientes pasos.

```
cd AIVA_2024_DCMIA/dcmia
DCMIA_ROOT=$(pwd)
pip install -r requirements.txt
```

Descargar modelo entrenado de detección de casas.
```
mkdir models; cd models
gdown https://drive.google.com/uc?id=1lKNUt3BgYel5lC5Hnq88wEF80Qix4Zxy
```

Descargar imágenes para probar y evaluar el algoritmo. Esto es un paso opcional, sólo es necesario si no tienes tus propias imágenes.

```
cd $DCMIA_ROOT
mkdir -p data/raw; cd data/raw
gdown --folder https://drive.google.com/drive/folders/1PSpBUJ381ENDvrEDP4qCrHzIQJ5rh6sk
```

### Ejecución

Los scripts ejecutables se encuentran en el directorio raiz $DCMIA_ROOT. 

```
cd $DCMIA_ROOT
```

#### Inferencia

Para probar el algoritmo ejecute

```
python main.py --image_path data/raw/test/images/austin1.tif
```

Si desea guardar una imagen con el resultado ejecute

```
python main.py --image_path data/raw/test/images/austin1.tif --output_image ./output.png
```

Si desea guardar un fichero txt con el número de casas detectadas y las coordenadas de los bounding boxes ejecute

```
python main.py --image_path data/raw/test/images/austin1.tif --output_results ./results.txt
```

#### Evaluación

Para evaluar el rendimiento del algoritmo ejecute

```
python main.py --image_path data/raw/test/images/austin1.tif --mask_path data/raw/test/masks/austin1.tif
```

### Entrenamiento

#### Preparación de los datos

Para entrenar un modelo con tus propios datos, organiza tus imágenes siguiendo esta estructura de directorios.

```
$DCMIA_ROOT
├── data
    └── raw
        └── train
            └── images
                ├── image1.tif
                ├── image2.tif
                └── ...
            └── masks
                ├── image1.tif
                ├── image2.tif
                └── ...
```

En el directorio `images` se almacenan las imágenes de entrenamiento RGB y en el directorio `masks` se almacena el ground truth en forma de máscara binaria para cada imagen de `images`. Todas las imágenes deben estar en formato .tif.

Ejecuta el siguiente script para generar una base de datos de parches de imagen en formato PASCAL VOC a partir de tus imágenes.

```
cd $DCMIA_ROOT/scripts

python build_images_dataset.py
```

#### Ejecución del entrenamiento

```
cd $DCMIA_ROOT
python train.py
```

Los parámetros del entrenamiento se pueden modificar en `$DCMIA_ROOT/src/dcmia/constants.py`.

El modelo entrenado se almacena en el directorio `$DCMIA_ROOT/models` con el nombre `model.pth`.

### Test unitarios

Descargar las imágenes necesarias para pasar los test.

```
cd $DCMIA_ROOT
cd test; mkdir data; cd data
gdown https://drive.google.com/uc?id=1x_3GxaDtfInjAZEWjxnXqWY68qR8snzQ
```

Ejecutar los test unitarios.

```
cd $DCMIA_ROOT/test

python -m unittest -v test_house_detector.py test_prediction_patch_matrix.py test_house_detector_evaluator.py
```

## Despliegue

Se ha construido una imagen docker que proporciona el despliegue del sistema en un servidor Flask local.

> IMPORTANTE: si deseas habilitar el uso de la GPU es necesario instalar el paquete [nvidia-container-toolkit](https://docs.docker.com/config/containers/resource_constraints/#gpu) y disponer de una GPU compatible con CUDA 12.1. Sin embargo, esto no es un paso necesario, el sistema también está preparado para correr en CPU.

Descarga la imagen

```
docker pull jamarma/dcmia-app:latest
```

Lanza el contenedor con el servidor

```
docker run --rm --gpus all -p 5000:5000 jamarma/dcmia-app
```

Abre un navegador y dirígente a http://localhost:5000/. Podrás usar el sistema DCMIA desde una interfaz web.

![](https://github.com/jamarma/AIVA_2024_DCMIA/blob/dev/docs/readme/app-example.png)
