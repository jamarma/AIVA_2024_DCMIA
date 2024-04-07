# Detector de Casas Municipales en Imágenes Aéreas (DCMIA)

## Información

DCMIA es un sistema de detección de casas en imágenes aéreas de alta resolución basado en algoritmos de visión artificial. Dada una imagen, proporciona el número de casas y las coordenadas píxel de los rectángulos delimitadores de cada una de las casas.
Se ha desarrollado para satisfacer las necesidades de un ayuntamiento que busca conocer las casas que tiene su municipio.

![](https://github.com/jamarma/AIVA_2024_DCMIA/blob/main/docs/readme/example.png)

## Prerrequisitos

Algoritmo testeado en Ubuntu 22.04:

* Python 3.10  
* torch 2.2.1  
* torchvision 0.17.1  
* CUDA 12.1

## Instalación

Clonar el repositorio.

```
git clone https://github.com/jamarma/AIVA_2024_DCMIA.git
cd AIVA_2024_DCMIA
DCMIA_ROOT=$(pwd)
```

Instalar de paquetes de Python requeridos.

```
pip install -r requirements.txt
```

Descargar modelo entrenado de detección de casas.
```
mkdir models
gdown https://drive.google.com/uc?id=1lKNUt3BgYel5lC5Hnq88wEF80Qix4Zxy
```

Descargar imágenes para probar y evaluar el algoritmo.

```
cd $DCMIA_ROOT
mkdir -p data/raw
cd data/raw
gdown --folder https://drive.google.com/drive/folders/1PSpBUJ381ENDvrEDP4qCrHzIQJ5rh6sk
```

## Ejecución

Todos los scripts se deben ejecutar desde dentro de la carpeta `src` del proyecto.

```
cd $DCMIA_ROOT/scr
```

### Inferencia

Para probar el algoritmo ejecute

```
python main.py --image_path ../data/raw/test/images/austin1.tif
```

Si desea guardar una imagen con el resultado ejecute

```
python main.py --image_path ../data/raw/test/images/austin1.tif --output_path ./output.png
```

### Evaluación

Para evaluar el rendimiento del algoritmo ejecute

```
python main.py --image_path ../data/raw/test/images/austin1.tif --mask_path ../data/raw/test/masks/austin1.tif
```

## Entrenamiento

### Preparación de los datos

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

### Ejecución del entrenamiento

```
python train.py
```

## Test unitarios

Descargar las imágenes necesarias para pasar los test.

```
cd $DCMIA_ROOT
cd test; mkdir data; cd data
gdown https://drive.google.com/uc?id=1x_3GxaDtfInjAZEWjxnXqWY68qR8snzQ
```

Ejecutar los test unitarios

```
cd $DCMIA_ROOT/test

python -m unittest -v test_house_detector.py test_prediction_patch_matrix.py test_house_detector_evaluator.py
```