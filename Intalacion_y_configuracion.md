# Instalación y configuración de Anaconda con la instalación de paquetes.



![](imag\img1.png)

Como primer paso descargamos el instalador y lo ejecutamos como administrador.

![](imag\img2.png)

Aceptamos la licencia y continuamos.

![](imag\img3.png)

Hay dos opciones para instalar, en mi caso escogí la primera que es la recomendada.

![](imag\img4.png)

En este elegimos el destino de donde se va aguardar la instalación, lo el que está por defecto.


![](imag\img5.png)

En este solo le di click en instalar.

![](imag\img6.png)

Hay que esperar hasta que termine la instalación.

![](imag\img7.png)

Y por último finalizamos, y lo abrimos por si lo va a necesitar

Con el comando jupyter notebook probamos en la Shell de Anaconda si la instalación es correcta, si lo es nos manda  a la web.

![](imag\img8.png)
![](imag\img9.png)

## 2.	Uso de paquetes predeterminados en Jupyter Notebook

### a.	Explore e investigue que paquetes se integran con la instalación de jupyter Notebook de Anaconda 

![](imag\img13.png)

### b.	Estructura de un cuaderno de jupyter notebook

### - ¿Cómo está dividido un cuaderno de jupyter notebook?

El cuaderno Jupyter consta de tres componentes.

La aplicación web del cuaderno: Para escribir y ejecutar código en un entorno interactivo.

Núcleos: Kernel es un motor de cálculo que ejecuta el código de los usuarios presente en el documento del cuaderno en un idioma determinado y devuelve la salida a la aplicación web del cuaderno. El kernel también se ocupa de cálculos para widgets interactivos, introspección y finalización de pestañas.

Documentos de cuaderno: Un documento es una representación de todo el contenido visible en la aplicación web del cuaderno. Esto incluye entrada y salida de cálculos, gráficos, texto narrativo, widgets interactivos, ecuaciones, imágenes y video. Hay un kernel separado para cada documento. Los documentos del cuaderno se pueden convertir a varios formatos y compartir entre otros mediante el correo electrónico, Dropbox y herramientas de control de versiones como git.



### - ¿Cómo ejecutar código?
Para ejecutar la celda, necesitamos agregar algo a la celda.
La celda puede contener las declaraciones escritas en un lenguaje de programación del kernel actual. Elegimos el kernel cada vez que creamos un nuevo cuaderno. Recuerde que creamos un cuaderno que elegimos Python 3. Eso significa que nuestra celda puede contener un código Python.


### - Exportar a formatos estáticos como Markdown, HTML y PDF

![](imag\img14.png)


### 3.	Instalación de paquetes adicionales recomendados 
### a.	Instalación de paquetes usando Anaconda


![](imag\img10.png)

scikit-learn es una librería de python para Machine Learning y Análisis de Datos. Está basada en NumPy, SciPy y Matplotlib. ... Con scikit-learn podemos realizar aprendizaje supervisado y no supervisado. Podemos usarlo para resolver problemas tanto de clasificación y como de regresión.

![](imag\img11.png)

pandas es una librería para el análisis de datos que cuenta con las estructuras de datos que necesitamos para limpiar los datos en bruto y que sean aptos para el análisis (por ejemplo, tablas).

![](imag\img12.png)

Numpy es una biblioteca para Python que facilita el trabajo con arrays (vectores y matrices), un tipo de dato estructurado muy utilizado en análisis de datos, en informática científica y en el área del aprendizaje automático (learning machine).

![](imag\img15.png)

TensorFlow es un sistema de programación que utiliza gráficos para representar tareas informáticas. En este momento, puede ejecutar Python y luego importar tensorflow como tf.

![](imag\img16.png)

PyTorch. Es un paquete de Python diseñado para realizar cálculos numéricos haciendo uso de la programación de tensores. Además, permite su ejecución en GPU para acelerar los cálculos.

### Pruebe instalando cualquier paquete con cada uno de los gestores e investigue la carpeta de instalación.

![](imag\img17.png)

