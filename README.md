# Creador de Embeddings local para ***Búsqueda Semántica***
Crea embeddings usando el modelo de lenguaje natural ***paraphrase-multilingual-MiniLM-L12-v2***, usando un WebService REST como interfaz.

### En la ruta ***/process*** del WebService
Recibe un POST, esperando 2 parámetros:
- mode: Que puede ser (single | chunks)
- textbase64: Sería el texto en Base64

Ejemplo:
```json
    {
        "mode": "single",
        "textbase64": "SG9sYSBNdW5kbyE="
    }
```


Cuando el **mode** es ***single***, el texto pasado en Base64, se limpia y se crea su embedding. Devolviendo la respuesta con el formato:
```json
    {
        "mode": "single",
        "text": Texto recibido y limpio,
        "embedding": arreglo de 128 números de punto flotante.
    }
```

Cuando el **mode** es ***chunks***, el texto pasado en Base64, se limpia, se separa en chunks y a cada chunk se le crea su embedding. La respuesta devuelta tendría el formato siguiente:
```json
    {
        "mode": "chunks",
        "chunks": Arreglo chunks,
        "embedding": Arreglo de embeddings
    }
```

### En la ruta ***/test*** del WebService
Con una operación GET, se recibe en el parámetro **base64text**, un texto en *Base64*.
Retornando una salida con el siguiente formato: 
```json
    {
        "mode": "test",
        "text": Texto recibido y limpio,
        "embedding": arreglo de 128 números de punto flotante.
    }
```
Ejemplo de uso: ```http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE=```


### La ruta ***/health***, igualmente en ***/*** del WebService
Sirve para comprobar que el servicio esta trabajando.

## **Actualización 16 Mayo 2025**
* Se agrega el cálculo de **Keywords** al procesar *embeddings*; será parte de la respuesta. 
  - Se calcularían 12 Keywords por todo el documento, y hasta 5 keywords por chunk devuelto.
* También se agrega la ruta ***/keywords*** del WebService, para el caso donde sólo se necesite este cálculo.
Sería con un POST, esperando los parámetros:
    - textbase64: Sería el texto en Base64.
    - quantity: Entero positivo que indique cuantas palabras se devolverían. 12 sería el default si no indica el parámetro o se indica mal. El máximo es 20. La cantidad resultante puede ser menor al lematizar.
Ejemplo:
```json
    {
        "textbase64": "SG9sYSBNdW5kbyE=",
        "quantity": 10
    }
```
 - Se usa **keybert**, con el mismo modelo **paraphrase-multilingual-MiniLM-L12-v2**.
 - Se aplican **StopWords** en inglés y español de **nltk**.
 - Se lematizan los resultados, para devolver la raiz de la palabra con **Spacy** usando el modelo **es_core_news_md**. Además de evitar repertir con esto palabras parecidas.
---

## Modelo de Embeddings
Se utiliza el modelo ***paraphrase-multilingual-MiniLM-L12-v2***, que es de tamaño medio y funcional para texto en varios idiomas (inglés y español incluidos). Con este modelo se generan vectores de 384 dimensiones.

Para evitar tener un esquema tan grande y previendo una indexación de millones de embeddings, se han reducido a 128 dimensiones, usando una ***matriz PCA de reducción***, previamente entrenada, incluida en el proyecto.

### Acerca de la "Limpieza del texto" recibido
La limpieza del texto antes de trabajarlo, es indispensable para que el modelo de procesamiento de lenguaje natural **es_core_news_md**, cumpla adecuadamente su función al separar en oraciones.

En este orden:
- Se quita todo el contenido delimitado por '***[[RS-***'  '***-RS]]***'.
- Se quitan todos los tokens : '***' (triple asterisco).
- Se quitan todos los tokens : '***[[¡***'.
- Se reemplazan todos los tokens : '***!]]***' por '```. ```'.
- Se reemplazan todos los tokens : '***!]]***' por '```. ```'.
- Se reemplazan todos los tokens : ``` '<br /><br />'```' por '```\n```'.
- Se reemplazan todos los tokens : ``` '<br/><br/>'``` por '```\n```'.
- Se reemplazan todos los tokens : ``` '<br/>'``` por '```\n```'.
- Se reemplazan todos los tokens : ``` '<br />'``` por '```\n```'.
- Se reemplazan todos los tokens : ``` ' —————————— '``` por '```\n```'.
- Se reemplazan todos los tokens : ``` ' ___ '``` por '```\n```'.
- Se eliminan el resto de TAGs de html.
- Se quitan todos los tokens : '***[[03]]***'.
- Se quitan todos los tokens : '***[[05]]***'.
- Normaliza diferentes tipos de comillas dobles y simples a sólo tener las simples básicas.
- Se reemplazan todos los tokens : '```\r\n\r\n```' por '```\r\n```'.


### Lógica de separación de texto largo en varios Chunks
Una vez limpio el texto y aplicando el modelo de procesamiento de lenguaje natural **es_core_news_md** con Spacy; se separa en oraciones y aplicando  ***Similaridad de Coseno*** con un umbral de **0.6** de similitud, se agrupan oraciones consecutivas en un mismo ***chunk***.

En resumen, se obtienen embeddings para cada oración, y usando operaciones de similitud entre esas oraciones consecutivas, se determinan si varias oraciones se pueden agrupar a un mismo chunk o se manejan en chunks separados. 

También se aplica la regla de que cada chunk tenga a al menos 150 caracteres. Con esto, oraciones muy cortas se agrupan en otras.

---

# Alternativas de uso
 - En máquina local: Con un ambiente en Python en toda regla. Sirve para desarrollo.
 - En Producción hay varias opciones:
    1. También con un ambiente Python, pero sirviendo la aplicación con Waitress.
    2. Ejecutable (EXE), pero es dificil de mantener; no tan recomendable y se vuelve un ejecutable enorme, dificil de hacer cambios.
    3. Administrar como Servicio de Sistema. Usando NSSM (al final requiere su ambiente Python y también se sirve con Waitres, pero NSSM lo registra como servicio).
    4. Como Imagen Docker (Docker + Flask + Waistress + NGINX). Es la opción ideal. Es lo más limpio y portable. Fácil de mover a Linux o la Nube ó escalar con Docker Compose / Kubernets.
    5. Como Imagen Docker (Docker + Flask + Waistress). Es la opción ideal, si se instala en una infraestructura que ya cuenta con las funciones de "Balanceo de Carga" y "Proxy Inverso" (Servidores IIS). En estos casos, NGINX ya no es necesario.

## Instalación para Desarrollo con su ambiente Python
Si quieres tener la aplicación funcionando en tu entorno de desarrollo:
 1. Clona el repositorio : ```git clone https://github.com/Hitokiri77777/Python-Embeddings-128.git```
 2. En tu terminal, cámbiate al folder creado.
 3. Crea el ambiente virtual con python (Se probó en Python 3.11.9). Usar : ``` python -m venv venv ```
 4. Activa el ambiente virtual:
    - En Windows : ``` venv\Scripts\activate ```
    - En Linux   : ``` source venv/bin/activate ```
 5. Instala las dependencias: ``` pip install -r requirements.txt ``` 
 6. Ejecuta la aplicación : ``` python app.py ```

 * Podría usarse un servidor como "waitress" de Python, para mejorar respuestas de la aplicación flask:
   - Instala el paquete ```pip install waitress``` con el ambiente activado.
   - Sirve la aplicación de manera más eficiente con el modelo multihilo de waitress con esto:

       ```waitress-serve --host=0.0.0.0 --port=5000 app:app```

- Ver prueba básica de creación de embedding en la última sección del documento.
- Con la demostración exitosa de la prueba; puedes ya hacer *POST* a la ruta **/process** para trabajar. 

---

## Creación de Imagen para ***Docker***
Usar el archivo **Dockerfile** en la raíz del proyecto, para creación y puesta en marcha del contenedor. 

1. Teniendo Docker instalado. Hacer : ```docker build -t python_embeddings .```
   para crear la imagen.
2. Pruébalo con  : ```docker run -it --rm -p 5000:5000 python_embeddings```
   Con esto, el puerto 5000 de la imagen se mapeará al también 5000 de tu máquina, e irás viendo los mensajes en consola.
3. Ahora si podrías hacer la prueba: ```http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE=```

**10-Abril-2024**
Se actualizó archivo ***Dockerfile***, donde se pulen y toman en cuenta varios aspectos:
- Debería ser ahora una imagen más estable, incluye variables de entorno y ya no se ejecuta como root.
- Lo más reducida posible: Ya no se incluyen compiladores, descargables, temporales y se reducen capas de creación de Docker.
- Ya incluiría el modelo *"paraphrase-multilingual-MiniLM-L12-v2"* cacheado dentro de la imagen.
* Se comprobó que la imagen es perfectamente funcional y sin errores.
* El tamaño final de la imagen es de 2.16 GigaBytes.


### Sugerencia para llevarlo a Producción con NGINX
Se puede poner esta aplicación detrás de un servidor ***Nginx*** para mejorar rendimiento, seguridad y escalabilidad.

Para hacerlo, puedes usar el archivo ***docker-compose.yml***. Esto crearía el flujo siguiente:
1. El usuario accede a http://localhost/.
2. Nginx recibe la solicitud en el puerto 80.
3. Nginx reenvía la solicitud al contenedor de Flask en backend:5000.
4. Flask procesa la petición y devuelve la respuesta a Nginx.
5. Nginx envía la respuesta final al usuario.

Utilizarías el comando ```docker-compose up --build -d```
- Esto crearía las 2 imágenes, una para la App de Flask, nuestra app. Y otra con un linux Alpine súper ligero, con un servidor de Nginx. Y un "puente de red" para hacer que las dos imágenes trabajen en conjunto.

Tener una aplicación Flask detrás de un servidor Nginx es una práctica recomendada en entornos de producción porque mejora la seguridad, el rendimiento y la escalabilidad. Nginx es un servidor web ligero y eficiente que actúa como intermediario entre los clientes (navegador, API, etc.) y la aplicación Flask.

### Beneficios en producción
- Optimización de tráfico → Nginx maneja archivos estáticos, evitando que Flask se sobrecargue.
- Seguridad → Evita que los clientes accedan directamente a Flask.
- Escalabilidad → Puedes agregar múltiples instancias de Flask detrás de Nginx.
- HTTPS fácil → Puedes configurar un certificado SSL en Nginx sin tocar Flask. 

## Creación de Imagenes en Docker (Flask y Nginx)
Teniendo Docker instalado, cámbiate al directorio de la aplicación. Y con el comando ```docker-compose up --build```
Eso crearía 2 imágenes en Docker dentro de una red.

Comprueba que las 2 imágenes se estan ejecutando  con ```docker ps```

Comprueba que Nginx recibe las llamadas las direcciona a Flask:
```http://127.0.0.1/test?base64text=SG9sYSBNdW5kbyE=```
A diferencia de las pruebas anteriores, aqui no se indica el puerto 5000.


## Prueba básica de creación de Embedding
- Usa: http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE=
    ó, cuando ya hay un servidor Nginx al frente: http://127.0.0.1/test?base64text=SG9sYSBNdW5kbyE=
  - Verás el ***Embedding*** de resultado, con el texto recibido ya limpio.
  - Se usa *Base64*, porque el texto puede contener caracteres que pueden chocar con el esquema de una URL correcta o de un JSON bien formado. Aplica igual para el caso de los datos JSON recibidos en los *POST*.
  - Es posible que textos muy largos en *Base64*, sobrepasen el límite para *GET* de ***/test***, por si vas a probar textos propios en sea ruta **/test**.
