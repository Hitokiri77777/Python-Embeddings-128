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
* Se agrega el cálculo de **Keywords** al procesar *embeddings*; será parte de la respuesta.  Se devolverá un arreglo de hasta 12 keywords por todo el documento. Y máximo 5 por cada chunk generado. 
* También se agrega la ruta ***/keywords*** al WebService, para el caso donde sólo se necesite este cálculo.
Recibiendo un POST que espera los parámetros:
    - textbase64: Sería el texto a procesar en Base64.
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
 * Prueba básica para **Keywords**: 
 ```
 http://127.0.0.1:5000/test?base64text=RWwgY2FtYmlvIGNsaW3DoXRpY28gZXN0w6EgcHJvdm9jYW5kbyB1biBhdW1lbnRvIGRlbCBuaXZlbCBkZWwgbWFyIHkgZXZlbnRvcyBjbGltw6F0aWNvcyBleHRyZW1vcyBlbiBtdWNoYXMgcGFydGVzIGRlbCBtdW5kby4=
 ```
---

## **Actualización 19 Mayo 2025**
* Se agrega el cálculo de **Entities** también al procesar *embeddings*. Véase NER (Named Entity Recognition), es un componente del procesamiento de lenguaje natural (PLN) que identifica categorías predefinidas de objetos en un texto. Categorías pueden incluir, entre otras, nombres de personas, organizaciones, ubicaciones, expresiones de tiempos, cantidades, códigos médicos, valores monetarios y porcentajes. 
  - Se retornaría un arreglo de tuplas "(text, label)" que definirán las entidades por todo el documento. Y un arreglo también por cada chunk.
* También se agrega la ruta ***/entities*** al WebService, para el caso donde sólo se necesite este cálculo. Aqui se recibe un POST con el parámetro **textbase64**: Sería el texto a procesar en Base64.

* Prueba básica para **Entidades**: 
```
http://127.0.0.1:5000/test?base64text=R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu
```

Ejemplo:
```json
    {
        "textbase64": "R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu"
    }
```
 - Se usa **Spacy** con el modelo **es_core_news_md**.
 - Se evita devolver entidades repetidas.
 - Se evitan las entidades de etiqueta 'MISC'. 
---

## **Actualización 20 Mayo 2025**
Se agregan 2 modos en la ruta ***/process*** del WebService:
- Modo: **extended**
Hace lo mismo que en el modo **chunks**, pero también genera
    * *keywordsFullDocument* Lista de Keywords de todo el documento.
    * *entitiesFullDocument* Lista de entidades de todo el documento. 
    * *embeddingFullDocument* Embedding de todo el documento.

- Modo : **full**
    Es igual que en **extended**, pero además permite generar los *embeddings*, *keywords* y *entidades* de un *título* y un *resumen* separados del *texto completo*. Por lo que espera los siguientes parámetros extra en el POST, aunque no son obligatorios:
    * *titlebase64*. Título en *base64*.
    * *summarybase64*. Resumen en *base64*.

    Por lo que se agregarían 6 datos extra en la respuesta:
    * *titleEmbedding*
    * *titleKeywords*
    * *titleEntities*
    * *summaryEmbedding*
    * *summaryKeywords*
    * *summaryEntities*

Estos 2 nuevos *modos* de la misma ruta ***/process*** del WebService, permitiría generar índices vectoriales con mucha información detallada en los metadatos, o incluso poder crear índices vectoriales complejos con esquemas de campos anidados.

**Esto permitiría búsquedas sobre los datos muy potentes y variadas: búsquedas por keywords, búsquedas por entidades, clasificación de documentos, búsquedas semánticas sólo por *título* y/ó *resumen*, comparación de documentos completos similares, etc**.

Además que habría información extra muy útil para enviarse a un LLM en caso de que se esté creando un RAG, como de entidades y keywords.

*Entendemos que las llamadas al WebService para estos cálculos, pueden ser muy demandantes en recursos e incluso en tiempo, por lo que se da opción a usar el modo que mejor se adapte a las necesidades de cáda índice.*



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

       ó 

       ```python -m waitress --host=0.0.0.0 --port=5000 app:app```

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
- Usa: http://127.0.0.1:5000/test?base64text=R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu
    ó, cuando ya hay un servidor Nginx al frente: http://127.0.0.1/test?base64text=R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu
  - Verás el ***Embedding*** de resultado, con el texto recibido ya limpio.
  - Se usa *Base64*, porque el texto puede contener caracteres que pueden chocar con el esquema de una URL correcta o de un JSON bien formado. Aplica igual para el caso de los datos JSON recibidos en los *POST*.
  - Es posible que textos muy largos en *Base64*, sobrepasen el límite para *GET* de ***/test***, por si vas a probar textos propios en sea ruta **/test**.
  - Aqui ya verías los nuevos Keywords y Entities calculados.
