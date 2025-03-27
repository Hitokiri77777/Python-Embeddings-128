# Creador de Embeddings local
Crea embeddings usando el modelo de lenguale natural **paraphrase-multilingual-MiniLM-L12-v2**.

Se accesa desde un WebService REST
#### En la ruta /process
Recibe un POST, esperando 2 parámetros
- mode : Que puede ser (single | chunks)
- textbase64 : Sería el texto en Base64


Cuando el **mode** es single, el texto pasado en Base64, se limpia y se crea su embedding. Y se devuelve la respuesta:
```json
    {
        "mode": "single",
        "text": Texto recibido y limpio,
        "embedding": arreglo de 128 números de punto flotante.
    }

Cuando el **mode** es chunks, el texto pasado en Base64, se limpia, se separa en chunks y a cada chunk se le crea su embedding. Y se devuelve la respuesta:
```json
    {
        "mode": "chunks",
        "chunks": Arreglo chunks
    }

#### En la ruta /test
Se recibe en un GET, se recibe en el parámetro **base64text**, un texto en Base64.
Retornando una salida con el siguiente formato: 
``json
    {
        "mode": "test",
        "text": Texto recibido y limpio,
        "embedding": arreglo de 128 números de punto flotante.
    }



#### En la ruta /health y en /
Sirve para comprobar que el servicio esta trabajando.


### Modelo de Embeddings

Se utiliza el modelo **paraphrase-multilingual-MiniLM-L12-v2*, con él se generan vectores de 384 dimensiones.

Para evitar tener un esquema tan grande, previendo una indexación de cientos de millones de embeddings, se han reducido a 128 dimensiones, usando una matriz PCA de reducción, previamente entrenada.

### Limpieza de texto
En este orde:
- Se quita todo .el contenido delimitado por '[[RS-'  '-RS]]'.
- Se quitan todos los tokens : '***' | (triple asterisco).
- Se quitan todos los tokens : '[[¡'.
- Se reemplazan todos los tokens : '!]]' por '. '.
- Se reemplazan todos los tokens : ''''!]]' por '. '.
- Se reemplazan todos los tokens : '```html<br /><br />' por ''''\n'.
- Se reemplazan todos los tokens : '```html<br/><br/>' por ''''\n'.
- Se reemplazan todos los tokens : '```html<br/>' por ''''\n'.
- Se reemplazan todos los tokens : '```html<br />' por ''''\n'.
- Se eliminan los TAGs de html
- Se quitan todos los tokens : '[[03]]'.
- Se quitan todos los tokens : '[[05]]'.
- Se reemplazan todos los tokens : ''''\r\n\r\n' por ''''\r\n'.

### Separación de texto largo en Chunks
Una vez limpio el texto, y aplicando el modelo de procesamiento de lenguaje natural **es_core_news_md** con Spacy.
Y así analizando el texto de manera avanzada, se separa en oraciones.

Se obtienen embeddings para cada oración, y usando operaciones de similitud entre esas oraciones consecutivas, se determinan si varias oraciones se pueden agrupar a un mismo chunk o se manejan en chunks separados. 

También se aplica la regla de que cada chunk tenga a al menos 150 caracteres. Con esto, oraciones muy cortas, se agrupan de manera completa en otras.

# Instalación

Usar **Dockerfile** para creación y puesta en marcha del contenedor, 

O bien, si quieres tener la aplicación funcionando en tu entorno de  desarrollo
 - 1.- Clona el repositorio
 - 2.- En tu terminal, cámbiate al folder creado
 - 3.- Crea el ambiente virtual con python. Usar : python -m venv venv
 - 4.- Actívalo
         * En Windows : venv\Scripts\activate
         * En Linux   : source venv/bin/activate
 - 5.- Instala las dependencias: pip install -r requirements.txt
 - 6.- Ejecuta la aplicación : python app.py
  
 Nota: Si vas a crear la imagen de Docker, no crees el ambiente virtual. Sólo agregarías espacio innecesario a la imagen (incluiría todo el directorio venv-)

 ## Prueba básica de creación de Embedding
  - Usa: http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE=
  - Con esto haces la prueba, enviando un texto corto en Base64.
  - Verás el Embedding de resultado, con el texto ya limpio sin codificar.
  * Es Base64, porque el texto puede contener caracteres que pueden chocar con el esquema de una URL correcta.
  * Es posible que textos muy largos en Base64, sobrepasen el límite para GET.