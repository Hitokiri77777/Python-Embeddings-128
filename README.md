# Creador de Embeddings local
Crea embeddings usando el modelo de lenguale natural **paraphrase-multilingual-MiniLM-L12-v2** de manera local (no lo carga desde web).

Se accesa desde un WebService REST
#### En la ruta /process
Recibe un POST, esperando 2 parámetros
- mode : Que puede ser (single | chunks)
- textbase64 : Sería el texto en Base64


Cuando el **mode** es single, el texto pasado en Base64, se limpia y se crea su embedding.
Y se devuelve la respuesta:
    {
        "mode": "single",
        "text": Texto recibido y limpio,
        "embedding": arreglo de 128 números de punto flotante.
    }

Cuando el **mode** es chunks, el texto pasado en Base64, se limpia, se separa en chunks y a cada chunk se le crea su embedding. Y se devuelve la respuesta:
    {
        "mode": "chunks",
        "chunks": Arreglo chunks
    }

#### En la ruta /test
Se recibe en un GET, se recibe en el parámetro **base64text**, un texto en Base64.
Retornando una salida con el siguiente formato: 
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
- Se reemplazan todos los tokens : '!]]' por '. '.
- Se reemplazan todos los tokens : '<br /><br />' por '\n'.
- Se reemplazan todos los tokens : '<br/><br/>' por '\n'.
- Se reemplazan todos los tokens : '<br/>' por '\n'.
- Se reemplazan todos los tokens : '<br />' por '\n'.
- Se eliminan los TAGs de html
- Se quitan todos los tokens : '[[03]]'.
- Se quitan todos los tokens : '[[05]]'.
- Se reemplazan todos los tokens : '\r\n\r\n' por '\r\n'.

### Separación de texto largo en Chunks
Una vez limpio el texto, y aplicando el modelo de procesamiento de lenguaje natural **es_core_news_md** con Spacy.
Y así analizando el texto de manera avanzada, se separa en oraciones.

Se obtienen embeddings para cada oración, y usando operaciones de similitud entre esas oraciones consecutivas, se determinan si varias oraciones se pueden agrupar a un mismo chunk o se manejan en chunks separados. 

También se aplica la regla de que cada chunk tenga a al menos 150 caracteres. Con esto, oraciones muy cortas, se agrupan de manera completa en otras.