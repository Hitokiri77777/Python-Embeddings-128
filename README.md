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
- Se eliminan el resto de TAGs de html.
- Se quitan todos los tokens : '***[[03]]***'.
- Se quitan todos los tokens : '***[[05]]***'.
- Normaliza diferentes tipos de comillas dobles y simples a sólo tener las simples básicas.
- Se reemplazan todos los tokens : '```\r\n\r\n```' por '```\r\n```'.


### Lógica de separación de texto largo en varios Chunks
Una vez limpio el texto y aplicando el modelo de procesamiento de lenguaje natural **es_core_news_md** con Spacy; se separa en oraciones y aplicando  ***Similaridad de Coseno*** con un umbral de **0.6** de similitud, se agrupan oraciones consecutivas en un mismo ***chunk***.

En detalle, se obtienen embeddings para cada oración, y usando operaciones de similitud entre esas oraciones consecutivas, se determinan si varias oraciones se pueden agrupar a un mismo chunk o se manejan en chunks separados. 

También se aplica la regla de que cada chunk tenga a al menos 150 caracteres. Con esto, oraciones muy cortas se agrupan en otras.

---

## Instalación para Desarrollo
Si quieres tener la aplicación funcionando en tu entorno de desarrollo:
 1. Clona el repositorio : ```git clone https://github.com/Hitokiri77777/Python-Embeddings-128.git```
 2. En tu terminal, cámbiate al folder creado.
 3. Crea el ambiente virtual con python. Usar : ``` python -m venv venv ```
 4. Activa el ambiente virtual:
    - En Windows : ``` venv\Scripts\activate ```
    - En Linux   : ``` source venv/bin/activate ```
 5. Instala las dependencias: ``` pip install -r requirements.txt ``` 
 6. Ejecuta la aplicación : ``` python app.py ```

 * Podría usarse un servidor como "waitress" de Python, para mejorar respuestas de la aplicación flask
   - Instala el paquete ```pip install waitress``` con el ambiente activado
   - Sirve la aplicación de manera más eficiente con el modelo multihilo de waitress con esto:

       ```waitress-serve --host=0.0.0.0 --port=5000 app:app```

- Ver prueba básica de creación de embedding en la última sección del documento.
- Con la demostración exitosa de la prueba; puedes ya hacer *POST* a la ruta **/process** para trabajar. 

---

## Creación de Imagen para ***Docker***
Usar el archivo **Dockerfile** en la raíz del proyecto, para creación y puesta en marcha del contenedor. 

1. Teniendo Docker instalado. Hacer : ```docker build -t python_embeddings .```
   para crear la imagen.
2. Obten el ID de la imagen creada, listando las imagenes existentes con : ```docker images```
3. Córrela, suponiendo que el ID es 562469e4e257 : ```docker run -p 5000:5000 562469e4e257```
   Con esto, el puerto 5000 de la imagen, se mapeará al también 5000 de tu máquina.
4. Ahora si podrías hacer la prueba: ```http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE=```

Se crearía una imagen de ***2.1GB*** ya funcional.

***Nota Importante:*** Recuerda que la imagen de la aplicación Flask, debe cargar el modelo *"paraphrase-multilingual-MiniLM-L12-v2"* desde internet en cada arranque. Ver el Log, para revisar cuando ya esta lista para trabajar.

Esto se puede evitar, descargando dicho modelo a un directorio, y hacer que forme parte de la imagen. Y modificar también el código fuente para cargarlo desde disco. Decidí dejarlo de esa forma, para no tener que agregar ese directorio en el repositorio GIT. Son unos 480 MB.


### Sugerencia para llevarlo a Producción
Puedes poner esta aplicación detrás de un servidor ***Nginx*** para mejorar rendimiento, seguridad y escalabilidad.

Para hacerlo, puedes usar el archivo ***docker-compose.yml***. Esto crearía el flujo siguiente:
1. El usuario accede a http://localhost/.
2. Nginx recibe la solicitud en el puerto 80.
3. Nginx reenvía la solicitud al contenedor de Flask en backend:5000.
4. Flask procesa la petición y devuelve la respuesta a Nginx.
5. Nginx envía la respuesta final al usuario.

Utilizarías el comando ```docker-compose up --build -d```
- Esto crearía las 2 imágenes, una para la App de Flask, nuestra app. Y otra con un linux Alpine súper ligero, con un servidor de Nginx. Y un "puente de red" para hacer que las dos imágenes trabajen en conjunto.

Tener una aplicación Flask detrás de un servidor Nginx es una práctica recomendada en entornos de producción porque mejora la seguridad, el rendimiento y la escalabilidad.

### Problemas de ejecutar Flask directamente
Si se ejecuta Flask de forma nativa con python app.py nos enfrentaríamos a varias limitaciones:

- Flask no es eficiente manejando múltiples conexiones → Puede procesar sólo un número limitado de solicitudes simultáneamente.
- No maneja archivos estáticos eficientemente → Sirviendo imágenes, CSS o JS desde Flask, el rendimiento será pobre (No es el caso de nuestra aplicación).
- No soporta balanceo de carga → No puedes escalar a múltiples instancias fácilmente.
- No maneja HTTPS nativamente → No puedes configurar certificados SSL directamente en Flask.

### ¿Por qué usar Nginx como proxy reverso?
Nginx es un servidor web ligero y eficiente que actúa como intermediario entre los clientes (navegador, API, etc.) y la aplicación Flask.

- Mejora el rendimiento → Puede manejar miles de conexiones concurrentes.
- Maneja archivos estáticos → Como imágenes, CSS y JS sin cargar Flask innecesariamente.
- Balanceo de carga → Si usas múltiples instancias de Flask, distribuye las peticiones.
- Soporte SSL/TLS → Se encarga de manejar certificados HTTPS.
- Protege contra ataques → Como DDoS y accesos no autorizados.

### Beneficios en producción
- Optimización de tráfico → Nginx maneja archivos estáticos, evitando que Flask se sobrecargue.
- Seguridad → Evita que los clientes accedan directamente a Flask.
- Escalabilidad → Puedes agregar múltiples instancias de Flask detrás de Nginx.
- HTTPS fácil → Puedes configurar un certificado SSL en Nginx sin tocar Flask. 

### Si al final solo hay una aplicación Flask ejecutándose, ¿por qué Nginx mejora el rendimiento? 
La clave está en cómo se manejan las conexiones y la distribución de carga.

Flask maneja mal muchas conexiones concurrentes
Flask, por sí solo, no está diseñado para manejar muchas conexiones al mismo tiempo, porque usa un servidor de desarrollo interno.

Nginx actúa como un buffer inteligente entre los clientes y Flask, mejorando el rendimiento por varios motivos:

1. Nginx maneja muchas conexiones simultáneas eficientemente
Nginx usa un modelo asíncrono basado en eventos, en lugar de crear un nuevo proceso/hilo para cada solicitud. Esto le permite gestionar miles de conexiones sin consumir muchos recursos.

📌 Ejemplo práctico:

Sin Nginx: Flask recibe 1000 peticiones y solo puede atender 5-10 a la vez. Las demás quedan bloqueadas.

Con Nginx: Nginx recibe 1000 peticiones y distribuye las solicitudes a Flask de manera controlada. Mientras Flask procesa una, Nginx retiene las demás sin bloquearlas.

2. Nginx maneja archivos estáticos sin pasar por Flask
Si sirves archivos como imágenes, CSS o JS con Flask, cada solicitud consume recursos de Flask innecesariamente.
🚀 Con Nginx, esos archivos se sirven directamente sin afectar el rendimiento de Flask.

📌 Ejemplo práctico:

Sin Nginx: Flask recibe 1000 solicitudes, incluyendo archivos estáticos. Se satura.

Con Nginx: Nginx atiende las solicitudes de archivos estáticos y solo pasa las solicitudes de API a Flask.

3. Nginx hace balanceo de carga (opcional)
Si tu aplicación Flask crece, puedes correr múltiples instancias de Flask y usar Nginx para distribuir la carga.

📌 Ejemplo práctico:

Sin Nginx: Una sola instancia de Flask se satura con 1000 usuarios.

Con Nginx + 3 Flask: Nginx distribuye las solicitudes entre 3 instancias, mejorando la escalabilidad.

4. Nginx mantiene conexiones abiertas y usa caché
Cuando un cliente accede a Flask directamente, cada nueva conexión se abre y cierra.
📌 Con Nginx, las conexiones pueden mantenerse abiertas y reutilizarse, reduciendo la latencia.

🔥 Conclusión:
Nginx no hace que Flask procese más rápido, pero evita que Flask se sature y distribuye mejor las solicitudes. Por eso es fundamental para producción. 

***NOTA:*** Si vas a crear la imagen de Docker, que no sea desde el folder usado para una instalación de desarrollo. Sólo agregarías espacio innecesario a la imagen (incluiría todo el subdirectorio /venv/).

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
  - Con esto envías un texto corto en *Base64* sin usar *POST*.
  - Verás el ***Embedding*** de resultado, con el texto recibido ya limpio.
  - Se usa *Base64*, porque el texto puede contener caracteres que pueden chocar con el esquema de una URL correcta o de un JSON bien formado. Aplica igual para el caso de los datos JSON recibidos en los *POST*.
  - Es posible que textos muy largos en *Base64*, sobrepasen el límite para *GET* de ***/test***, por si vas a probar textos propios en sea ruta **/test**.
