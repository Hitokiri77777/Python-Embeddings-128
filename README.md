# Servicio NLP Local para Búsqueda Semántica y RAG

**Embeddings 128-D + Keywords + Entidades NER + Chunking Semántico**, expuesto como
WebService REST con Flask y desplegable con Docker (Flask + Waitress + NGINX).

Servicio de procesamiento de lenguaje natural (NLP) que corre **completamente local**
(sin llamadas a APIs externas de terceros) y que, a partir de un texto recibido en
*Base64*, devuelve una o varias de las siguientes salidas, según el endpoint o modo
invocado:

- **Embeddings** reducidos de 384 a 128 dimensiones vía PCA.
- **Keywords** del documento (hasta 12 por defecto, configurable hasta 20).
- **Entidades NER** (Named Entity Recognition) con patrones personalizados.
- **Chunking semántico** del texto, con embedding por cada chunk.
- Opcionalmente, el mismo tratamiento para **título** y **resumen** por separado.

Está pensado para alimentar índices vectoriales, motores de búsqueda semántica,
pipelines de RAG (Retrieval-Augmented Generation) y clasificación de documentos.

---

## Arquitectura

El proyecto puede ejecutarse de dos maneras:

### 1) Desarrollo local (Flask directo, puerto 5000)

Cliente → Flask (app.py) → puerto 5000

### 2) Producción recomendada (con NGINX como proxy inverso)

Cliente → NGINX :80 → Flask/Waitress :5000 (solo red interna mynetwork)

En el modo de producción:

1. El usuario accede a http://localhost/.
2. NGINX recibe la solicitud en el puerto 80.
3. NGINX reenvía la solicitud al contenedor de Flask en backend:5000.
4. Flask procesa la petición y devuelve la respuesta a NGINX.
5. NGINX envía la respuesta final al usuario.

Esto se logra con el archivo docker-compose.yml incluido en el repositorio, que
define dos servicios (flask_app y nginx_proxy) conectados por una red bridge
aislada (mynetwork). El contenedor de Flask **no publica** el puerto 5000 al
exterior (solo expose), de modo que únicamente es accesible a través de NGINX.

---

## Endpoints del WebService

Todos los endpoints que reciben texto lo hacen en **Base64**, para evitar
problemas con caracteres especiales que romperían URLs o JSON mal formados.

| Método | Ruta | Propósito |
|---|---|---|
| POST | /process | Procesamiento principal. Acepta mode ∈ {single, chunks, extended, full}. |
| POST | /keywords | Solo cálculo de keywords del texto recibido. |
| POST | /entities | Solo cálculo de entidades NER del texto recibido. |
| GET  | /test | Prueba rápida con base64text en query string. |
| GET  | /health y / | Healthcheck del servicio. |

---

## POST /process — Contrato actual

Parámetros comunes:

- mode (string): single | chunks | extended | full.
- textbase64 (string): texto del documento en Base64.

Parámetros opcionales (solo mode=full):

- titlebase64 (string): título del documento en Base64.
- summarybase64 (string): resumen del documento en Base64.

Tabla consolidada del contrato actual por modo:

| Modo | Devuelve |
|---|---|
| single | mode, text (limpio), embedding (128 floats). |
| chunks | mode, chunks, embedding (arreglo de embeddings por chunk). Keywords y entities por chunk también se incluyen en la respuesta. |
| extended | Todo lo de chunks, más keywordsFullDocument, entitiesFullDocument, embeddingFullDocument. |
| full | Todo lo de extended, más titleEmbedding, titleKeywords, titleEntities, summaryEmbedding, summaryKeywords, summaryEntities (cuando se envían titlebase64 y/o summarybase64). |

### Ejemplo de request
```json
{
    "mode": "single",
    "textbase64": "SG9sYSBNdW5kbyE="
}
```

### Ejemplo de response (mode = single)
```json
{
    "mode": "single",
    "text": "Texto recibido y limpio",
    "embedding": [0.0123, -0.0456, "...", 128 valores]
}
```

### Ejemplo de response (mode = chunks)
```json
{
    "mode": "chunks",
    "chunks": ["chunk 1 ...", "chunk 2 ...", "..."],
    "embedding": [[...128], [...128], "..."]
}
```
### Ejemplo de response (mode = extended)

Además de lo anterior:
```json
{
    "keywordsFullDocument": ["...", "..."],
    "entitiesFullDocument": [["texto", "LABEL"], ["...", "..."]],
    "embeddingFullDocument": [...128]
}
```

### Ejemplo de response (mode = full)

Además de lo anterior, si se enviaron titlebase64 / summarybase64:
```json
{
    "titleEmbedding": [...128],
    "titleKeywords": ["...", "..."],
    "titleEntities": [["texto", "LABEL"], "..."],
    "summaryEmbedding": [...128],
    "summaryKeywords": ["...", "..."],
    "summaryEntities": [["texto", "LABEL"], "..."]
}
```

> **Nota sobre costo computacional:** los modos extended y full son
> significativamente más demandantes en CPU y tiempo que single y chunks.
> Elige el modo que mejor se adapte a las necesidades de cada índice.

---

## POST /keywords — Solo keywords

Recibe un POST con:

- textbase64 (string): texto a procesar en Base64.
- quantity (entero positivo, opcional): cuántas palabras devolver.
  - Default: **12** si no se indica o si el valor es inválido.
  - Máximo: **20**.
  - El número resultante puede ser menor tras la lematización.

Ejemplo de request:
```json
{
    "textbase64": "SG9sYSBNdW5kbyE=",
    "quantity": 10
}
```

Detalles internos:

- Se usa **KeyBERT** con el mismo modelo *"paraphrase-multilingual-MiniLM-L12-v2"*.
- Se aplican **StopWords** en inglés y español de **NLTK**.
- Se lematizan los resultados con **spaCy** usando el modelo es_core_news_md,
  devolviendo la raíz de la palabra y evitando repetir palabras parecidas.
- Desde la actualización del **23 Septiembre 2025**, se intenta el método **MMR**
  (Maximal Marginal Relevance) con factor discreto **0.6** (balance entre
  diversidad y relevancia). Si MMR falla, se reintenta sin MMR.
- En ambos casos se buscan **palabras individuales**, no frases.

Prueba rápida por URL:

```http://127.0.0.1:5000/test?base64text=RWwgY2FtYmlvIGNsaW3DoXRpY28gZXN0w6EgcHJvdm9jYW5kbyB1biBhdW1lbnRvIGRlbCBuaXZlbCBkZWwgbWFyIHkgZXZlbnRvcyBjbGltw6F0aWNvcyBleHRyZW1vcyBlbiBtdWNoYXMgcGFydGVzIGRlbCBtdW5kby4=```

---

## POST /entities — Solo entidades NER

Recibe un POST con:

- textbase64 (string): texto a procesar en Base64.

Devuelve un arreglo de tuplas (text, label) con las entidades detectadas en el
documento, sin repetidos y excluyendo etiquetas no deseadas.

Detalles internos:

- Se usa **spaCy** con el modelo es_core_news_md.
- Se evita devolver entidades repetidas.
- Se **excluyen** las entidades de etiqueta MISC (comportamiento original;
  ver actualización del 23 Septiembre 2025 para excepciones).
- Desde el **23 Septiembre 2025**, se añadieron **patrones personalizados de NER**
  para entidades no cubiertas por el modelo, y ya se toman también las entidades
  misceláneas que antes se evitaban. Categorías cubiertas por patrones:
  - Tratados comerciales.
  - Organismos y dependencias.
  - Partidos políticos.
  - Programas y políticas públicas.
  - Empresas e instituciones relevantes en México.
  - Medios de comunicación.
  - Figuras políticas muy frecuentes.
  - Tecnologías y conceptos.
  - Inteligencia Artificial.

Ejemplo de request:
```json
{
    "textbase64": "R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu"
}
```

Prueba rápida por URL:

```http://127.0.0.1:5000/test?base64text=R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu```

---

## GET /test — Prueba rápida

Recibe en el parámetro base64text un texto en Base64 y devuelve la misma salida
que mode=single de /process:
```json
{
    "mode": "test",
    "text": "Texto recibido y limpio",
    "embedding": [0.0123, -0.0456, "...", 128 valores]
}
```

Ejemplo de uso:

```http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE=```

> **Advertencia:** textos muy largos en Base64 pueden sobrepasar el límite de
> longitud para URLs en un GET. Para textos largos, usa POST /process.

---

## GET /health y GET / — Healthcheck

Sirven para comprobar que el servicio está en ejecución y respondiendo.

---

## Qué puedes construir con este servicio

Gracias a que cada documento puede devolver embeddings, keywords y entidades —
y opcionalmente un tratamiento separado para título y resumen — puedes:

- **Búsqueda semántica** clásica sobre documentos o fragmentos (chunks).
- **Búsqueda por keywords** o por entidades específicas (p. ej. "documentos que
  mencionan al partido X").
- **Clasificación de documentos** a partir de sus entidades / keywords.
- **Búsqueda semántica restringida a título y/o resumen** (con mode=full).
- **Comparación de documentos completos** por similitud de sus embeddings
  agregados (embeddingFullDocument).
- **Índices vectoriales complejos** con campos anidados (un vector por chunk, más
  metadatos de keywords y entidades).
- **Pipelines de RAG**: enriquecer cada documento con keywords y entidades que se
  envían al LLM como contexto adicional, además del chunk recuperado.

---

## Modelo de Embeddings

Se utiliza el modelo **paraphrase-multilingual-MiniLM-L12-v2**, de tamaño medio
y funcional para texto en varios idiomas (inglés y español incluidos). Con este
modelo se generan vectores de **384 dimensiones**.

Para evitar tener un esquema tan grande y previendo una indexación de millones de
embeddings, se han reducido a **128 dimensiones** usando una **matriz PCA de
reducción** previamente entrenada, incluida en el proyecto.

---

## Limpieza de texto recibido

La limpieza del texto antes de trabajarlo es indispensable para que el modelo de
procesamiento de lenguaje natural es_core_news_md cumpla adecuadamente su
función al separar en oraciones.

En este orden:

1. Se quita todo el contenido delimitado por [[RS- … -RS]].
2. Se quitan todos los tokens *** (triple asterisco).
3. Se quitan todos los tokens [[¡.
4. Se reemplazan todos los tokens !]] por . .
5. Se reemplazan todos los tokens !]] por . . (regla conservada tal cual estaba en la versión original).
6. Se reemplazan todos los tokens <br /><br /> por \n.
7. Se reemplazan todos los tokens <br/><br/> por \n.
8. Se reemplazan todos los tokens <br/> por \n.
9. Se reemplazan todos los tokens <br /> por \n.
10. Se reemplazan todos los tokens  ——————————  por \n.
11. Se reemplazan todos los tokens  ___  por \n.
12. Se eliminan el resto de TAGs de HTML.
13. Se quitan todos los tokens [[03]].
14. Se quitan todos los tokens [[05]].
15. Se normalizan diferentes tipos de comillas dobles y simples a solo las simples básicas.
16. Se reemplazan todos los tokens \r\n\r\n por \r\n.

---

## Separación de texto largo en Chunks (chunking semántico)

Una vez limpio el texto y aplicando el modelo es_core_news_md con **spaCy**, se
separa en oraciones. Luego, aplicando **similitud de coseno** con un umbral de
**0.6** de similitud, se agrupan oraciones consecutivas en un mismo chunk.

En resumen:

1. Se obtienen embeddings para cada oración.
2. Se comparan oraciones consecutivas por similitud.
3. Se agrupan en el mismo chunk si superan el umbral.
4. Se aplica la regla de que cada chunk tenga al menos **150 caracteres**. Con
   esto, oraciones muy cortas se agrupan con otras.

---

## Lematización de Keywords (detalle)

Desde la actualización del **23 Septiembre 2025**, la lematización de keywords
aplica varias capas de corrección en español:

- Patrones de corrección en español.
- Corrección de verbos mal lematizados terminados en -ir y en -ar.
- Lista de correcciones a palabras puntuales.
- Lista de anglicismos.
- Lista de palabras con acentos.
- Lista de nombres propios y errores comunes.

---

## Historial funcional (changelog integrado)

Esta sección conserva, con sus fechas, las capacidades que se fueron añadiendo al
servicio. Se mantiene íntegra como referencia de evolución y de decisiones de
diseño.

### Actualización 16 Mayo 2025
- Se agrega el cálculo de **Keywords** al procesar embeddings; será parte de
  la respuesta. Se devolverá un arreglo de hasta 12 keywords por todo el
  documento, y máximo 5 por cada chunk generado.
- Se agrega la ruta /keywords al WebService, para el caso donde solo se
  necesite este cálculo. Recibe un POST con textbase64 y quantity.
- Se usa **KeyBERT** con el modelo *"paraphrase-multilingual-MiniLM-L12-v2"*.
- Se aplican **StopWords** en inglés y español de **NLTK**.
- Se lematizan los resultados con **spaCy** y es_core_news_md.

### Actualización 19 Mayo 2025
- Se agrega el cálculo de **Entities** al procesar embeddings. Véase NER
  (Named Entity Recognition): componente de PLN que identifica categorías
  predefinidas de objetos en un texto. Categorías pueden incluir, entre otras:
  nombres de personas, organizaciones, ubicaciones, expresiones de tiempo,
  cantidades, códigos médicos, valores monetarios y porcentajes.
- Se retorna un arreglo de tuplas (text, label) por todo el documento, y un
  arreglo también por cada chunk.
- Se agrega la ruta /entities al WebService. Recibe POST con textbase64.
- Se usa **spaCy** con es_core_news_md.
- Se evita devolver entidades repetidas.
- Se evitan las entidades de etiqueta MISC.

### Actualización 20 Mayo 2025
- Se agregan 2 modos a /process:
  - **extended**: igual que chunks, más keywordsFullDocument,
    entitiesFullDocument y embeddingFullDocument.
  - **full**: igual que extended, más el tratamiento de titlebase64 y
    summarybase64 (opcionales) con sus 6 campos extra: titleEmbedding,
    titleKeywords, titleEntities, summaryEmbedding, summaryKeywords,
    summaryEntities.
- Estos modos permiten generar índices vectoriales con metadatos detallados o
  índices con esquemas de campos anidados.
- Posibilita búsquedas por keywords, búsquedas por entidades, clasificación de
  documentos, búsquedas semánticas solo por título y/o resumen, comparación de
  documentos completos similares, etc. Además, entrega información extra útil
  para enviarse a un LLM en caso de estar construyendo un RAG.
- Se reconoce que estos modos pueden ser muy demandantes en recursos y tiempo;
  por eso se ofrece la opción de elegir el modo que mejor se adapte a cada índice.

### Actualización 23 Septiembre 2025
- **Entidades NER:** se agregan patrones personalizados de entidades no
  incluidas en es_core_news_md, y ya se toman las entidades misceláneas que
  antes se evitaban. Categorías cubiertas:
  - Tratados comerciales.
  - Organismos y dependencias.
  - Partidos políticos.
  - Programas y políticas públicas.
  - Empresas e instituciones relevantes en México.
  - Medios de comunicación.
  - Figuras políticas muy frecuentes.
  - Tecnologías y conceptos.
  - Inteligencia Artificial.
- **Keywords:** se mejora la lógica con método **MMR** basado en factor discreto
  (0.6, entre diversidad y relevancia). Si MMR falla, se reintenta sin MMR.
  Siempre buscando palabras individuales, no frases.
- **Lematización de Keywords:** se aplican patrones de corrección en español,
  corrección de verbos mal lematizados terminados en -ir y -ar, lista de
  correcciones puntuales, lista de anglicismos, lista de palabras con acentos y
  lista de nombres propios y errores comunes.

---

## Alternativas de uso

- **En máquina local:** con un ambiente Python completo. Sirve para desarrollo.
- **En producción** hay varias opciones:
  1. Ambiente Python sirviendo la aplicación con **Waitress**.
  2. Ejecutable (EXE): difícil de mantener, no tan recomendable; se vuelve un
     ejecutable enorme y difícil de modificar.
  3. Administrar como **Servicio de Sistema** con **NSSM** (requiere su ambiente
     Python y también se sirve con Waitress; NSSM lo registra como servicio).
  4. **Imagen Docker (Docker + Flask + Waitress + NGINX)**: opción ideal. Limpio
     y portable. Fácil de mover a Linux, a la nube, o de escalar con Docker
     Compose / Kubernetes.
  5. **Imagen Docker (Docker + Flask + Waitress)**: opción ideal si se instala en
     una infraestructura que ya cuenta con "Balanceo de Carga" y "Proxy Inverso"
     (servidores IIS). En estos casos, NGINX ya no es necesario.

---

## Instalación para Desarrollo con su ambiente Python

1. Clona el repositorio:
   ``` git clone https://github.com/Hitokiri77777/Python-Embeddings-128.git ```
2. En tu terminal, cámbiate al folder creado.
3. Crea el ambiente virtual con Python (probado en **Python 3.11.9**):
   ```python -m venv venv```
4. Activa el ambiente virtual:
   - Windows: ``` venv\Scripts\activate ```
   - Linux:   ``` source venv/bin/activate ```
5. Instala las dependencias:
   ``` pip install -r requirements.txt ```
6. Ejecuta la aplicación:
   ``` python app.py ```

Podrías usar **waitress** para mejorar las respuestas de Flask:

- Instala: pip install waitress (con el ambiente activado).
- Sirve con:

  ``` waitress-serve --host=0.0.0.0 --port=5000 app:app ```

  ó

  ``` python -m waitress --host=0.0.0.0 --port=5000 app:app``` 

- Ver *"Prueba básica de creación de Embedding"* al final del documento.
- Con la prueba exitosa, ya puedes hacer POST a /process para trabajar.

---

## Creación de Imagen para Docker (solo Flask)

Usa el archivo **Dockerfile** en la raíz del proyecto.

1. Teniendo Docker instalado:
   docker build -t python_embeddings .
2. Pruébalo:
   docker run -it --rm -p 5000:5000 python_embeddings
   Esto mapea el puerto 5000 de la imagen al 5000 de tu máquina, y verás los
   mensajes en consola.
3. Prueba: ``` http://127.0.0.1:5000/test?base64text=SG9sYSBNdW5kbyE= ```

**10 Abril 2024** — Se actualizó el Dockerfile con varios aspectos:

- Imagen más estable: incluye variables de entorno y ya no se ejecuta como root.
- Lo más reducida posible: ya no se incluyen compiladores, descargables ni
  temporales, y se reducen capas de creación de Docker.
- Incluye el modelo *"paraphrase-multilingual-MiniLM-L12-v2"* cacheado dentro de
  la imagen.
- Se comprobó que la imagen es funcional y sin errores.
- Tamaño final de la imagen: **2.16 GB**.

---

## Producción con NGINX (arquitectura de proxy inverso)

Puedes poner esta aplicación detrás de **NGINX** para mejorar rendimiento,
seguridad y escalabilidad. Se usa el archivo **docker-compose.yml** del
repositorio, que crea el flujo descrito al inicio:

1. El usuario accede a http://localhost/.
2. NGINX recibe la solicitud en el puerto 80.
3. NGINX reenvía la solicitud al contenedor de Flask en backend:5000.
4. Flask procesa la petición y devuelve la respuesta a NGINX.
5. NGINX envía la respuesta final al usuario.

Comando:
``` docker-compose up --build -d ```

Esto crea las 2 imágenes (Flask + Alpine/NGINX), las conecta con un *puente de
red* y las deja trabajando en conjunto.

Tener una aplicación Flask detrás de NGINX es una práctica recomendada en
producción porque mejora seguridad, rendimiento y escalabilidad. NGINX actúa
como intermediario entre los clientes (navegador, API, etc.) y la aplicación
Flask.

### Beneficios en producción

- **Optimización de tráfico:** NGINX maneja archivos estáticos, evitando que
  Flask se sobrecargue.
- **Seguridad:** evita que los clientes accedan directamente a Flask.
- **Escalabilidad:** — dos patrones soportados:
1. *Escalamiento vertical de concurrencia:* varias instancias de Flask
detrás de un mismo NGINX (requiere un balanceador delante de NGINX o
upstream en NGINX).

2. *Escalamiento horizontal por nodos:* varias instancias del stack
completo (NGINX + Flask + Waitress) en servidores Linux distintos, con
partición estática de carga controlada por el cliente. Este es el
patrón usado actualmente en producción; ver "Despliegue actual en
producción" más abajo.
- **HTTPS fácil:** puedes configurar un certificado SSL en NGINX sin tocar
  Flask.

### Despliegue actual en producción (escalamiento horizontal por nodos)

El servicio se encuentra desplegado en **dos nodos independientes** sobre
servidores Linux. Cada nodo ejecuta su propio stack completo
(**NGINX + Flask/Waitress**) y tiene su propia IP de intranet. No hay un
balanceador dinámico delante de ambos nodos: la distribución de carga se
realiza de forma **manual, por configuración del cliente**.

#### Esquema general

```
Índice Semántico 1 ─┐
Índice Semántico 2 ─┴─→ IP A
Índice Semántico 3 ─┐
Índice Semántico 4 ─┴─→ IP B
Buscador Web ─────────→ IP A
RAG ──────────────────→ IP B
```

#### Distribución actual

- **4 índices semánticos 24/7** que consumen el servicio para generar
  embeddings, keywords, entidades y chunks que luego se indexan. 
- **2 índices** apuntan al **Nodo A** y **2 índices** al **Nodo B**
  (reparto fijo, definido en la configuración de cada índice).
- El **Buscador Web** consulta **sólo al Nodo A**.
- El **RAG** consulta **solo el Nodo B**.

#### Decisión de diseño: sin balanceador

Se evaluó la opción de colocar un balanceador delante de ambos nodos, a cargo
del Centro de Cómputo sobre la infraestructura existente. La implementación
no llegó a concretarse como una solución estable para el equipo, por lo que
se optó por **partición estática de carga en el cliente**.

Para el perfil de uso actual (4 índices semánticos, 1 buscador web y 1 RAG),
esta opción resulta suficiente y predecible.

> **Nota de contexto:** el ecosistema de la organización es mayoritariamente
> Windows, mientras que los nodos del servicio NLP operan sobre Linux.
> Cualquier solución de balanceo futura deberá considerar esta convivencia.

#### Ventajas del esquema actual

- **Sin dependencia de un balanceador:** si un nodo cae, solo se ven afectados los
  índices configurados contra ese nodo; se pueden reconfigurar al otro nodo
  de forma manual.
- **Latencia predecible:** cada cliente "sabe" a qué IP ir siempre; no hay
  salto de red adicional por balanceo.
- **Operación simple:** no requiere infraestructura adicional ni
  descubrimiento de servicios.

#### Consideraciones y evolución futura

- El reparto es **manual**; agregar/quitar nodos o rebalancear requiere
  reconfigurar los clientes afectados.
- Si en el futuro se requiere **balanceo dinámico**, bastará con colocar un
  balanceador (por ejemplo, uno basado en las herramientas de Windows Server
  disponibles en la infraestructura, o un servicio de nube) delante de ambos
  nodos y apuntar todos los clientes a esa única entrada.
- A mediano plazo existe interés en acercarse a un esquema auto-contenido
  desplegable en nube (Kubernetes, **EKS de AWS**, servicios gestionados, o pipelines desde
  el mismo GitHub), pero por costos no se ha iniciado. El diseño actual
  (stack completo por nodo, sin dependencias externas al nodo) es compatible
  con esa migración futura.

---

## Creación de Imágenes en Docker (Flask + NGINX)

Teniendo Docker instalado, cámbiate al directorio de la aplicación y ejecuta:

``` docker-compose up --build ```

Eso crea 2 imágenes en Docker dentro de una red.

Comprueba que las 2 imágenes están ejecutándose con:

``` docker ps ```

Comprueba que NGINX recibe las llamadas y las direcciona a Flask:

``` http://127.0.0.1/test?base64text=SG9sYSBNdW5kbyE= ```

A diferencia de las pruebas anteriores, aquí **no** se indica el puerto 5000.

---

## Desarrollo local con token de Hugging Face (opcional)

En producción el build se hace sin token: el modelo de
embeddings se descarga de forma anónima y el build se completa igual, solo que
más lento (Hugging Face aplica rate limit a usuarios no autenticados).

En desarrollo, si quieres acelerar la descarga del modelo, puedes usar un token
gratuito de Hugging Face.

### 1. Generar el token

1. Crea una cuenta (gratuita) en https://huggingface.co/join
2. Ve a https://huggingface.co/settings/tokens
3. Pulsa "New token", dale un nombre (ej. `docker-embeddings-build`) y
   selecciona el rol **Read** (solo lectura, es suficiente).
4. Copia el token (empieza con `hf_...`). **No lo compartas ni lo subas al repo.**

### 2. Guardarlo en disco (fuera del proyecto)

```bash
# Guardar el token en un archivo fuera del repositorio
echo "hf_tu_token_aqui" > ~/.hf_token

# Restringir permisos: solo tu usuario puede leerlo
chmod 600 ~/.hf_token
```

`chmod 600` significa: lectura y escritura solo para el propietario, nada para
grupo ni otros. Es importante porque el token es una credencial sensible.

### 3. Construir la imagen de Flask con el token

El token se pasa como **secreto de BuildKit** (no queda grabado en la imagen).
El Dockerfile lo monta solo durante el `RUN` de descarga del modelo:

```bash
# Construir Flask con token (rápido)
docker buildx build --secret id=HF_TOKEN,src=$HOME/.hf_token \
  -t python-embeddings-128-flask_app:latest --load .
```

Si cambiaste `requirements.txt` o el Dockerfile (y necesitas reconstruir todo):

```bash
docker buildx build --no-cache --secret id=HF_TOKEN,src=$HOME/.hf_token \
  -t python-embeddings-128-flask_app:latest --load .
```

> **Nota:** `--no-cache` solo es necesario cuando cambias `requirements.txt` o
> el Dockerfile. Para cambios solo en `app.py`, sin `--no-cache` es mucho más
> rápido (reutiliza las capas cacheadas).

### 4. Construir NGINX (no necesita token) y levantar todo

```bash
# NGINX se construye en segundos, sin token
docker compose build nginx_proxy

# Levantar ambos contenedores
docker compose up -d
```

### 5. Alternativa: build sin token (como en producción)

Si no quieres usar token (o no tienes uno a mano), el build funciona igual,
solo más lento:

```bash
docker compose up --build
```

### Resumen de comandos

| Escenario | Comando |
|---|---|
| **Primera vez / cambios en requirements.txt o Dockerfile** | `docker buildx build --no-cache --secret id=HF_TOKEN,src=$HOME/.hf_token -t python-embeddings-128-flask_app:latest --load .` |
| **Solo cambios en app.py (rápido)** | `docker buildx build --secret id=HF_TOKEN,src=$HOME/.hf_token -t python-embeddings-128-flask_app:latest --load .` |
| **Construir NGINX** | `docker compose build nginx_proxy` |
| **Levantar todo** | `docker compose up -d` |
| **Sin token (como producción)** | `docker compose up --build` |

> **Seguridad:** el token **nunca** se escribe en el Dockerfile ni se commitea.
> Se pasa como `--secret` en el build, y BuildKit lo monta de forma efímera
> (desaparece al terminar el `RUN`). El archivo `~/.hf_token` está fuera del
> repositorio y con permisos `600`.

## Prueba básica de creación de Embedding

Usa:

``` http://127.0.0.1:5000/test?base64text=R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu ```

O, cuando ya hay un servidor NGINX al frente:

``` http://127.0.0.1/test?base64text=R29iaWVybm8gZGUgRXN0YWRvcyBVbmlkb3Mu ```

Verás el ***Embedding*** de resultado, con el texto recibido ya limpio, y
también los *Keywords* y *Entities* calculados.

Notas:

- Se usa **Base64** porque el texto puede contener caracteres que chocarían con
  el esquema de una URL correcta o con un JSON bien formado. Aplica igual para
  el caso de los datos JSON recibidos en los POST.
- Es posible que textos muy largos en Base64 sobrepasen el límite para GET de
  /test; en ese caso usa POST /process.
