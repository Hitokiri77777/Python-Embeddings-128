from   sentence_transformers    import SentenceTransformer
from   sklearn.metrics.pairwise import cosine_similarity
from   bs4                      import BeautifulSoup
from   keybert                  import KeyBERT
from   nltk.corpus              import stopwords
from   flask                    import jsonify
from   pathlib                  import Path
import faiss
import spacy
import re 
import os


class ChunksAndEmbeddings:
    def __init__(self):
        self.PCA_Matrix_FileName = 'DB_matrix.faiss'
        self.PCA_Matrix_Trained  = faiss.read_VectorTransform(self.PCA_Matrix_FileName)

    def Load_LanguageModel(self):
        # Lógica de selección automática
        if self._is_running_in_docker():
            # Cargar modelo de embeddings desde internet (Así debe usarse cuando se ejecuta desde Docker)
            model_path = "paraphrase-multilingual-MiniLM-L12-v2"
            environment = "DOCKER/PRODUCTION"
        elif self._is_local_model_available():
            # Cargar modelo de embeddings desde disco duro
            model_path = "../ModelosIA/paraphrase-multilingual-MiniLM-L12-v2"
            environment = "DEVELOPMENT (local model)"
        else:
            # Por si estamos en Desarrollo y no existe el modelo en disco duro
            model_path = "paraphrase-multilingual-MiniLM-L12-v2"
            environment = "DEVELOPMENT (remote model)"
        
        print(f"🔍 Entorno detectado: {environment}")
        print(f"📁 Cargando modelo desde: {model_path}")
        
        try:
            self.EmbeddigModel = SentenceTransformer(model_path)
            print("✅ Modelo cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            # Fallback: intentar con modelo remoto si falla el local
            if model_path != "paraphrase-multilingual-MiniLM-L12-v2":
                print("🔄 Intentando fallback con modelo remoto...")
                self.EmbeddigModel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            else:
                raise

        #Mismo modelo cargado, se usa para KeyBERT
        self.KeyBertModel  = KeyBERT(self.EmbeddigModel)
        # Cargar modelo de Spacy para español
        self.nlp = spacy.load("es_core_news_md")
        
        #Se agregan entidades propias, que el modelo "es_core_news_md" no tiene
        self._add_custom_entities()
        
        # Añadir reglas para abreviaturas
        abreviaturas = ["Sr.", "Sra.", "Dr.", "Dra.", "vs.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
        for abrev in abreviaturas:
            self.nlp.tokenizer.add_special_case(abrev, [{"ORTH": abrev}])
            
        #Se cargan los corpus de stopwords en inglés y español de nltk (son más de 500 entre ambas)
        # Para pruebas en ambientes de desarrollo. Debe existir folder \venv\nltk_data\corpora\stopwords
        # descargado desde https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip 
        # ó bien hacer antes nltk.download('stopwords')  Se requiere: import nltk
        english_sw = set(stopwords.words('english'))
        spanish_sw = set(stopwords.words('spanish'))

        # Unimos ambas listas en una
        self.StopwordsList = list(english_sw.union(spanish_sw))
        
        # Patrones regex sistemáticos para correcciones en lematización
        self._Lematizacion_load_correction_patterns()        
        
    def CleanText(self, Text):
        if Text == None:
            return ""
        pattern      = r'\[\[RS-.*?-RS\]\]'
        cleaned_text = re.sub(pattern, '', Text, flags=re.DOTALL)
        cleaned_text = cleaned_text.replace("***", "")
        cleaned_text = cleaned_text.replace("[[¡", "")
        cleaned_text = cleaned_text.replace("!]]", ". ")
        cleaned_text = cleaned_text.replace("<br /><br />", os.linesep)
        cleaned_text = cleaned_text.replace("<br/><br/>", os.linesep)
        cleaned_text = cleaned_text.replace("<br/>", os.linesep)
        cleaned_text = cleaned_text.replace("<br />", os.linesep)
        cleaned_text = cleaned_text.replace(" —————————— ", os.linesep)        
        cleaned_text = cleaned_text.replace(" ___ ", os.linesep)
        soup         = BeautifulSoup(cleaned_text, "html.parser")
        cleaned_text = soup.get_text(separator=' ', strip=True)
        cleaned_text = cleaned_text.replace("[[05]]", "")
        cleaned_text = cleaned_text.replace("[[03]]", "")
        # Normalizar comillas (reemplazar comillas tipográficas por rectas)
        cleaned_text = cleaned_text.replace('"', '"').replace('"', '"')  # Comillas dobles
        cleaned_text = cleaned_text.replace(''', "'").replace(''', "'")  # Comillas simples
        cleaned_text = cleaned_text.replace('"', "'")                    # Reemplazando comillas dobles por simples
        cleaned_text = cleaned_text.replace("\\r\\n", os.linesep)
        cleaned_text = cleaned_text.replace("\\", "")
        
        # Limpiar espacios extra y saltos de línea
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    
    def GetChunks(self, Text):
        ### Esta función y  GetChunksAndEntities()  comparten totalmente la misma lógica
        ###  excepto por lo que retornan y la obtención de entidades
        ### Si hay un cambio o corrección que no sea relativo a eso, debe hacerse en las 2 funciones
        
        # Limpiamos texto
        Text = self.CleanText(Text)
        # Procesar el texto con spaCy
        doc = self.nlp(Text)

        # Dividir el texto en oraciones
        sentences = [sent.text for sent in doc.sents]
        
        if not sentences:  # Si no hay oraciones después del procesamiento
            return [Text]  # Devolver el texto completo como un único chunk

        # Generar embeddings para cada oración
        sentence_embeddings = self.EmbeddigModel.encode(sentences)

        # Calcular similitud entre oraciones consecutivas
        chunks = []
        current_chunk = [sentences[0]]  # Primera oración, se asigna al primer chunk
        threshold = 0.60                # Umbral de similitud para agrupar oraciones

        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i + 1]

            # Calcular similitud coseno entre oraciones consecutivas
            similarity = cosine_similarity(
                [sentence_embeddings[i]],
                [sentence_embeddings[i + 1]]
            )[0][0]            

            # Si se parecen, se agrega al chunk que se está formando
            if similarity > threshold or len(next_sentence) < 150 or len(current_sentence) < 150:
                current_chunk.append(next_sentence)
            # Cuando ya no se parecen, el chunk que se formaba, se termina
            # Y se empieza a formar otro chunk
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk.append(next_sentence)

        # Al terminar el ciclo, si había un chunk formándose se agrega como último generado
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        #Limpia los brincos de linea iniciales de cada chunk
        for i in range(len(chunks) - 1):
            chunks[i] = re.sub(r'^[\\r\\n]+', '', chunks[i]).strip()

        return chunks

    def reduce_embeddings_With_PCA_matrix(self, embeddings):
        if embeddings.ndim == 1:
            # Si es un vector 1D, expandir a 2D
            embeddings = embeddings.reshape(1, -1)
        # Reducir dimensiones
        return self.PCA_Matrix_Trained.apply_py(embeddings)

    def GetSingleEmbedding(self, text):
        # Generar y reducir el embedding
        embedding = self.EmbeddigModel.encode([text])
        reduced_embedding = self.reduce_embeddings_With_PCA_matrix(embedding)
        
        return reduced_embedding
    
    def GetEmbeddings(self, Chunks):
        #Ahora siempre que retornamos los embeddings generados, los reducimos
        embeddings_384 = self.EmbeddigModel.encode(Chunks)
        
        reduced_embedding = self.reduce_embeddings_With_PCA_matrix(embeddings_384)
        
        return reduced_embedding

    def GetKeywords(self, Chunks):
        Keywords = []
            
        for chunk in Chunks:        
            extracted  = self.GetSingleTextKeywords(chunk, 5)
            Keywords.append(extracted)

        return Keywords
    
    def GetSingleTextKeywords(self, text, quantity=12):
        """
        Extrae keywords manteniendo el orden por score y asegurando cantidad mínima.
        Devuelve máximo 'quantity' keywords, ordenados por relevancia (score descendente).
        Incluye corrección de errores de lematización en español.
        """
        try:
            # Extraer keywords con scores - más de los necesarios para compensar
            initial_quantity = quantity * 3  # Extraer triple para compensar pérdidas
            
            extracted = self.KeyBertModel.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 1),  # Sólo palabras individuales (no frases)
                stop_words=self.StopwordsList,
                top_n=initial_quantity,
                use_mmr=True,           # Mejor que use_maxsum para consistencia
                diversity=0.6           # Balance entre relevancia y diversidad
            )
            
            # Si no se extrajo nada, intentar fallback sin MMR
            if not extracted:
                extracted = self.KeyBertModel.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 1), # Sólo palabras individuales (no frases)
                    stop_words=self.StopwordsList,
                    top_n=initial_quantity
                )
            
            # Filtrar por score mínimo y mantener orden original (por score descendente)
            min_score = 0.15  # Score mínimo aceptable
            filtered_keywords = []
            
            for kw, score in extracted:
                if score >= min_score:
                    #Se soporta dejar 't-mec'
                    if kw == 'mec' and 't-mec' in text.lower():
                        filtered_keywords.append(('t-mec', score))
                    else:
                        filtered_keywords.append((kw, score))
                    
            
            # Lematización inteligente con corrección de errores
            final_keywords = []
            seen_lemmas = set()
            
            for keyword, score in filtered_keywords:
                # Procesar con spaCy para lematización y luego corregir errores
                lemma = self._Lematizacion_correct_spanish(keyword)
                
                # Si la lematización devuelve None o string vacío, saltar
                if not lemma or len(lemma.strip()) < 3:
                    continue
                    
                # Evitar duplicados por lema y stopwords residuales
                if (lemma not in self.StopwordsList and
                    lemma not in seen_lemmas):
                    
                    seen_lemmas.add(lemma)
                    final_keywords.append((lemma, score))
                    
                    # Cortar si ya tenemos suficientes
                    if len(final_keywords) >= quantity * 2:  # Doble para selección final
                        break
            
            # Ordenar por score descendente y tomar los mejores
            final_keywords.sort(key=lambda x: x[1], reverse=True)
            best_keywords = [kw for kw, score in final_keywords[:quantity]]
            return best_keywords
            
        except Exception as e:
            # Logging adecuado en lugar de print (usando tu sistema de logging)
            # logger.error(f"Error extracting keywords: {e}")
            return []  # Fallback seguro sin romper la aplicación

    def _Lematizacion_correct_spanish(self, word: str) -> str:
        """Corrección optimizada de lematización preservando acentos"""
        if not word or len(word.strip()) < 2:
            return ""
        
        try:
            # Procesar con spaCy para lematización básica (preserva acentos)
            doc = self.nlp(word)
            if doc and len(doc) > 0:
                spacy_lemma = doc[0].lemma_.lower().strip()
            else:
                spacy_lemma = word.lower()
        except Exception:
            spacy_lemma = word.lower()
        
        # Aplicar correcciones sistemáticas
        corrected_lemma = self._Lematizacion_systematic_correction(spacy_lemma)
        
        # Validaciones finales
        if (len(corrected_lemma) < 3 or 
            not any(c.isalpha() for c in corrected_lemma)):
            return ""
            
        return corrected_lemma
    
    def _Lematizacion_systematic_correction(self, word: str) -> str:
        """Aplica correcciones sistemáticas basadas en patrones"""
        normalized_word = word.lower().strip()
        
        # Aplicar patrones de corrección
        for pattern, correction in self.Lematizacion_correction_patterns:
            match = pattern.search(normalized_word)
            if match:
                if callable(correction):
                    # Si la corrección es una función (como _correct_ir_verb)
                    result = correction(match)
                    if result != normalized_word:
                        return result
                else:
                    # Si la corrección es un string directo
                    return correction
        
        return normalized_word  # Devolver original si no hay corrección
    
    def _Lematizacion_correct_ir_verb(self, match) -> str:
        """Corrige verbos mal lematizados terminados en -ir"""
        base = match.group(1).lower()
        full_word = match.group(0).lower()
        
        # Verbos irregulares válidos que NO deben corregirse
        valid_ir_verbs = {
            'ir', 'venir', 'mentir', 'dormir', 'sentir', 'pedir', 
            'preferir', 'elegir', 'seguir', 'conseguir', 'vestir',
            'reír', 'sonreír', 'medir', 'servir', 'divertir'
        }
        
        if full_word in valid_ir_verbs:
            return full_word
            
        # Patrones de verbos que probablemente deberían ser -ar
        ar_verb_prefixes = {
            'habl', 'mencion', 'gener', 'gobern', 'analiz', 'utiliz',
            'implement', 'desarroll', 'proces', 'administr', 'gestion',
            'planific', 'organiz', 'coordinar', 'comunic', 'investig'
        }
        
        if any(base.startswith(prefix) for prefix in ar_verb_prefixes):
            return base + 'ar'
            
        return full_word  # Devolver original si no hay corrección clara

    def _Lematizacion_load_correction_patterns(self):
        # Patrones regex sistemáticos para correcciones en lematización
        
        self.Lematizacion_correction_patterns = [
            # Anglicismos con patrones regex
            (re.compile(r'report(e?s?)$',    re.IGNORECASE), 'reporte'),
            (re.compile(r'marketing$',       re.IGNORECASE), 'mercadeo'),
            (re.compile(r'management$',      re.IGNORECASE), 'gestión'),
            (re.compile(r'meeting(s?)$',     re.IGNORECASE), 'reunión'),
            (re.compile(r'briefing(s?)$',    re.IGNORECASE), 'informe'),
            (re.compile(r'training$',        re.IGNORECASE), 'capacitación'),
            (re.compile(r'ranking(s?)$',     re.IGNORECASE), 'clasificación'),
            (re.compile(r'data$',            re.IGNORECASE), 'datos'),
            (re.compile(r'cloud$',           re.IGNORECASE), 'nube'),
            (re.compile(r'device(s?)$',      re.IGNORECASE), 'dispositivo'),
            (re.compile(r'compan(y|ies)$',   re.IGNORECASE), 'empresa'),
            (re.compile(r'product(s?)$',     re.IGNORECASE), 'producto'),
            (re.compile(r'service(s?)$',     re.IGNORECASE), 'servicio'),
            
            # Corrección específica para "noticias" y plurales comunes
            (re.compile(r'^noticias$',       re.IGNORECASE), 'noticia'),
            
            # Verbos mal lematizados (patrón -ir → raíz + ar)
            (re.compile(r'(.+)(ir)$',        re.IGNORECASE), self._Lematizacion_correct_ir_verb),
            
            # Términos técnicos específicos
            (re.compile(r'^ai$',             re.IGNORECASE), 'inteligencia artificial'),
            (re.compile(r'^ml$',             re.IGNORECASE), 'aprendizaje automático'),
            (re.compile(r'^nlp$',            re.IGNORECASE), 'procesamiento lenguaje natural'),
            
            # Correcciones específicas de lematización con acentos
            (re.compile(r'corrupcion(ir)?$', re.IGNORECASE), 'corrupción'),
            (re.compile(r'politicair$',      re.IGNORECASE), 'política'),
            (re.compile(r'generativir$',     re.IGNORECASE), 'generativo'),
            (re.compile(r'inteligenciair$',  re.IGNORECASE), 'inteligencia'),
            (re.compile(r'gobiernoir$',      re.IGNORECASE), 'gobierno'),
            (re.compile(r'documentoir$',     re.IGNORECASE), 'documento'),
            (re.compile(r'mencionir$',       re.IGNORECASE), 'mencionar'),
            (re.compile(r'hablarir$',        re.IGNORECASE), 'hablar'),
            (re.compile(r'decirir$',         re.IGNORECASE), 'decir'),
            (re.compile(r'artificialir$',    re.IGNORECASE), 'artificial'),
            (re.compile(r'analizarir$',      re.IGNORECASE), 'analizar'),
            (re.compile(r'utilizarir$',      re.IGNORECASE), 'utilizar'),
            
            # Correcciones específicas para nombres propios y errores comunes
            (re.compile(r'^teslo$',          re.IGNORECASE), 'tesla'),
            (re.compile(r'^teslas$',         re.IGNORECASE), 'tesla'),
            (re.compile(r'^microsofto$',     re.IGNORECASE), 'microsoft'),
            (re.compile(r'^googleo$',        re.IGNORECASE), 'google'),
            (re.compile(r'^applo$',          re.IGNORECASE), 'apple'),
            (re.compile(r'^américo$',        re.IGNORECASE), 'américa'),
            (re.compile(r'^chino$',          re.IGNORECASE), 'china') 
        ]

    def GetSingleTextEntities(self, text):
        """
        NER (Named Entity Recognition), es un componente del procesamiento de lenguaje natural (PLN) que
        identifica categorías predefinidas de objetos en un texto. Categorías pueden incluir, entre otras, 
        nombres de personas, organizaciones, ubicaciones, expresiones de tiempos, cantidades, códigos médicos, 
        valores monetarios y porcentajes.
        """
        doc       = self.nlp(text)
        
        # Usamos un set para eliminar duplicados
        entidades_set = set(
            (ent.text, ent.label_) 
            for ent in doc.ents 
            #if ent.label_ != "MISC" --Siempre si se devolverán las entidades "MISCelaneas"
        )        
        
        # Convertimos de nuevo a lista de diccionarios
        entidades = [{"text": text, "label": label} for text, label in entidades_set]
        return entidades

    def _add_custom_entities(self):
        """Agrega entidades personalizadas al pipeline de spaCy"""
        # Crear un EntityRuler
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        # Patrones para entidades personalizadas
        patterns = [
            # --- Tratados comerciales ---
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": "T-?MEC"}}]},
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": "TMEC"}}]},
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": "USMCA"}}]},
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": "NAFTA"}}]},
            # --- Organismos y dependencias ---
            {"label": "ORG", "pattern": "IMSS"},
            {"label": "ORG", "pattern": "ISSSTE"},
            {"label": "ORG", "pattern": "SEP"},
            {"label": "ORG", "pattern": "SAT"},
            {"label": "ORG", "pattern": "SHCP"},
            {"label": "ORG", "pattern": "Banxico"},
            {"label": "ORG", "pattern": "PEMEX"},
            {"label": "ORG", "pattern": "UNAM"},
            {"label": "ORG", "pattern": "IPN"},
            {"label": "ORG", "pattern": "CFE"},
            {"label": "ORG", "pattern": "INAI"},
            {"label": "ORG", "pattern": "INE"},
            {"label": "ORG", "pattern": "SRE"},
            {"label": "ORG", "pattern": "SCJN"},
            {"label": "ORG", "pattern": "INE"},
            {"label": "ORG", "pattern": "SEGOB"},
            {"label": "ORG", "pattern": "SEDENA"},
            {"label": "ORG", "pattern": "SEMAR"},
            {"label": "ORG", "pattern": "SSP"},
            {"label": "ORG", "pattern": "SEMARNAT"},
            {"label": "ORG", "pattern": "PROFECO"},
            {"label": "ORG", "pattern": "CNDH"},
            {"label": "ORG", "pattern": "CONACYT"},
            {"label": "ORG", "pattern": "INEGI"},
            {"label": "ORG", "pattern": "CONAGUA"},
            {"label": "ORG", "pattern": "STPS"},
            {"label": "ORG", "pattern": "SECTUR"},
            # --- Partidos políticos ---
            {"label": "ORG", "pattern": "PRI"},
            {"label": "ORG", "pattern": "PAN"},
            {"label": "ORG", "pattern": "PRD"},
            {"label": "ORG", "pattern": "PVEM"},
            {"label": "ORG", "pattern": "PT"},
            {"label": "ORG", "pattern": "MC"},
            {"label": "ORG", "pattern": "Morena"},
            # --- Programas y políticas públicas ---
            {"label": "MISC", "pattern": "Bienestar"},
            {"label": "MISC", "pattern": "Sembrando Vida"},
            {"label": "MISC", "pattern": "Jóvenes Construyendo el Futuro"},
            {"label": "MISC", "pattern": "Becas Benito Juárez"},
            # --- Empresas e instituciones relevantes en México ---
            {"label": "ORG", "pattern": "Telmex"},
            {"label": "ORG", "pattern": "Telcel"},
            {"label": "ORG", "pattern": "América Móvil"},
            {"label": "ORG", "pattern": "Grupo Bimbo"},
            {"label": "ORG", "pattern": "Cemex"},
            {"label": "ORG", "pattern": "Femsa"},
            {"label": "ORG", "pattern": "Banamex"},
            {"label": "ORG", "pattern": "BBVA"},
            {"label": "ORG", "pattern": "Santander"},
            # --- Medios de comunicación ---
            {"label": "ORG", "pattern": "Televisa"},
            {"label": "ORG", "pattern": "TV Azteca"},
            {"label": "ORG", "pattern": "Milenio"},
            {"label": "ORG", "pattern": "El Universal"},
            {"label": "ORG", "pattern": "Reforma"},
            {"label": "ORG", "pattern": "La Jornada"},
            {"label": "ORG", "pattern": "El Norte"},
            # --- Figuras políticas muy frecuentes ---
            {"label": "PER", "pattern": "CSP"},
            {"label": "PER", "pattern": "AMLO"},
            {"label": "PER", "pattern": [{"LOWER": "claudia"}, {"LOWER": "sheinbaum"}],  "id": "Claudia Sheinbaum"},
            {"label": "PER", "pattern": [{"LOWER": "csp"}],                              "id": "Claudia Sheinbaum"},            
            {"label": "PER", "pattern": [{"LOWER": "marcelo"}, {"LOWER": "ebrard"}],     "id": "Marcelo Ebrard"},
            {"label": "PER", "pattern": [{"LOWER": "xóchitl"}, {"LOWER": "gálvez"}],     "id": "Xóchitl Gálvez"},
    
            {"label": "PER", "pattern": [{"LOWER": "felipe"}, {"LOWER": "calderón"}],    "id": "Felipe Calderón Hinojosa"},
            {"label": "PER", "pattern": [{"LOWER": "calderón"}],                         "id": "Felipe Calderón Hinojosa"},
            {"label": "PER", "pattern": [{"LOWER": "fch"}],                              "id": "Felipe Calderón Hinojosa"},

            {"label": "PER", "pattern": [{"LOWER": "vicente"}, {"LOWER": "fox"}],        "id": "Vicente Fox"},
            {"label": "PER", "pattern": [{"LOWER": "ernesto"}, {"LOWER": "zedillo"}],    "id": "Ernesto Zedillo"},
            {"label": "PER", "pattern": [{"LOWER": "luis"},    {"LOWER": "echeverría"}], "id": "Luis Echeverría"},
            {"label": "PER", "pattern": [{"LOWER": "enrique"}, {"LOWER": "peña"},   {"LOWER": "nieto"}], "id": "Enrique Peña Nieto"},
            {"label": "PER", "pattern": [{"TEXT":  "Andrés"},  {"TEXT": "Andres"},  {"TEXT": "Manuel"}, {"TEXT": "López"}, {"TEXT": "Lopez"}, {"TEXT": "Obrador"}]},
            {"label": "PER", "pattern": [{"LOWER": "andrés"},  {"LOWER": "manuel"}, {"LOWER": "lópez"}, {"LOWER": "obrador"}], "id": "Andrés Manuel López Obrador"},
            {"label": "PER", "pattern": [{"LOWER": "amlo"}],                             "id": "AMLO"}
        ]
        # --- Tecnologías y conceptos ---
        patterns_tecnologia = [
            {"label": "MISC", "pattern": [{"LOWER": "inteligencia"}, {"LOWER": "artificial"}], "id": "Inteligencia Artificial"},
            {"label": "MISC", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}], "id": "Machine Learning"},
            {"label": "MISC", "pattern": [{"LOWER": "aprendizaje"}, {"LOWER": "automático"}]},
            {"label": "MISC", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}], "id": "Deep Learning"},
            {"label": "MISC", "pattern": [{"LOWER": "big"}, {"LOWER": "data"}]},
            {"label": "MISC", "pattern": [{"LOWER": "análisis"}, {"LOWER": "de"}, {"LOWER": "datos"}]},
            {"label": "MISC", "pattern": [{"LOWER": "minería"}, {"LOWER": "de"}, {"LOWER": "datos"}]},
            {"label": "MISC", "pattern": [{"LOWER": "blockchain"}]},
            {"label": "MISC", "pattern": [{"LOWER": "criptomonedas"}]},
            {"label": "MISC", "pattern": [{"LOWER": "bitcoin"}]},
            {"label": "MISC", "pattern": [{"LOWER": "ethereum"}]},
            {"label": "MISC", "pattern": [{"LOWER": "realidad"}, {"LOWER": "aumentada"}]},
            {"label": "MISC", "pattern": [{"LOWER": "realidad"}, {"LOWER": "virtual"}]},
            {"label": "MISC", "pattern": [{"LOWER": "metaverso"}]},
            {"label": "MISC", "pattern": [{"LOWER": "internet"}, {"LOWER": "de"}, {"LOWER": "las"}, {"LOWER": "cosas"}]},
            {"label": "MISC", "pattern": [{"LOWER": "iot"}]},
            {"label": "MISC", "pattern": [{"LOWER": "computación"}, {"LOWER": "en"}, {"LOWER": "la"}, {"LOWER": "nube"}]},
            {"label": "MISC", "pattern": [{"LOWER": "cloud"}, {"LOWER": "computing"}]},
            {"label": "MISC", "pattern": [{"LOWER": "5g"}]},
            {"label": "MISC", "pattern": [{"LOWER": "redes"}, {"LOWER": "sociales"}]},
            {"label": "MISC", "pattern": [{"LOWER": "ciberseguridad"}]},
            {"label": "MISC", "pattern": [{"LOWER": "computación"}, {"LOWER": "cuántica"}]},
            {"label": "MISC", "pattern": [{"LOWER": "automatización"}]},
            {"label": "MISC", "pattern": [{"LOWER": "robots"}]},
            {"label": "MISC", "pattern": [{"LOWER": "vehículos"}, {"LOWER": "autónomos"}]},
            {"label": "MISC", "pattern": [{"LOWER": "autos"}, {"LOWER": "autónomos"}]},
            {"label": "MISC", "pattern": [{"LOWER": "smart"}, {"LOWER": "cities"}]},
            {"label": "MISC", "pattern": [{"LOWER": "ciudades"}, {"LOWER": "inteligentes"}]}
        ]
        # --- Inteligencia Artificial ---
        patterns_ia = [
            # Sigla común en español
            {"label": "MISC", "pattern": "IA"},
            {"label": "MISC", "pattern": [{"LOWER": "i.a."}]},   # algunas notas lo escriben con puntos
            
            # Sigla común en inglés
            {"label": "MISC", "pattern": "AI"},
            
            # Variante compuesta en inglés
            {"label": "MISC", "pattern": [{"LOWER": "artificial"}, {"LOWER": "intelligence"}]},
            
            # Variante extendida en español
            {"label": "MISC", "pattern": [{"LOWER": "sistemas"}, {"LOWER": "de"}, {"LOWER": "inteligencia"}, {"LOWER": "artificial"}]}
        ]
        ruler.add_patterns(patterns + patterns_tecnologia + patterns_ia)

    def GetEntities(self, Chunks):
        entities  = []
        
        for chunk in Chunks:        
            extracted  = self.GetSingleTextEntities(chunk)
            entities.append(extracted)
        return entities
        
        # doc       = self.nlp(text)
        # entidades = []
        # entidades = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ not in ("MISC")]
        # return entidades
        # #return jsonify(entidades)

    def GetChunksAndEntities(self, Text):
        ### Esta función y  GetChunks()  comparten totalmente la misma lógica
        ###  excepto por lo que retornan y la obtención de entidades
        ### Si hay un cambio o corrección que no sea relativo a eso, debe hacerse en las 2 funciones
        Chunks   = []
        Entities = []
        # Limpiamos texto
        Text = self.CleanText(Text)
        # Procesar el texto con spaCy
        doc = self.nlp(Text)
        
        #Aprovechamos que ya se tiene ese doc creado, para obtener las entidades
        # Usamos un set para eliminar duplicados
        entidades_set = set(
            (ent.text, ent.label_) 
            for ent in doc.ents 
            if ent.label_ != "MISC"
        )
        # Convertimos de nuevo a lista de diccionarios
        Entities = [{"text": text, "label": label} for text, label in entidades_set]
        
        # Dividir el texto en oraciones
        sentences = [sent.text for sent in doc.sents]
        
        if not sentences:  # Si no hay oraciones después del procesamiento
            return [Text]  # Devolver el texto completo como un único chunk

        # Generar embeddings para cada oración
        sentence_embeddings = self.EmbeddigModel.encode(sentences)

        # Calcular similitud entre oraciones consecutivas
        current_chunk = [sentences[0]]  # Primera oración, se asigna al primer chunk
        threshold = 0.60                # Umbral de similitud para agrupar oraciones

        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i + 1]

            # Calcular similitud coseno entre oraciones consecutivas
            similarity = cosine_similarity(
                [sentence_embeddings[i]],
                [sentence_embeddings[i + 1]]
            )[0][0]            

            # Si se parecen, se agrega al chunk que se está formando
            if similarity > threshold or len(next_sentence) < 150 or len(current_sentence) < 150:
                current_chunk.append(next_sentence)
            # Cuando ya no se parecen, el chunk que se formaba, se termina
            # Y se empieza a formar otro chunk
            else:
                Chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk.append(next_sentence)

        # Al terminar el ciclo, si había un chunk formándose se agrega como último generado
        if current_chunk:
            Chunks.append(" ".join(current_chunk))
        
        #Limpia los brincos de linea iniciales de cada chunk
        for i in range(len(Chunks) - 1):
            Chunks[i] = re.sub(r'^[\\r\\n]+', '', Chunks[i]).strip()
            
        return Chunks, Entities, Text
    
    def _is_running_in_docker(self):
        """Detecta si la aplicación se ejecuta dentro de Docker"""
        # Método 1: Verificar archivo .dockerenv
        if Path('/.dockerenv').exists():
            print("Método 1: Verificar archivo .dockerenv: '/.dockerenv' does exist")
            return True
        
        # Método 2: Verificar si estamos en un contenedor
        try:
            with open('/proc/1/cgroup', 'rt') as f:
                print("Método 2: Verificar si estamos en un contenedor 'docker' in '/proc/1/cgroup'")
                return 'docker' in f.read()
        except:
            pass
        
        # Método 3: Variable de entorno común en Docker
        if os.getenv('DOCKER_CONTAINER'):
            print("Método 3: 'Variable de entorno común en Docker: 'DOCKER_CONTAINER'")
            return True
        
        return False
    
    def _is_local_model_available(self):
        """Verifica si el modelo local existe"""
        local_path = Path("../ModelosIA/paraphrase-multilingual-MiniLM-L12-v2")
        return local_path.exists()