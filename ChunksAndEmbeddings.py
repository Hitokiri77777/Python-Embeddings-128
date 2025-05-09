from   sentence_transformers    import SentenceTransformer
from   sklearn.metrics.pairwise import cosine_similarity
from   bs4                      import BeautifulSoup
import faiss
import spacy
import re 
import os

class ChunksAndEmbeddings:
    def __init__(self):
        self.PCA_Matrix_FileName = 'DB_matrix.faiss'
        self.PCA_Matrix_Trained  = faiss.read_VectorTransform(self.PCA_Matrix_FileName)

    def Load_LanguageModel(self):
        # Cargar modelo de embeddings desde disco duro
        #self.EmbeddigModel = SentenceTransformer("../ModelosIA/paraphrase-multilingual-MiniLM-L12-v2", )
        # Cargar modelo de embeddings desde internet 
        self.EmbeddigModel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Cargar modelo de Spacy para español
        self.nlp = spacy.load("es_core_news_md")
        
        # Añadir reglas para abreviaturas
        abreviaturas = ["Sr.", "Sra.", "Dr.", "Dra.", "vs.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
        for abrev in abreviaturas:
            self.nlp.tokenizer.add_special_case(abrev, [{"ORTH": abrev}])    
    
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
