import pandas as pd

class textEncoder:
    def __init__(self,
                 df: pd.DataFrame, 
                 col_name_text: str = "content",
                 model_name: str = "all-MiniLM-L6-v2",
                 model_args: dict = {
                    "batch_size": 64,
                    "show_progress_bar": True,
                    "output_value": 'sentence_embedding', 
                    "convert_to_numpy": True,
                    "convert_to_tensor": False,
                    "device": None,
                    "normalize_embeddings": False
                 }
                 ):
        self.df = df
        self.col_name_text = col_name_text
        self.model_name = model_name
        self.model_args = model_args
        self.st_model = None
    
    def encode_sentence_transformer(self):
        """
        Obtains sentence embeddings and saves in .embeddings_sentence
        """
        self.load_model()
        sentences = self.df[self.col_name_text].to_list()
        print(f"[INFO] number of sentences to encode: {len(sentences)}")
        self.embeddings_sentence = self.st_model.encode(sentences, **self.model_args)
    
    def load_model(self, force_reload=False):
        """
        Loads model into .st_model
        """
        if (not force_reload) and (self.st_model is not None):
            print(f"[INFO] model is already loaded.")
            return
        detected_model_library = self.detect_model_library()
        if detected_model_library is None:
            raise NotImplementedError
        if detected_model_library == "sentence_embedding":
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
    
    def detect_model_library(self):
        """
        Checks if .model_name is a valid model in our library
        """
        model_dics = {
            "all-MiniLM-L6-v2": "sentence_embedding"
        }
        if self.model_name in model_dics:
            return model_dics[self.model_name]
        else:
            return None

