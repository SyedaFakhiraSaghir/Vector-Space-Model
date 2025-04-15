import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import math
import nltk

# downloading NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# defining the GUI and search system class
class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VSM")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f5")

        # setting up GUI styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        # initializing the vector space model search engine
        self.vsm = VectorSpaceModel("Abstracts", "Stopword-List.txt", tf_method='log', idf_method='smooth')

        # building the GUI widgets
        self.create_widgets()

        # loading the index in background after 100ms
        self.root.after(100, self.initialize_engine)

    def configure_styles(self):
        self.style.configure('TFrame', background="#f0f2f5")
        self.style.configure('TLabel', background="#f0f2f5", font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10), padding=6)
        self.style.configure('Search.TButton', foreground='white', background='#4CAF50')
        self.style.map('Search.TButton', background=[('active', '#45a049')])
        self.style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#333333')

    def create_widgets(self):
        # creating header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=20, padx=20, fill=tk.X)

        ttk.Label(
            header_frame,
            text="Vector space model",
            style='Title.TLabel'
        ).pack(side=tk.LEFT)

        # creating search bar
        search_frame = ttk.Frame(self.root)
        search_frame.pack(pady=10, padx=20, fill=tk.X)

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(
            search_frame,
            textvariable=self.search_var,
            font=('Segoe UI', 12),
            width=50
        )
        search_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        search_entry.bind('<Return>', lambda e: self.perform_search())

        ttk.Button(
            search_frame,
            text="Search",
            style='Search.TButton',
            command=self.perform_search
        ).pack(side=tk.LEFT)

        # creating area for showing results
        results_frame = ttk.Frame(self.root)
        results_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg='white',
            padx=10,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # creating status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing search engine...")

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def initialize_engine(self):
        try:
            self.vsm.build_index()  # building index of documents
            self.status_var.set("Ready. Enter your search query")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize search engine: {str(e)}")
            self.root.destroy()

    def perform_search(self):
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search term")
            return

        self.status_var.set(f"Searching for: {query}...")
        self.root.update()

        try:
            doc_nums = self.vsm.search(query)  # processing the query and searching
            self.display_results(query, sorted(doc_nums))
            self.status_var.set(f"Found {len(doc_nums)} documents for: {query}")
        except Exception as e:
            messagebox.showerror("Search Error", f"An error occurred: {str(e)}")
            self.status_var.set("Search failed")

    def display_results(self, query, doc_nums):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, f"Results for: ", 'bold')
        self.results_text.insert(tk.END, f"'{query}'\n\n", 'query')

        if not doc_nums:
            self.results_text.insert(tk.END, "No matching documents found\n", 'error')
            self.results_text.config(state=tk.DISABLED)
            return

        # displaying results in lines of 10
        line = []
        for i, num in enumerate(doc_nums, 1):
            line.append(str(num))
            if i % 10 == 0 or i == len(doc_nums):
                self.results_text.insert(tk.END, " , ".join(line) + (" ," if i < len(doc_nums) else ""))
                self.results_text.insert(tk.END, "\n")
                line = []

        self.results_text.tag_config('bold', font=('Segoe UI', 10, 'bold'))
        self.results_text.tag_config('query', foreground='#4CAF50')
        self.results_text.tag_config('error', foreground='#f44336')
        self.results_text.config(state=tk.DISABLED)

# defining the vector space model with TF-IDF options
class VectorSpaceModel:
    def __init__(self, abstracts_dir, stopwords_file, tf_method='raw', idf_method='standard'):
        self.abstracts_dir = abstracts_dir
        self.stopwords = self.load_stopwords(stopwords_file)
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.documents = []
        self.terms = set()
        self.doc_term_freq = defaultdict(dict)
        self.term_doc_freq = defaultdict(int)
        self.term_docs = defaultdict(set)
        self.N = 0
        self.processed = False
        self.tf_method = tf_method
        self.idf_method = idf_method

    def load_stopwords(self, stopwords_file):
        with open(stopwords_file, 'r') as f:
            return set(line.strip() for line in f)

    # preprocessing documents
    def preprocess_text(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        processed_tokens = []
        for token in tokens:
            if token and token not in self.stopwords:
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
        return processed_tokens

    # creating the index of documents
    def build_index(self):
        for filename in os.listdir(self.abstracts_dir):
            if filename.endswith('.txt'):
                doc_id = filename.split('.')[0]
                try:
                    int(doc_id)  # making sure doc ID is numeric
                except ValueError:
                    continue

                filepath = os.path.join(self.abstracts_dir, filename)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            text = f.read()
                    except Exception:
                        continue

                tokens = self.preprocess_text(text)  # preprocessing text
                self.documents.append(doc_id)

                term_freq = defaultdict(int)
                for term in tokens:
                    term_freq[term] += 1
                    self.terms.add(term)
                    self.term_docs[term].add(doc_id)

                self.doc_term_freq[doc_id] = term_freq

                for term in term_freq:
                    self.term_doc_freq[term] += 1

        self.N = len(self.documents)
        self.processed = True

    # computing TF
    def compute_tf(self, count):
        if self.tf_method == 'raw':
            return count
        elif self.tf_method == 'log':
            return 1 + math.log(count)
        elif self.tf_method == 'bool':
            return 1 if count > 0 else 0
        else:
            return count

    # computing IDF
    def compute_idf(self, term):
        df = self.term_doc_freq.get(term, 0)
        if self.idf_method == 'smooth':
            return math.log(1 + (self.N / (1 + df)))
        elif self.idf_method == 'prob':
            return math.log((self.N - df + 1) / (df + 1))
        else:
            return math.log(self.N / (df + 1))

    # handling query processing and computing scores
    def search(self, query_text):
        if not self.processed:
            self.build_index()

        query_terms = self.preprocess_text(query_text)
        doc_scores = defaultdict(float)

        for term in query_terms:
            idf = self.compute_idf(term)
            for doc_id in self.term_docs.get(term, []):
                tf = self.compute_tf(self.doc_term_freq[doc_id].get(term, 0))
                doc_scores[doc_id] += tf * idf

        try:
            return sorted([int(doc_id) for doc_id in doc_scores], key=lambda x: -doc_scores[str(x)])
        except ValueError:
            return []

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()
