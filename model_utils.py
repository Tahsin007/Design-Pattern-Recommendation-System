import os
import sys
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMRanker
import joblib
import re

# Import the PatternVisualizer
from pattern_visualizer import PatternVisualizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'reranker_model.pkl')

class EnhancedPatternRecommender:
    """Enhanced version of the pattern recommender with visualization capabilities"""
    
    def __init__(self):
        self.sbert_model = None
        self.df_train = None
        self.enhanced_vectorizer = None
        self.reranker = None
        self.pattern_keywords = None
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.visualizer = PatternVisualizer()  # Initialize the visualizer
        
        # Initialize pattern descriptions for better context
        self.pattern_descriptions = {
            "Singleton": "Ensures a class has only one instance and provides a global point of access to it.",
            "Factory Method": "Defines an interface for creating an object, but lets subclasses decide which class to instantiate.",
            "Abstract Factory": "Provides an interface for creating families of related or dependent objects without specifying their concrete classes.",
            "Builder": "Separates the construction of a complex object from its representation, allowing the same construction process to create different representations.",
            "Prototype": "Creates new objects by copying an existing object, known as the prototype.",
            "Adapter": "Converts the interface of a class into another interface clients expect.",
            "Bridge": "Decouples an abstraction from its implementation so that the two can vary independently.",
            "Composite": "Composes objects into tree structures to represent part-whole hierarchies.",
            "Decorator": "Attaches additional responsibilities to an object dynamically.",
            "Facade": "Provides a unified interface to a set of interfaces in a subsystem.",
            "Flyweight": "Uses sharing to support large numbers of fine-grained objects efficiently.",
            "Proxy": "Provides a surrogate or placeholder for another object to control access to it.",
            "Chain of Responsibility": "Passes a request along a chain of handlers until one processes the request.",
            "Command": "Encapsulates a request as an object to parameterize clients with different requests.",
            "Iterator": "Provides a way to access elements of an aggregate object sequentially without exposing its underlying representation.",
            "Mediator": "Defines an object that encapsulates how a set of objects interact.",
            "Memento": "Captures and externalizes an object's internal state without violating encapsulation.",
            "Observer": "Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified.",
            "State": "Allows an object to alter its behavior when its internal state changes.",
            "Strategy": "Defines a family of algorithms, encapsulates each one, and makes them interchangeable.",
            "Template Method": "Defines the skeleton of an algorithm, deferring some steps to subclasses.",
            "Visitor": "Represents an operation to be performed on elements of an object structure."
        }
        
    def load_data(self, train_file_path):
        """Load training data"""
        self.df_train = pd.read_excel(train_file_path)
        
        # Preprocess training data
        train_columns = ['Intent', 'Applicability']
        for col in train_columns:
            self.df_train[col + "_processed"] = self.df_train[col].apply(self.preprocess_text)
        
        self.df_train["combined_text"] = self.df_train.apply(
            lambda row: ' '.join([row[col + "_processed"] for col in train_columns]), axis=1
        )
        
        # Compute SBERT embeddings
        self.df_train["vector"] = self.df_train["combined_text"].apply(lambda x: self.sbert_model.encode(x))
        
        # Extract pattern keywords
        self.pattern_keywords = self.extract_discriminative_features()
        
        # Create enhanced vectorizer
        self.enhanced_vectorizer = self.create_enhanced_vectorizer()
        self.X_train_enhanced = self.enhanced_vectorizer.fit_transform(self.df_train["combined_text"])
    
    def preprocess_text(self, text):
        """Clean and normalize text data"""
        text = str(text).lower()  # Convert to lowercase
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
        tokens = [word for word in tokens if word not in self.stop_words]  # Remove stopwords
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        return ' '.join(tokens)  # Return as a string
    
    def extract_discriminative_features(self):
        """Extract keywords that distinguish between patterns"""
        pattern_groups = self.df_train.groupby("Pattern Name")
        pattern_texts = {}

        for pattern, group in pattern_groups:
            combined = " ".join(group["combined_text"])
            pattern_texts[pattern] = combined

        # For each pattern, find keywords that distinguish it
        pattern_keywords = {}
        for target_pattern, target_text in pattern_texts.items():
            # Create corpus: target pattern vs all others combined
            other_patterns_text = " ".join([text for pattern, text in pattern_texts.items()
                                          if pattern != target_pattern])

            # Use TF-IDF to find distinctive words
            mini_corpus = [target_text, other_patterns_text]
            mini_vectorizer = TfidfVectorizer(max_features=50)
            X = mini_vectorizer.fit_transform(mini_corpus)

            # Words with high TF-IDF in target but low in others
            feature_names = mini_vectorizer.get_feature_names_out()
            target_scores = X[0].toarray()[0]

            # Get top 10 keywords
            top_indices = target_scores.argsort()[-10:][::-1]
            pattern_keywords[target_pattern] = [feature_names[i] for i in top_indices]

        return pattern_keywords
    
    def create_enhanced_vectorizer(self):
        # Create a custom analyzer that gives more weight to pattern-specific keywords
        def custom_analyzer(text):
            # Get base tokens
            tokens = text.split()

            # For each pattern, check if its keywords are in the text
            for pattern, keywords in self.pattern_keywords.items():
                # Count matches
                matches = sum(1 for keyword in keywords if keyword in text)
                # Add pattern name as a feature if matches found
                if matches > 0:
                    # Add weighted pattern name based on match count
                    tokens.extend([f"PATTERN_{pattern}"] * matches)

            return tokens

        # Create vectorizer with custom analyzer
        enhanced_vectorizer = TfidfVectorizer(analyzer=custom_analyzer, ngram_range=(1, 2))
        return enhanced_vectorizer
    
    def adjust_similarity(self, sbert_sim, tfidf_sim, pattern_name, test_text):
        adjustments = {
            "Bridge": 1.1 if "platform" in test_text else 0.9,
            "Decorator": 1.2 if "wrapper" in test_text else 1.0,
            "Singleton": 1.3 if "global" in test_text else 1.0
        }
        return (sbert_sim * adjustments.get(pattern_name, 1.0)) * 0.6 + \
               (tfidf_sim * adjustments.get(pattern_name, 1.0)) * 0.4
    
    def get_unique_top_patterns(self, scores, top_k=3):
        pattern_scores = {}

        for idx, score in enumerate(scores):
            pattern_name = self.df_train.iloc[idx]["Pattern Name"]
            # Keep only the highest score for each pattern
            if pattern_name not in pattern_scores or score > pattern_scores[pattern_name]["score"]:
                pattern_scores[pattern_name] = {"score": score, "index": idx}

        # Sort patterns by their highest score
        sorted_patterns = sorted(pattern_scores.items(),
                               key=lambda x: -x[1]["score"])[:top_k]

        # Return the top patterns with their original indices and scores
        return [(pattern[0], pattern[1]["index"], pattern[1]["score"])
                for pattern in sorted_patterns]
    
    def create_ranking_features(self, test_text_vector, test_text_processed, sbert_sim, tfidf_sim):
        """Create features optimized for LambdaMART"""
        scores = sbert_sim * 0.6 + tfidf_sim * 0.4  # Initial combined score

        top_patterns_info = self.get_unique_top_patterns(scores, top_k=15)
        features = []
        candidate_patterns = [] # Store pattern names for relevance matching

        for pattern_name, pattern_idx, similarity in top_patterns_info:
            # Ensure pattern_name exists in df_train before trying to access it
            pattern_rows = self.df_train[self.df_train["Pattern Name"] == pattern_name]
            if pattern_rows.empty:
                continue

            pattern_row = pattern_rows.iloc[0]
            pattern_text = pattern_row["combined_text"]

            # Calculate new features
            name_words = set(pattern_name.lower().split())
            problem_words = set(test_text_processed.split())
            # Handle potential division by zero if name_words is empty
            name_overlap = len(name_words & problem_words) / len(name_words) if len(name_words) > 0 else 0
            # Handle potential division by zero if test_text is empty
            keyword_density = sum(test_text_processed.count(kw) for kw in self.pattern_keywords.get(pattern_name, [])) / len(test_text_processed.split()) if len(test_text_processed.split()) > 0 else 0
            position_weight = 1 - (pattern_idx / len(self.df_train))

            bridge_vs_decorator = [
              int("abstraction" in test_text_processed and "implementation" in test_text_processed),
              int("wrapper" in test_text_processed or "enhance" in test_text_processed),
              len(re.findall(r"\bextension\b|\baddon\b", test_text_processed))
            ]

            template_method_indicators = [
              int("algorithm" in test_text_processed and "steps" in test_text_processed),
              int("skeleton" in test_text_processed or "framework" in test_text_processed),
              len(re.findall(r"\bhook\b|\boverride\b", test_text_processed))
            ]

            singleton_indicators = [
              int("unique" in test_text_processed or "global" in test_text_processed),
              int(re.search(r"\binstance\b.*\bone\b", test_text_processed) is not None),
              test_text_processed.count("access")
            ]

            # Feature vector construction
            feature_vec = [
                similarity,
                sbert_sim[pattern_idx],
                tfidf_sim[pattern_idx],
                len(pattern_row["Intent"]),
                len(pattern_row["Applicability"]),
                len(test_text_processed.split()),
                len(pattern_name.split()),
                sum(1 for kw in self.pattern_keywords.get(pattern_name, []) if kw in test_text_processed),
                len(set(test_text_processed.split()) & set(pattern_text.split())),
                pattern_idx / len(self.df_train),
                name_overlap,
                keyword_density,
                position_weight
            ]
            feature_vec.extend(bridge_vs_decorator)
            feature_vec.extend(template_method_indicators)
            feature_vec.extend(singleton_indicators)
            features.append(feature_vec)
            candidate_patterns.append(pattern_name)

        return features, candidate_patterns
    
    def get_reranked_recommendations(self, test_text, top_k=3):
        """Get recommendations for a single problem description"""
        # Preprocess the input text
        processed_text = self.preprocess_text(test_text)
        
        # Compute SBERT embedding
        text_vector = self.sbert_model.encode(processed_text)
        
        # Compute SBERT similarity
        sbert_similarities = cosine_similarity([text_vector], np.vstack(self.df_train["vector"]))[0]
        
        # Compute TF-IDF similarity
        X_test_enhanced = self.enhanced_vectorizer.transform([processed_text])
        enhanced_tfidf_similarities = cosine_similarity(X_test_enhanced, self.X_train_enhanced)[0]
        
        # Combine Similarity Scores (Weighted Hybrid Approach)
        final_scores = np.zeros_like(sbert_similarities)
        for j in range(len(self.df_train)):
            pattern_name = self.df_train.iloc[j]["Pattern Name"]
            final_scores[j] = self.adjust_similarity(
                sbert_similarities[j],
                enhanced_tfidf_similarities[j],
                pattern_name,
                processed_text
            )
        
        # Create features for reranking
        features, candidates = self.create_ranking_features(
            text_vector, processed_text, sbert_similarities, enhanced_tfidf_similarities
        )
        
        if not features:
            # Fallback to simple similarity if no features generated
            top_patterns = self.get_unique_top_patterns(final_scores, top_k)
            return [pattern[0] for pattern in top_patterns]
        
        # Get scores from LambdaMART
        scores = self.reranker.predict(np.array(features))
        
        # Sort by descending score
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        
        # Deduplicate while preserving order
        seen = set()
        results = []
        for score, pattern in ranked:
            if pattern not in seen:
                seen.add(pattern)
                results.append(pattern)
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_pattern_description(self, pattern_name):
        """Get the description of a pattern"""
        return self.pattern_descriptions.get(pattern_name, "No description available.")
    
    def get_pattern_diagram(self, pattern_name, problem_context=None):
        """Generate a class diagram for the given pattern"""
        return self.visualizer.generate_diagram(pattern_name, problem_context)
        
    def get_enhanced_recommendations(self, problem_text):
        """
        Get enhanced recommendations including the top pattern's diagram
        
        Args:
            problem_text: User's problem description
            
        Returns:
            Dictionary with recommendations, diagram, and description
        """
        # Get top patterns
        top1 = self.get_reranked_recommendations(problem_text, top_k=1)
        top3 = self.get_reranked_recommendations(problem_text, top_k=3)
        
        top1_pattern = top1[0] if top1 else "No recommendation"
        
        # Get diagram for top pattern
        diagram = None
        if top1_pattern != "No recommendation":
            # Use the problem text as context to potentially customize the diagram
            diagram = self.get_pattern_diagram(top1_pattern, problem_text)
            
        # Get description for top pattern
        description = self.get_pattern_description(top1_pattern)
            
        return {
            'problem': problem_text,
            'top1_recommendation': top1_pattern,
            'top3_recommendations': top3,
            'diagram': diagram,
            'description': description
        }

def load_model():
    """Load the trained model and data"""
    print("Loading model...")
    recommender = EnhancedPatternRecommender()
    
    # Load SBERT model
    recommender.sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Load training data
    train_file_path = 'data/Training_Dataset_RefactorGuru.xlsx'
    test_file_path = 'data/Expanded_Design_Pattern_Dataset.xlsx'
    recommender.load_data(train_file_path)
    
    # Load or train reranker
    if not os.path.exists(MODEL_PATH):
        print("Reranker not found, training new model...")
        # Import the training function
        from pattern_recommender import train_reranker, prepare_training_data
        
        # Load test data for training
        df_test = pd.read_excel(test_file_path)
        df_test["processed_text"] = df_test["Pattern Description"].apply(recommender.preprocess_text)
        df_test["vector"] = df_test["processed_text"].apply(lambda x: recommender.sbert_model.encode(x))
        
        # Train the reranker
        train_reranker(recommender, df_test)
    
    recommender.reranker = joblib.load(MODEL_PATH)
    
    return recommender

def get_recommendations_with_diagram(model, problem_text):
    """Get enhanced recommendations with class diagram"""
    return model.get_enhanced_recommendations(problem_text)