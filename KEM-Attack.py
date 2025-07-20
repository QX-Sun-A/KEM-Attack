import os
import tensorflow_hub as hub
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.modeling import BertForSequenceClassification, BertConfig
from BERT.tokenization import BertTokenizer

import numpy as np
import argparse
import nltk
from nltk.corpus import stopwords, wordnet
import OpenHowNet
import os
from tqdm import tqdm
import dataloader
from train_classifier import Model
import json

from sentence_transformers import SentenceTransformer, util
import editdistance

#--------------------------------------------------------
# Module 1: Semantic Similarity Calculation
#--------------------------------------------------------

class SemanticSimilarity:
    """Base class for semantic similarity calculation"""
    def semantic_sim(self, sents1, sents2):
        """Calculate semantic similarity between two groups of text"""
        raise NotImplementedError("Subclasses must implement semantic_sim method")

class USE(SemanticSimilarity):
    """Universal Sentence Encoder semantic similarity calculation"""
    def __init__(self, cache_path):
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "universal-sentence-encoder-tensorflow2-large-v2"
        self.embed = hub.load(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores
    
    def close(self):
        """Release TensorFlow session resources"""
        if hasattr(self, 'sess'):
            self.sess.close()
            tf.reset_default_graph()
    
    def __del__(self):
        """Destructor to ensure resource release"""
        self.close()

class SentenceBERT(SemanticSimilarity):
    """SentenceBERT semantic similarity calculation"""
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def semantic_sim(self, sents1, sents2):
        embeddings1 = self.model.encode(sents1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sents2, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return cosine_scores.cpu().numpy()

#--------------------------------------------------------
# Module 2: BERT Model Interface
#--------------------------------------------------------

class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class NLIDataset_BERT(Dataset):
    def __init__(self, pretrained_dir, max_seq_length=128, batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        features = []
        for text_a in examples:
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            features.append(
                InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        eval_features = self.convert_examples_to_features(data, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader

class NLI_infer_BERT(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        if torch.cuda.is_available():
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        else:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)

        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)
        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

#--------------------------------------------------------
# Module 3: Synonym Generation
#--------------------------------------------------------

def initialize_hownet():
    """Initialize OpenHowNet interface"""
    hownet = OpenHowNet.HowNetDict()
    return hownet

def get_stopwords(custom_stopwords_path=None, preserved_stopwords=None):
    """Get stopwords set"""
    stop_words = set(stopwords.words('english'))

    # Load custom stopwords (if path exists)
    if custom_stopwords_path and os.path.exists(custom_stopwords_path):
        with open(custom_stopwords_path, 'r', encoding='utf-8') as f:
            custom_stopwords = set(line.strip() for line in f)
            stop_words.update(custom_stopwords)

    # Exclude preserved stopwords
    if preserved_stopwords:
        preserved_words = set(word.strip() for word in preserved_stopwords.split(','))
        stop_words -= preserved_words

    return stop_words

def calculate_sememe_similarity(sememes1, sememes2):
    """Calculate similarity between two sets of sememes"""
    common_sememes = set(sememes1) & set(sememes2)
    similarity = len(common_sememes) / max(len(sememes1), len(sememes2)) if max(len(sememes1), len(sememes2)) > 0 else 0
    return similarity

def get_synonyms(word, hownet, synonym_num=20, sim_threshold=0.5):
    """Get high-quality synonyms using WordNet and HowNet - optimized version"""
    # Use WordNet for base synonyms
    synsets = wordnet.synsets(word)
    synonyms = set()
    
    # 1. Get all synonyms from all parts of speech
    for syn in synsets:
        for lemma in syn.lemmas():
            syn_name = lemma.name().replace('_', ' ')
            synonyms.add(syn_name)
    
    # 2. Add derivations and variants
    word_forms = set(synonyms)
    for syn_word in list(synonyms):
        # Get singular/plural, comparative forms, etc.
        synsets_for_word = wordnet.synsets(syn_word)
        for syn in synsets_for_word:
            for lemma in syn.lemmas():
                # Add derivational forms
                for derived in lemma.derivationally_related_forms():
                    word_forms.add(derived.name().replace('_', ' '))
    
    # Merge all synonyms
    synonyms = list(synonyms.union(word_forms))
    
    if not synonyms:
        return []

    # Return WordNet results if HowNet is unavailable
    if hownet is None:
        return synonyms[:synonym_num]

    # Try to get sememe information
    try:
        word_sememes = hownet.get_sememes_by_word(word, lang="en")
        if not word_sememes:
            word_sememes = hownet.get_sememes_by_word(word)
    except Exception as e:
        word_sememes = None

    # Return WordNet synonyms if no sememes found
    if not word_sememes:
        return synonyms[:synonym_num]

    # Filter synonyms - enhanced version
    filtered_synonyms = []
    backup_synonyms = []
    
    for syn in synonyms:
        # Try to get sememes for synonym
        try:
            syn_sememes = hownet.get_sememes_by_word(syn, lang="en")
            if not syn_sememes:
                syn_sememes = hownet.get_sememes_by_word(syn)
        except:
            syn_sememes = None
        
        # Calculate sememe similarity
        if syn_sememes:
            similarity = calculate_sememe_similarity(word_sememes, syn_sememes)
            
            # Strict and backup filtering
            if similarity >= sim_threshold:
                filtered_synonyms.append((syn, similarity))
            elif similarity >= sim_threshold * 0.7:  # Relaxed backup standard
                backup_synonyms.append((syn, similarity))
    
    # Sort by similarity
    filtered_synonyms.sort(key=lambda x: x[1], reverse=True)
    filtered_words = [word for word, _ in filtered_synonyms]
    
    # Add backup words if strict filtering yields insufficient results
    if len(filtered_words) < synonym_num and backup_synonyms:
        backup_synonyms.sort(key=lambda x: x[1], reverse=True)
        backup_words = [word for word, _ in backup_synonyms]
        filtered_words.extend(backup_words)
    
    # Add original synonyms if still insufficient
    if len(filtered_words) < synonym_num:
        remaining = synonym_num - len(filtered_words)
        unused_words = [w for w in synonyms if w not in filtered_words]
        filtered_words.extend(unused_words[:remaining])
    
    return filtered_words[:synonym_num]

def generate_candidates_for_text(text, hownet_handler, stop_words, relaxed=False, top_k=3, synonym_num=20, sim_threshold=0.5):
    """Generate candidate replacements for each word in text with filtering"""
    candidates_dict = {}

    for i, word in enumerate(text):
        is_stopword = word in stop_words
        candidates = []

        if not is_stopword:
            # Get synonyms
            synonyms = get_synonyms(word, hownet_handler, synonym_num=synonym_num, sim_threshold=sim_threshold)
            candidates = synonyms

        candidates_dict[i] = {
            'is_stopword': is_stopword,
            'candidates': candidates
        }

    return candidates_dict

def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):
    """Calculate semantic similarity between original and candidate texts within local range"""
    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_range_min = 0
        text_range_max = len_text
        
    batch_size = 16
    total_semantic_sims = np.array([])
    for i in range(0, len(new_texts), batch_size):
        batch = new_texts[i:i+batch_size]
        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), batch)))[0]
        total_semantic_sims = np.concatenate((total_semantic_sims, semantic_sims))
    return total_semantic_sims

#--------------------------------------------------------
# Module 4: Discrete Grey Wolf Optimizer
#--------------------------------------------------------

def calculate_edit_distance(text1, text2):
    """Calculate edit distance"""
    return editdistance.eval(text1, text2)

def dynamic_mutation_rate(iteration, max_iterations, initial_mutation_rate=0.05, final_mutation_rate=0.01):
    """Dynamically adjust mutation probability"""
    return initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * (iteration / max_iterations)

class DGWO:
    """Discrete Grey Wolf Optimization algorithm"""
    def __init__(self, text, candidates_dict, predictor, orig_label,
                 sim_predictor, wolf_population=50, max_iterations=100,
                 mutation_rate=0.05, max_replacements=5, sim_threshold=0.8):
        self.text = text
        self.candidates_dict = candidates_dict
        self.predictor = predictor
        self.orig_label = orig_label
        self.sim_predictor = sim_predictor
        self.wolf_population = wolf_population
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.max_replacements = max_replacements
        self.sim_threshold = sim_threshold
        self.text_len = len(text)
        self.query_count = 0  # Query counter

        # 1. Get all replaceable position indices
        self.valid_positions = [
            i for i, info in candidates_dict.items()
            if not info['is_stopword'] and len(info['candidates']) > 0
        ]

        # 2. Get stopword position indices
        self.stopword_positions = [
            i for i, info in candidates_dict.items()
            if info['is_stopword'] and len(info['candidates']) > 0
        ]

        # 3. Create candidate word index structure
        self.candidates_by_position = {}
        for pos in self.valid_positions:
            self.candidates_by_position[pos] = self.candidates_dict[pos]['candidates']

        # 4. Initialize population
        self.wolves = self.initialize_population()

    def initialize_population(self):
        """Initialize wolf population - reflecting GWO's exploration/exploitation balance"""
        wolves = []
        
        # 1. Exploration wolves - corresponding to early stage GWO (a near 2)
        explore_count = int(self.wolf_population * 0.4)
        for _ in range(explore_count):
            # More replacements (2-4 words)
            num_replacements = min(np.random.randint(2, 5), len(self.valid_positions))
            if self.valid_positions:
                positions = np.random.choice(self.valid_positions, 
                                            min(num_replacements, len(self.valid_positions)), 
                                            replace=False)
                replacements = []
                for pos in positions:
                    candidates = self.candidates_dict[pos]['candidates']
                    if candidates:
                        replacement = (pos, self.text[pos], np.random.choice(candidates))
                        replacements.append(replacement)
                wolves.append(replacements)
            else:
                wolves.append([])
        
        # 2. Exploitation wolves - corresponding to late stage GWO (a near 0)
        exploit_count = int(self.wolf_population * 0.4)
        for _ in range(exploit_count):
            # Fewer replacements (1-2 words)
            num_replacements = min(np.random.randint(1, 3), len(self.valid_positions))
            if self.valid_positions:
                positions = np.random.choice(self.valid_positions, 
                                            min(num_replacements, len(self.valid_positions)), 
                                            replace=False)
                replacements = []
                for pos in positions:
                    candidates = self.candidates_dict[pos]['candidates']
                    if candidates:
                        replacement = (pos, self.text[pos], np.random.choice(candidates))
                        replacements.append(replacement)
                wolves.append(replacements)
            else:
                wolves.append([])
        
        # 3. Fill remaining wolves
        remaining = self.wolf_population - len(wolves)
        for _ in range(remaining):
            wolves.append(self.initialize_one_wolf())
        
        return wolves

    def calculate_fitness(self, wolf):
        """Calculate fitness for a wolf"""
        if not wolf:  # If no replacements, return original text and minimum fitness
            return self.text, 0, False, 0

        # Apply replacements to generate adversarial sample
        adv_text = self.text.copy()
        for pos, _, replacement in wolf:
            adv_text[pos] = replacement

        # Determine if attack is successful
        with torch.no_grad():
            adv_probs = self.predictor([adv_text])
            self.query_count += 1  # Increase query count
            predicted_label = torch.argmax(adv_probs, dim=-1).item()
            success = (predicted_label != self.orig_label)

        # Calculate semantic similarity
        sim_score = self.sim_predictor.semantic_sim([' '.join(self.text)], [' '.join(adv_text)])[0][0]

        # Calculate replacement ratio
        replace_ratio = len(wolf) / self.text_len

        # Multi-objective fitness function
        fitness = 0
        if success:
            fitness += 100  # Attack success reward

        fitness += sim_score * 30  # Semantic similarity reward
        
        # Dynamic modification magnitude penalty
        length_factor = min(1.0, 20.0/self.text_len) if self.text_len > 20 else 1.0
        fitness += (1 - replace_ratio) * 50 * length_factor

        # If similarity is below threshold, greatly reduce fitness
        if sim_score < self.sim_threshold:
            fitness -= 50

        edit_distance = calculate_edit_distance(self.text, adv_text)

        return adv_text, fitness, success, edit_distance
    
    def calculate_fitness_batch(self, wolves_batch):
        """Batch calculate fitness for multiple wolves - optimized version"""
        if not wolves_batch:
            return []
        
        # 1. Batch generate adversarial texts
        adv_texts = []
        for wolf in wolves_batch:
            if not wolf:
                adv_texts.append(self.text.copy())
                continue
                
            adv_text = self.text.copy()
            for pos, _, replacement in wolf:
                adv_text[pos] = replacement
            adv_texts.append(adv_text)
        
        # 2. Batch predict labels
        with torch.no_grad():
            batch_probs = self.predictor(adv_texts)
            self.query_count += len(adv_texts)  # Batch increase query count
            batch_labels = torch.argmax(batch_probs, dim=-1).cpu().numpy()
        
        # 3. Batch calculate semantic similarity
        original_texts = [' '.join(self.text)] * len(adv_texts)
        joined_adv_texts = [' '.join(text) for text in adv_texts]
        
        sim_scores = self.sim_predictor.semantic_sim(original_texts, joined_adv_texts)[0]
        
        # 4. Calculate fitness for each wolf
        results = []
        for i, wolf in enumerate(wolves_batch):
            # First calculate edit distance
            edit_distance = calculate_edit_distance(self.text, adv_texts[i])
            
            success = (batch_labels[i] != self.orig_label)
            sim_score = sim_scores[i]
            replace_ratio = len(wolf) / self.text_len if wolf else 0
            
            # Dynamic fitness calculation, considering text length
            length_factor = min(1.0, 20.0/self.text_len) if self.text_len > 20 else 1.0
            
            # Multi-objective fitness function
            fitness = 0
            
            # 1. Attack success reward - extra check for original label probability drop
            if success:
                # Get probability for original class, more reduction means higher reward
                orig_class_prob = batch_probs[i][self.orig_label].item()
                confidence_bonus = max(0, (1.0 - orig_class_prob) * 50)  # Up to 50 extra points
                fitness += 150 + confidence_bonus
            
            # 2. Semantic similarity reward - weight inversely proportional to edit distance
            semantic_weight = max(20, 40 - min(20, edit_distance))  # Dynamically varies between 20-40
            fitness += sim_score * semantic_weight
            
            # 3. Modification magnitude penalty - reduced for short texts
            replacement_penalty = replace_ratio * 40 * length_factor
            fitness -= replacement_penalty
            
            # 4. Similarity threshold penalty - severe penalty for not meeting similarity requirements
            if sim_score < self.sim_threshold:
                # Penalty proportional to threshold gap
                threshold_gap = self.sim_threshold - sim_score
                fitness -= 50 + threshold_gap * 100  # Base penalty 50 + dynamic penalty
                
                # Special case: If successful but similarity just slightly below threshold, reduce penalty
                if success and sim_score > self.sim_threshold * 0.9:
                    fitness += 30  # Slightly reduce penalty
            
            results.append((adv_texts[i], fitness, success, edit_distance))
        
        return results

    def update_positions(self, alpha, beta, delta, a):
        """Update wolf positions - fully reflecting the original GWO algorithm's a parameter"""
        new_wolves = []
        
        # Determine current search stage based on a value
        exploration_stage = a > 1.0  # Early exploration stage is True
        
        for wolf in self.wolves:
            # Skip leader wolves
            if wolf == alpha or wolf == beta or wolf == delta:
                new_wolves.append(wolf)
                continue
                
            # New replacement scheme
            new_replacements = []
            all_positions = set()
            
            # Simulate GWO's A and C parameters
            r1 = np.random.random()
            A = 2*a*r1 - a  # A ranges from [-a, a]
            r2 = np.random.random()
            C = 2*r2  # C ranges from [0, 2]
            
            # Based on |A|>1 determine exploration or exploitation
            if abs(A) > 1:  # Exploration phase
                # Weaken leader wolf influence
                alpha_influence = max(0.2, min(0.4, 0.3 * C))
                beta_influence = max(0.1, min(0.3, 0.2 * C))
                delta_influence = max(0.0, min(0.2, 0.1 * C))
                
                # Enhance random selection
                for leader, prob in [(alpha, alpha_influence), (beta, beta_influence), (delta, delta_influence)]:
                    if not leader:
                        continue
                        
                    # Randomly select few features to inherit
                    num_inherit = max(1, min(len(leader) // 3, 2))
                    if leader and np.random.random() < prob:
                        selected_idx = np.random.choice(range(len(leader)), min(num_inherit, len(leader)), replace=False)
                        for idx in selected_idx:
                            if idx < len(leader):
                                pos, orig, repl = leader[idx]
                                if pos not in all_positions:
                                    new_replacements.append((pos, orig, repl))
                                    all_positions.add(pos)
                
                # Strengthen random exploration
                if self.valid_positions:
                    available_positions = [p for p in self.valid_positions if p not in all_positions]
                    if available_positions:
                        # More replacements during exploration
                        num_explore = min(2 + int(a), len(available_positions))
                        explore_positions = np.random.choice(
                            available_positions, 
                            min(num_explore, len(available_positions)),
                            replace=False
                        )
                        
                        for pos in explore_positions:
                            candidates = self.candidates_dict[pos]['candidates']
                            if candidates:
                                replacement = (pos, self.text[pos], np.random.choice(candidates))
                                new_replacements.append(replacement)
                                all_positions.add(pos)
                                
            else:  # Exploitation phase (|A|<=1)
                # Enhance leader wolf influence, reduce randomness
                alpha_influence = min(0.9, max(0.5, 0.8 - a * 0.3))  # Lower a means higher inheritance probability
                beta_influence = min(0.7, max(0.3, 0.6 - a * 0.3))
                delta_influence = min(0.5, max(0.1, 0.4 - a * 0.3))
                
                # Focus on inheriting more features from leader wolves
                for leader, prob in [(alpha, alpha_influence), (beta, beta_influence), (delta, delta_influence)]:
                    if not leader:
                        continue
                        
                    for pos, orig, repl in leader:
                        if pos not in all_positions and np.random.random() < prob:
                            new_replacements.append((pos, orig, repl))
                            all_positions.add(pos)
                
                # Slight local search
                local_mutation_prob = self.mutation_rate * (1 - a/2)  # Lower a means higher mutation probability
                if np.random.random() < local_mutation_prob and self.valid_positions:
                    available_positions = [p for p in self.valid_positions if p not in all_positions]
                    if available_positions:
                        # Only replace 1 word during exploitation
                        pos = np.random.choice(available_positions)
                        candidates = self.candidates_dict[pos]['candidates']
                        if candidates:
                            replacement = (pos, self.text[pos], np.random.choice(candidates))
                            new_replacements.append(replacement)
                            all_positions.add(pos)
            
            # Special case: If no replacements, randomly add one
            if not new_replacements and self.valid_positions:
                pos = np.random.choice(self.valid_positions)
                candidates = self.candidates_dict[pos]['candidates']
                if candidates:
                    replacement = (pos, self.text[pos], np.random.choice(candidates))
                    new_replacements.append(replacement)
            
            # Limit maximum replacements
            if len(new_replacements) > self.max_replacements:
                np.random.shuffle(new_replacements)
                new_replacements = new_replacements[:self.max_replacements]
                
            new_wolves.append(new_replacements)
        
        return new_wolves
    
    def reduce_search_space(self, adv_text, orig_label, a_value):
        """Intelligently reduce search space - adjust strategy strength based on a value"""
        # Only reduce during exploitation phase (small a)
        if a_value > 1.0:
            return adv_text
        
        # Record current edit distance
        initial_edit_distance = calculate_edit_distance(self.text, adv_text)
        if initial_edit_distance <= 1:  # If only one word replaced, no need to reduce
            return adv_text
        
        improved = True
        rounds = 0
        max_rounds = 3  # Maximum 3 reduction rounds
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            best_sim_improved = -float('inf')
            best_position = None
            
            # Try restoring each replaced word
            for i in range(len(self.text)):
                if adv_text[i] != self.text[i]:
                    # Temporarily restore this position
                    temp_text = adv_text.copy()
                    temp_text[i] = self.text[i]
                    
                    # Check if attack remains successful after restoration
                    with torch.no_grad():
                        adv_probs = self.predictor([temp_text])
                        self.query_count += 1  # Increase query count
                        new_label = torch.argmax(adv_probs, dim=-1).item()
                    
                    if new_label != orig_label:  # Attack still successful
                        # Calculate semantic similarity improvement
                        orig_sim = self.sim_predictor.semantic_sim(
                            [' '.join(self.text)], [' '.join(adv_text)])[0][0]
                        new_sim = self.sim_predictor.semantic_sim(
                            [' '.join(self.text)], [' '.join(temp_text)])[0][0]
                        sim_improvement = new_sim - orig_sim
                        
                        # Find position with maximum improvement
                        if sim_improvement > best_sim_improved:
                            best_sim_improved = sim_improvement
                            best_position = i
            
            # Apply best restoration
            if best_position is not None:
                adv_text[best_position] = self.text[best_position]
                improved = True
                print(f"✓ Restored position {best_position} word: '{self.text[best_position]}', similarity improved: {best_sim_improved:.4f}")
            
        # Calculate reduction effect
        final_edit_distance = calculate_edit_distance(self.text, adv_text)
        reduced_words = initial_edit_distance - final_edit_distance
        
        if reduced_words > 0:
            print(f"★ Search space reduction successful! Reduced {reduced_words} word replacements")
        
        return adv_text
    
    def optimize(self):
        """Execute enhanced optimization process - dynamic exploration and intelligent restart"""
        stagnation_count = 0
        best_fitness_so_far = -float('inf')
        best_success_solution = None
        best_success_adv_text = None
        best_success_edit_distance = float('inf')
        
        # Add temperature parameter to control exploration intensity
        temperature = 1.0
        min_temperature = 0.3
        
        for iteration in tqdm(range(self.max_iterations), desc="GWO Iteration"):
            # 1. Dynamically adjust mutation rate based on stagnation
            if stagnation_count > 5:
                # Increase mutation rate to promote exploration when stagnating
                self.mutation_rate = min(0.25, self.mutation_rate * 1.5)
                temperature = max(min_temperature, temperature * 0.8)  # Lower temperature to enhance exploration
                
                # Reset part of population when severely stagnating
                if stagnation_count > 10:
                    print("\n⚠️ Severe stagnation detected, resetting 30% of population...")
                    num_to_reset = max(5, int(self.wolf_population * 0.3))
                    random_indices = np.random.choice(range(self.wolf_population), num_to_reset, replace=False)
                    for idx in random_indices:
                        self.wolves[idx] = self.initialize_one_wolf()
                    stagnation_count = max(5, stagnation_count // 2)  # Partially reset stagnation count
            else:
                # Normal iteration mutation rate for fine-grained search
                self.mutation_rate = dynamic_mutation_rate(iteration, self.max_iterations, 
                                                        initial_mutation_rate=0.1,
                                                        final_mutation_rate=0.05)
            
            # 2. Dynamic batch size based on text length for performance optimization
            batch_size = min(32, max(4, 128 // len(self.text))) if len(self.text) > 0 else 16
            
            # 3. Batch calculate fitness
            fitness_results = []
            for i in range(0, len(self.wolves), batch_size):
                batch = self.wolves[i:i+batch_size]
                batch_results = self.calculate_fitness_batch(batch)
                
                for j, result in enumerate(batch_results):
                    wolf_idx = i + j
                    if wolf_idx < len(self.wolves):
                        fitness_results.append((self.wolves[wolf_idx], *result))
            
            # 4. Sort by fitness
            fitness_results.sort(key=lambda x: x[2], reverse=True)
            current_best_fitness = fitness_results[0][2]
            
            # 5. Check for improvement
            if current_best_fitness > best_fitness_so_far:
                improvement = current_best_fitness - best_fitness_so_far
                best_fitness_so_far = current_best_fitness
                stagnation_count = 0
                
                # Adjust temperature based on improvement magnitude
                if improvement > 10:
                    temperature = min(1.0, temperature * 1.2)  # Increase temperature for significant improvements
            else:
                stagnation_count += 1
            
            # 6. Save successful attack solutions
            for wolf, adv_text, _, success, edit_distance in fitness_results:
                if success:
                    if best_success_solution is None or edit_distance < best_success_edit_distance:
                        best_success_solution = wolf
                        best_success_adv_text = adv_text
                        best_success_edit_distance = edit_distance
                        print(f"★ Successful attack found! Edit distance: {edit_distance}")
            
            # 7. Update leader wolves
            alpha = fitness_results[0][0]
            beta = fitness_results[1][0] if len(fitness_results) > 1 else None
            delta = fitness_results[2][0] if len(fitness_results) > 2 else None
            
            # 8. Update a parameter and positions
            a = 2 - iteration * (2 / self.max_iterations)  # Linear decrease from 2 to 0
            a_adjusted = a * temperature  # Apply temperature adjustment

            # Adjust a based on stagnation
            if stagnation_count > 5:
                # Increase a to promote exploration when stagnating
                a_adjusted = min(1.8, a * 1.2)
                print(f"Stagnation detected, increasing a value to promote exploration: {a:.2f} → {a_adjusted:.2f}")

            # Use adjusted a value to update positions
            self.wolves = self.update_positions(alpha, beta, delta, a_adjusted)
            
            # 9. Apply crossover every 3 iterations to increase diversity
            if iteration % 3 == 0:
                self.apply_crossover()

            # Apply search space reduction after finding successful attack
            for wolf, adv_text, _, success, edit_distance in fitness_results:
                if success:
                    # Only perform space reduction in exploitation phase (small a)
                    if a_adjusted < 1.0:
                        print("Attempting to reduce search space to improve semantic similarity...")
                        reduced_text = self.reduce_search_space(adv_text.copy(), self.orig_label, a_adjusted)
                        
                        # Compare effects before and after reduction
                        reduced_edit_distance = calculate_edit_distance(self.text, reduced_text)
                        
                        if reduced_edit_distance < edit_distance:
                            # Construct new wolf solution from reduced_text
                            new_wolf = []
                            for i in range(len(self.text)):
                                if self.text[i] != reduced_text[i]:
                                    new_wolf.append((i, self.text[i], reduced_text[i]))
                            
                            # Update best solution
                            best_success_solution = new_wolf
                            best_success_adv_text = reduced_text
                            best_success_edit_distance = reduced_edit_distance
                            print(f"★ Search space reduction successful! Edit distance: {edit_distance} → {reduced_edit_distance}")
                            break
                
            # Status output with a parameter information
            print(f"Iteration {iteration+1}/{self.max_iterations}, Fitness: {best_fitness_so_far:.2f}, " +
                f"Stagnation: {stagnation_count}, Temp: {temperature:.2f}, Mutation: {self.mutation_rate:.3f}, " +
                f"a param: {a_adjusted:.2f}")
                
            # Clean GPU memory every 5 iterations
            if torch.cuda.is_available() and iteration % 5 == 0:
                torch.cuda.empty_cache()
                
            # Trigger Python garbage collection
            if iteration % 3 == 0:
                import gc
                gc.collect()
            
        # 10. Final sprint stage - high-intensity exploration
        if best_success_solution is None:
            print("\n⚡ No attack solution found, performing final sprint exploration...")
            self.mutation_rate = 0.3  # Greatly increase mutation rate
            
            # Execute high-intensity mutations and extra iterations
            for extra_iter in range(5):
                # Apply high-intensity mutations to population
                extra_results = self.extra_exploration()
                
                # Check for successful attacks
                for wolf, adv_text, _, success, edit_distance in extra_results:
                    if success:
                        best_success_solution = wolf
                        best_success_adv_text = adv_text
                        best_success_edit_distance = edit_distance
                        break
                        
                if best_success_solution is not None:
                    print(f"✓ Attack solution found in sprint phase!")
                    break
                    
        # 11. Return results
        if best_success_solution is not None:
            return best_success_adv_text, best_success_solution, True, best_success_edit_distance, self.query_count
        else:
            best_wolf = fitness_results[0][0]
            adv_text = fitness_results[0][1]
            return adv_text, best_wolf, False, fitness_results[0][4], self.query_count

    def initialize_one_wolf(self):
        """Initialize a single wolf - enhanced to match a parameter mechanism"""
        # Randomly select initialization mode
        mode = np.random.choice(['explore', 'exploit', 'balanced'], p=[0.4, 0.3, 0.3])
        
        if not self.valid_positions:
            return []
            
        if mode == 'explore':  # Exploration mode - corresponding to early a large
            num_replacements = min(np.random.randint(2, 4), len(self.valid_positions))
        elif mode == 'exploit':  # Exploitation mode - corresponding to late a small
            num_replacements = min(np.random.randint(1, 2), len(self.valid_positions))
        else:  # Balanced mode
            num_replacements = min(np.random.randint(1, 3), len(self.valid_positions))
        
        positions = np.random.choice(self.valid_positions, num_replacements, replace=False)
        
        # Create replacement scheme
        replacements = []
        for pos in positions:
            candidates = self.candidates_dict[pos]['candidates']
            if candidates:
                replacement = (pos, self.text[pos], np.random.choice(candidates))
                replacements.append(replacement)
        
        return replacements

    def apply_crossover(self):
        """Apply crossover operations to increase population diversity"""
        new_wolves = []
        
        # Randomly select parent wolves for crossover
        for _ in range(max(2, self.wolf_population // 5)):
            parent1_idx = np.random.randint(0, self.wolf_population)
            parent2_idx = np.random.randint(0, self.wolf_population)
            
            if parent1_idx == parent2_idx:
                continue
                
            parent1 = self.wolves[parent1_idx]
            parent2 = self.wolves[parent2_idx]
            
            # Create offspring
            child = []
            all_positions = set()
            
            # Convert parents' replacements by position index
            p1_replacements = {pos: (pos, orig, repl) for pos, orig, repl in parent1}
            p2_replacements = {pos: (pos, orig, repl) for pos, orig, repl in parent2}
            
            # Merge replacement positions from both parents
            all_pos = set(list(p1_replacements.keys()) + list(p2_replacements.keys()))
            
            for pos in all_pos:
                # Randomly select replacement from one parent
                if pos in p1_replacements and pos in p2_replacements:
                    selected = p1_replacements[pos] if np.random.random() < 0.5 else p2_replacements[pos]
                    child.append(selected)
                    all_positions.add(pos)
                elif pos in p1_replacements:
                    child.append(p1_replacements[pos])
                    all_positions.add(pos)
                else:
                    child.append(p2_replacements[pos])
                    all_positions.add(pos)
            
            # Limit replacement count
            if len(child) > self.max_replacements:
                np.random.shuffle(child)
                child = child[:self.max_replacements]
            
            new_wolves.append(child)
        
        # Randomly replace some wolves
        if new_wolves:
            replace_indices = np.random.choice(range(self.wolf_population), 
                                            min(len(new_wolves), self.wolf_population // 5),
                                            replace=False)
            for i, idx in enumerate(replace_indices):
                if i < len(new_wolves):
                    self.wolves[idx] = new_wolves[i]

    def extra_exploration(self):
        """High-intensity exploration for final sprint phase - enhanced a parameter mechanism"""
        # Set large a value to promote global exploration
        a_exploration = 1.8
        
        extra_wolves = []
        for _ in range(10):
            wolf = []
            available_positions = self.valid_positions.copy()
            np.random.shuffle(available_positions)
            
            # More replacements
            num_replacements = min(max(3, int(a_exploration * 2)), 
                                min(self.max_replacements, len(available_positions)))
            
            # Simulate GWO's A and C parameters
            r1_factors = np.random.random(size=num_replacements)
            A_values = 2*a_exploration*r1_factors - a_exploration  # A ranges from [-a*2, a]
            
            for i in range(num_replacements):
                if i < len(available_positions):
                    pos = available_positions[i]
                    candidates = self.candidates_dict[pos]['candidates']
                    if candidates:
                        # Adjust candidate word selection strategy based on A value
                        if abs(A_values[i]) > 1:  # Exploration mode
                            # Random selection
                            replacement = (pos, self.text[pos], np.random.choice(candidates))
                        else:  # Exploitation mode
                            # Could select replacements more likely to match context
                            # Simplified implementation, still random selection
                            replacement = (pos, self.text[pos], np.random.choice(candidates))
                        
                        wolf.append(replacement)
            
            extra_wolves.append(wolf)
        
        # Calculate fitness
        batch_results = self.calculate_fitness_batch(extra_wolves)
        
        # Format conversion
        enhanced_results = []
        for i, (adv_text, fitness, success, edit_distance) in enumerate(batch_results):
            if i < len(extra_wolves):
                enhanced_results.append((extra_wolves[i], adv_text, fitness, success, edit_distance))
        
        return enhanced_results

    def update_positions(self, alpha, beta, delta, a):
        """Update wolf positions - increase actual use of a parameter"""
        new_wolves = []

        for wolf in self.wolves:
            # Skip leader wolves
            if wolf == alpha or wolf == beta or wolf == delta:
                new_wolves.append(wolf)
                continue

            # Build replacement scheme for new wolf
            new_replacements = []
            all_positions = set()
            
            # 1. Dynamically adjust leader wolf inheritance probability based on a parameter
            # Larger a means lower inheritance probability (more exploration)
            # Smaller a means higher inheritance probability (more exploitation)
            alpha_influence = max(0.3, min(0.7, 0.5 - a * 0.1))
            beta_influence = max(0.2, min(0.4, 0.3 - a * 0.05))
            delta_influence = max(0.1, min(0.3, 0.2 - a * 0.05))
            
            for leader, prob in [(alpha, alpha_influence), (beta, beta_influence), (delta, delta_influence)]:
                if not leader:
                    continue

                for pos, orig, repl in leader:
                    if pos in all_positions:
                        continue

                    # Inherit leader wolf's replacement by probability
                    if np.random.random() < prob:
                        new_replacements.append((pos, orig, repl))
                        all_positions.add(pos)

            # 2. Mutation operations - a parameter affects mutation probability
            mutation_prob = self.mutation_rate * (1 + a * 0.5)  # Higher a means higher mutation probability
            
            # 2.1 Add non-stopword replacements
            if np.random.random() < mutation_prob and self.valid_positions:
                available_positions = [p for p in self.valid_positions if p not in all_positions]

                if available_positions:
                    num_to_select = min(1 + int(a), len(available_positions))  # a affects selection quantity
                    
                    for _ in range(num_to_select):
                        if not available_positions:
                            break
                        
                        pos = np.random.choice(available_positions)
                        available_positions.remove(pos)
                    
                        if self.candidates_dict[pos]['candidates']:
                            replacement = np.random.choice(self.candidates_dict[pos]['candidates'])
                            new_replacements.append((pos, self.text[pos], replacement))
                            all_positions.add(pos)

            # 2.2 Add stopword replacement (maximum 1)
            if np.random.random() < mutation_prob * 0.5 and self.stopword_positions:  # Lower probability for stopwords
                available_stopwords = [p for p in self.stopword_positions if p not in all_positions]
                if available_stopwords:
                    pos = np.random.choice(available_stopwords)
                    if self.candidates_dict[pos]['candidates']:
                        replacement = np.random.choice(self.candidates_dict[pos]['candidates'])
                        new_replacements.append((pos, self.text[pos], replacement))
                        all_positions.add(pos)

            # 2.3 Randomly remove replacement - higher a means lower removal probability
            if new_replacements and np.random.random() < self.mutation_rate * (1 - a * 0.3):
                idx_to_remove = np.random.randint(0, len(new_replacements))
                new_replacements.pop(idx_to_remove)

            # 3. Limit maximum replacements
            if len(new_replacements) > self.max_replacements:
                np.random.shuffle(new_replacements)
                new_replacements = new_replacements[:self.max_replacements]

            new_wolves.append(new_replacements)

        return new_wolves

#--------------------------------------------------------
# Module 5: Adversarial Attack
#--------------------------------------------------------

def attack_one_sample(text, target_model, hownet_handler, stop_words, args):
    """Attack a single sample"""
    print(f"\n\n======= Starting attack on sample =======")
    print(f"Original text: {' '.join(text)}")
    
    # Generate candidate words
    print(f"\n1. Generating candidate replacements...")
    candidates_dict = generate_candidates_for_text(
        text,
        hownet_handler,
        stop_words,
        relaxed=args.relaxed_sememe_match,
        top_k=args.top_k_sememe,
        synonym_num=args.synonym_num,
        sim_threshold=args.sememe_similarity_threshold
    )
    
    # Print replaceable word count statistics
    valid_positions = [
        i for i, info in candidates_dict.items()
        if not info['is_stopword'] and len(info['candidates']) > 0
    ]
    print(f"\nReplaceable non-stopword positions: {len(valid_positions)}/{len(text)}")
    
    # Get original prediction label
    print(f"\n2. Getting original prediction label...")
    with torch.no_grad():
        if args.target_model == 'bert':
            orig_probs = target_model([text])
        else:
            orig_probs = target_model([text])
        orig_label = torch.argmax(orig_probs, dim=-1).item()
        probs_np = orig_probs.cpu().numpy()[0]
        print(f"Original prediction label: {orig_label}, prediction probability: {probs_np}")

    print(f"\n3. Initializing semantic similarity model: {args.semantic_sim_model}...")
    # Initialize semantic similarity model
    if args.semantic_sim_model == 'USE':
        sim_predictor = USE(args.USE_cache_path)
    elif args.semantic_sim_model == 'SentenceBERT':
        sim_predictor = SentenceBERT(args.sentence_bert_model_name)
    else:
        raise ValueError("Invalid semantic similarity model specified.")
        
    print(f"\n4. Initializing Discrete Grey Wolf Optimizer...")
    print(f"Parameters: wolf population={args.wolf_population}, max iterations={args.max_iterations}, " + 
          f"mutation rate={args.mutation_rate}, max replacements={args.max_replacements}")
    
    # Initialize Discrete Grey Wolf Optimizer
    optimizer = DGWO(
        text=text,
        candidates_dict=candidates_dict,
        predictor=target_model,
        orig_label=orig_label,
        sim_predictor=sim_predictor,
        wolf_population=args.wolf_population,
        max_iterations=args.max_iterations,
        mutation_rate=args.mutation_rate,
        max_replacements=args.max_replacements,
        sim_threshold=args.sim_score_threshold
    )

    print(f"\n5. Starting optimization...")
    # Execute optimization
    adv_text, best_solution, success, edit_distance, query_count = optimizer.optimize()

    # Clean up resources
    if args.semantic_sim_model == 'USE':
        sim_predictor.close()
    
    # Clean GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n6. Attack results:")
    print(f"Success: {success}")
    print(f"Edit distance: {edit_distance}")
    print(f"Query count: {query_count}")
    
    if best_solution:
        print(f"Word replacements:")
        for pos, original, replacement in best_solution:
            print(f"  Position {pos}: '{original}' -> '{replacement}'")
            
    return adv_text, best_solution, success, orig_label, edit_distance, query_count

#--------------------------------------------------------
# Module 6: Main Program
#--------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser()

    # 1. Sememe related parameters
    sememe_group = parser.add_argument_group("Sememe Settings")
    sememe_group.add_argument("--relaxed_sememe_match", action="store_true", help="Allow partial sememe matching")
    sememe_group.add_argument("--top_k_sememe", type=int, default=3, help="Number of top-K candidate words to keep")
    sememe_group.add_argument("--sememe_similarity_threshold", type=float, default=0.5, help="Sememe similarity threshold")

    # 2. Grey Wolf Optimization parameters
    gwo_group = parser.add_argument_group("GWO Optimization Settings")
    gwo_group.add_argument("--wolf_population", type=int, default=60, help="Grey wolf population size")
    gwo_group.add_argument("--max_iterations", type=int, default=20, help="Maximum optimization iterations")
    gwo_group.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation probability")
    gwo_group.add_argument("--max_replacements", type=int, default=5, help="Maximum word replacements per sample")

    # 3. Stopword parameters
    stopword_group = parser.add_argument_group("Stopword Filtering Settings")
    stopword_group.add_argument("--enable_stopword_filter", action="store_true", default=True, help="Whether to filter stopwords")
    stopword_group.add_argument("--custom_stopwords_path", type=str, default="data/custom_stopwords.txt", help="Custom stopwords file path")
    stopword_group.add_argument("--preserved_stopwords", type=str, default="not,no,never,very,extremely", help="Comma-separated list of stopwords to preserve")

    # 4. Semantic similarity and general parameters
    general_group = parser.add_argument_group("General Settings")
    general_group.add_argument("--sim_score_threshold", default=0.8, type=float, help="Required minimum semantic similarity score")
    general_group.add_argument("--synonym_num", default=50, type=int, help="Number of synonyms to extract")
    general_group.add_argument("--semantic_sim_model", type=str, default="USE", choices=["USE", "SentenceBERT"], help="Semantic similarity model to use")
    general_group.add_argument("--USE_cache_path", type=str, default="cache/USE_cache", help="Universal Sentence Encoder cache path")
    general_group.add_argument("--sentence_bert_model_name", type=str, default="all-mpnet-base-v2", help="Sentence-BERT model name")

    # Other required parameters
    parser.add_argument("--dataset_path", type=str, required=True, default="/data/mr", help="Which dataset to attack")
    parser.add_argument("--nclasses", type=int, default=2, help="How many classes for classification")
    parser.add_argument("--target_model", type=str, default="bert", choices=['wordLSTM', 'bert', 'wordCNN'], help="Target model type")
    parser.add_argument("--target_model_path", type=str, default="/dependencies/models/cnn/mr", help="Pre-trained target model path")
    parser.add_argument("--word_embeddings_path", type=str, default='/dependencies/others/glove.6B.200d.txt', help="Path to word embeddings")
    parser.add_argument("--output_dir", type=str, default='dgwo_adv_results', help="Output directory")
    parser.add_argument("--data_size", type=int, default=1000, help="Data size")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--data_offset", type=int, default=0, help="Dataset starting index")
    parser.add_argument("--gpu_id", type=int, default=0, help="Specified GPU ID to use")
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage instead of GPU')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Handle data offset
    texts, labels = dataloader.read_corpus(args.dataset_path, csvf=False)
    data = list(zip(texts, labels))
    
    # Apply offset and size limits
    start_idx = args.data_offset
    end_idx = min(start_idx + args.data_size, len(data))
    data = data[start_idx:end_idx]
    print(f"Data import finished! Processing samples from index {start_idx} to {end_idx-1}")

    # Load target model
    print("Loading target model...")
    if args.target_model == "wordLSTM":
        model = Model(args.word_embeddings_path, args.target_model, nclasses=args.nclasses)
        model.load_state_dict(torch.load(args.target_model_path))
        model.text_pred = model.predict
    elif args.target_model == "wordCNN":
        model = Model(args.word_embeddings_path, args.target_model, nclasses=args.nclasses)
        model.load_state_dict(torch.load(args.target_model_path))
        model.text_pred = model.predict
    elif args.target_model == "bert":
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
        predictor = model.text_pred
    else:
        raise ValueError("Invalid target model specified.")

    print("Model built!")

    # Initialize OpenHowNet
    print("Initializing OpenHowNet...")
    hownet_handler = initialize_hownet()

    # Get stopwords
    stop_words = get_stopwords(args.custom_stopwords_path, args.preserved_stopwords)

    # Filter data
    filtered_data = []
    for text, label in data:
        with torch.no_grad():
            pred_probs = predictor([text])
            pred_label = torch.argmax(pred_probs, dim=-1).item()
            if pred_label == label:  # Model classifies correctly
                filtered_data.append((text, label))
    
    test_data = filtered_data
    print(f"Correctly classified samples: {len(test_data)}/{len(data)}")

    # Prepare output files with timestamps
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(args.dataset_path)
    
    # Output files for all samples and successful attacks
    formatted_output_file = os.path.join(args.output_dir, f'DGWO_adversaries_{args.target_model}_{dataset_name}_{timestamp}.txt')
    success_output_file = os.path.join(args.output_dir, f'DGWO_success_{args.target_model}_{dataset_name}_{timestamp}.txt')
    
    # Initialize statistics
    total_samples = len(test_data)
    success_count = 0
    results = []
    
    # Execute attacks
    for i, (text, label) in enumerate(tqdm(test_data, desc="Attacking samples")):
        # Run attack
        adv_text, best_solution, success, orig_label, edit_distance, query_count = attack_one_sample(
            text, predictor, hownet_handler, stop_words, args)

        # Save results
        result = {
            'original_text': text,
            'adversarial_text': adv_text,
            'original_label': orig_label,
            'success': success,
            'best_solution': best_solution,
            'edit_distance': edit_distance,
            'query_count': query_count
        }
        results.append(result)

        # Print results
        print(f"Sample {i+1}/{total_samples}:")
        print(f"  Original Text: {' '.join(text)}")
        print(f"  Adversarial Text: {' '.join(adv_text)}")
        print(f"  Original Label: {orig_label}")
        print(f"  Attack Success: {success}")
        print(f"  Edit Distance: {edit_distance}")
        print(f"  Query Count: {query_count}")
        
        # Update counter
        if success:
            success_count += 1
        
        # Write results to file immediately after each sample
        with open(formatted_output_file, 'a', encoding='utf-8') as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            orig_text = ' '.join(text)
            adv_text = ' '.join(adv_text)
            
            # Get adversarial sample label
            with torch.no_grad():
                adv_probs = predictor([adv_text.split()])
                adv_label = torch.argmax(adv_probs, dim=-1).item()
            
            # Write processing time and sample ID
            f.write(f"# Sample ID: {i+1}, Processing time: {current_time}\n")
            f.write(f"orig sent ({orig_label}):\t{orig_text}\n")
            f.write(f"adv sent ({adv_label}):\t{adv_text}\n")
            f.write(f"query count:\t{query_count}\n\n")
            
            # Ensure content is written to disk
            f.flush()
            os.fsync(f.fileno())
        
        # Save successful attacks separately
        if success:
            with open(success_output_file, 'a', encoding='utf-8') as f:
                f.write(f"# Sample ID: {i+1}, Processing time: {current_time}\n")
                f.write(f"orig sent ({orig_label}):\t{orig_text}\n")
                f.write(f"adv sent ({adv_label}):\t{adv_text}\n\n")
                f.flush()
                os.fsync(f.fileno())
        
        # Print statistics every 5 samples
        if (i+1) % 5 == 0:
            current_success_rate = success_count / (i+1)
            print(f"\nProgress: {i+1}/{total_samples}, Attack success rate: {current_success_rate:.4f}")

        # Periodically clean resources
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    # Calculate final attack success rate
    final_success_rate = success_count / total_samples if total_samples > 0 else 0
    print(f"\nAttack completed! Final attack success rate: {final_success_rate:.4f}")
    print(f"All sample results saved to: {formatted_output_file}")
    print(f"Successful attack samples saved to: {success_output_file}")
    
    # Record final results in summary file
    summary_file = os.path.join(args.output_dir, f'DGWO_summary_{args.target_model}_{dataset_name}_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Attack time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Target model: {args.target_model}\n")
        f.write(f"Attack parameters:\n")
        f.write(f"  - Wolf population: {args.wolf_population}\n")
        f.write(f"  - Max iterations: {args.max_iterations}\n")
        f.write(f"  - Mutation rate: {args.mutation_rate}\n")
        f.write(f"Attack results:\n")
        f.write(f"  - Total samples: {total_samples}\n")
        f.write(f"  - Successful attacks: {success_count}\n")
        f.write(f"  - Attack success rate: {final_success_rate:.4f}\n")

if __name__ == '__main__':
    main()
   