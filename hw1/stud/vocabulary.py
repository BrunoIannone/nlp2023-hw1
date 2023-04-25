from typing import List
import utils
class Vocabulary():
    def __init__(self,sentences: List[List[str]], labels: List[List[str]]):

        self.token_vocabulary = self.build_tokens_vocabulary(sentences)
        
        self.word_to_idx = self.token_vocabulary["word_to_idx"]
        self.idx_to_word = self.token_vocabulary["idx_to_word"]

        self.labels_vocabulary = self.build_labels_vocabulary(labels)

        self.labels_to_idx = self.labels_vocabulary["labels_to_idx"]
        self.idx_to_labels = self.labels_vocabulary["idx_to_labels"]





    def build_tokens_vocabulary(self,sentences:List[List[str]]):  
        """Create a vocabulary from tokens in sentences with padding having index -1

        Args:
            sentences (List[List[str]]): list of list of strings

        Returns:
            dictionary: Returns a dictionary with the structure {token:index} 
        """
        word_to_idx = {}
        idx_to_word = {}
        special_characters = "!@#$%^&*()-+?_=,<>/"

        idx = 0

        word_to_idx["<pad>"] = idx
        idx_to_word[idx] = "<pad>"

        idx+=1
        word_to_idx["<unk>"] = idx
        idx_to_word[idx] = "<unk>"

        idx +=1
        word_to_idx["<num>"]=idx
        idx_to_word[idx] = "<num>"

        idx+=1
        for sentences_list in sentences:
            for token in sentences_list:
                if token.isnumeric():
                    continue

                elif token.lower() not in word_to_idx and not any(character in special_characters for character in token.lower()):
                    word_to_idx[token.lower()] = idx
                    idx_to_word[idx] = token.lower()
                    idx += 1

        return {
            "word_to_idx" : word_to_idx,
            "idx_to_word" : idx_to_word
        }


            
    def build_labels_vocabulary(self,sentences_labels: List[List[str]]):
        """Converts labels in integral indexes with padding having the greater index

        Args:
            sentences_labels (List[List[str]]): List of list of strings (labels)
            

        Returns:
            dictionary: Returns a dictionary with the structure {label:index}
        """
        labels_to_idx = {}
        idx_to_labels = {}
        idx = 0
        for labels_list in sentences_labels:
            for label in labels_list:
                if label not in labels_to_idx:
                    labels_to_idx[label] = idx
                    idx_to_labels[idx] = label
                    idx += 1
        
        labels_to_idx["<pad>"] = idx
        idx_to_labels[idx] = "<pad>"

        return {
            "labels_to_idx": labels_to_idx,
            "idx_to_labels": idx_to_labels
        }