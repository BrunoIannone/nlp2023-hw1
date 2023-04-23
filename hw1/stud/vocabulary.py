from typing import List
import utils
class Vocabulary():
    def __init__(self,sentences: List[List[str]], labels: List[List[str]]):

        self.token_vocabulary = self.build_tokens_vocabulary(sentences)
        
        self.word_to_ix = self.token_vocabulary["word_to_ix"]
        self.ix_to_word = self.token_vocabulary["ix_to_word"]

        self.tag_vocabulary = self.build_labels_vocabulary(labels)

        self.tag_to_ix = self.tag_vocabulary["tag_to_ix"]
        self.ix_to_tag = self.tag_vocabulary["ix_to_tag"]





    def build_tokens_vocabulary(self,sentences:List[List[str]]):  
        """Create a vocabulary from tokens in sentences with padding having index -1

        Args:
            sentences (List[List[str]]): list of list of strings

        Returns:
            dictionary: Returns a dictionary with the structure {token:index} 
        """
        word_to_ix = {}
        ix_to_word = {}
        special_characters = "!@#$%^&*()-+?_=,<>/"

        idx = 0

        word_to_ix["<pad>"] = idx
        ix_to_word[idx] = "<pad>"

        idx+=1
        word_to_ix["<unk>"] = idx
        ix_to_word[idx] = "<unk>"

        idx +=1
        word_to_ix["<num>"]=idx
        ix_to_word[idx] = "<num>"

        idx+=1
        for sentences_list in sentences:
            for token in sentences_list:
                if token.isnumeric():
                    continue

                elif token.lower() not in word_to_ix and not any(character in special_characters for character in token.lower()):
                    word_to_ix[token.lower()] = idx
                    ix_to_word[idx] = token.lower()
                    idx += 1

        return {
            "word_to_ix" : word_to_ix,
            "ix_to_word" : ix_to_word
        }


            
    def build_labels_vocabulary(self,sentences_labels: List[List[str]]):
        """Converts labels in integral indexes with padding having the greater index

        Args:
            sentences_labels (List[List[str]]): List of list of strings (labels)
            

        Returns:
            dictionary: Returns a dictionary with the structure {label:index}
        """
        tag_to_ix = {}
        ix_to_tag = {}
        idx = 0
        for labels_list in sentences_labels:
            for label in labels_list:
                if label not in tag_to_ix:
                    tag_to_ix[label] = idx
                    ix_to_tag[idx] = label
                    idx += 1
        
        tag_to_ix["<pad>"] = idx
        ix_to_tag[idx] = "<pad>"

        return {
            "tag_to_ix": tag_to_ix,
            "ix_to_tag": ix_to_tag
        }