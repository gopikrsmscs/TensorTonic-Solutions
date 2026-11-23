class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.vocab_size+=4
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token

        self.word_to_id[self.unk_token] = 1
        self.id_to_word[1] = self.unk_token

        self.word_to_id[self.bos_token] = 2
        self.id_to_word[2] = self.bos_token

        self.word_to_id[self.eos_token] = 3
        self.id_to_word[3] = self.eos_token
        words = set(word.lower() for s in texts for word in s.split())
        texts = sorted(words)   # sorted unique words
        for text in texts:
            self.word_to_id[text] = self.vocab_size
            self.id_to_word[self.vocab_size] = text            
            self.vocab_size += 1
        
        pass
        #print(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        #print(text)
        words = text.lower().split()
        integer_list = []
        for word in words:
            if word in self.word_to_id:
                integer_list.append(self.word_to_id[word])
            else:
                integer_list.append(1)
            
        
        return integer_list

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        texts = []
        for id in ids:
            if id in self.id_to_word:
                texts.append(self.id_to_word[id])
            else:
                 texts.append(self.id_to_word[1])
                
        return ' '.join(texts)