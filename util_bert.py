class ImdbPt(Dataset):
    ''' Loads IMDB-pt dataset. 
    
    It will tokenize our inputs and cut-off those that exceed 512 tokens (the pretrained BERT limit)
    '''
    
    def __init__(self, tokenizer, data, cachefile, rebuild=False):
        if os.path.isfile(cachefile) and rebuild is False:
            self.deserialize(cachefile)
        else:
            self.build(tokenizer, data)
            self.serialize(cachefile)
        
    
    def build(self, tokenizer, data):    
        data = data.copy()
    
        tqdm.pandas()
        data['tokenized'] = data['text_pt'].progress_apply(tokenizer.tokenize)
        
        data['input_ids'] = data['tokenized'].apply(
            lambda tokens: tokenizer.build_inputs_with_special_tokens(
                tokenizer.convert_tokens_to_ids(tokens)))
        
        data = data[data['input_ids'].apply(len)<512]
        
        data['labels'] = (data['sentiment'] == 'pos').astype(int)
        
        self.examples = data[['input_ids', 'labels']].to_dict('records')
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return {key: torch.tensor(value) for key, value in self.examples[i].items()}
        else:
            return [{key: torch.tensor(value) for key, value in sample.items()} for sample in self.examples[i]]
     
    def __len__(self):
        return len(self.examples)
    
    def serialize(self, cachefile):
        with open(cachefile, 'wb') as file:
            pickle.dump(self.examples, file)
    
    def deserialize(self, cachefile):
        with open(cachefile, 'rb') as file:
            self.examples = pickle.load(file)