import torch
import re
from transformers import pipeline,BertTokenizer, BertModel
from sklearn.decomposition import PCA
from transformers import logging



def clean_text(text):
    r"""
    Remove non alphanumeric characters from the text
    @param text: string 
    """
    
    text = re.sub("'", "", text)
    text = re.sub("(\\W+)", " ", text)
    return text

def get_bert_embedding(text, model = 'bert-base-uncased'):
    logging.set_verbosity_warning()
    # text = "An ionophorous, polyether antibiotic from Streptomyces chartreusensis. It binds and transports CALCIUM and other divalent cations across membranes and uncouples oxidative phosphorylation while inhibiting ATPase of rat liver mitochondria. The substance is used mostly as a biochemical tool to study the role of divalent cations in various biological systems."

    # sentences = text.split('. ')

    # marked_text = "[CLS] " + text + " [SEP]"
    text = clean_text(text)

    # Tokenize our sentence with the BERT tokenizer.
    # 
    tokenizer = BertTokenizer.from_pretrained(model,do_lower_case=True)
                                            
    encoded_data = tokenizer.batch_encode_plus(
        text, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt'
    )

    # print(f'Encoded Text: {encoded_data}')

    # tokenized_text = tokenizer.tokenize(encoded_data)

    # Print out the tokens. 
    # print (encoded_data)

    indexed_tokens = tokenizer.convert_tokens_to_ids(encoded_data)
    #print(indexed_tokens)

    # Display the words with their indeces.
    # for tup in zip(encoded_data, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))


    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(encoded_data)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(model,
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    # print(model.eval())


    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
        # print(hidden_states)

        # print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        # layer_i = 0

        # print ("Number of batches:", len(hidden_states[layer_i]))
        # batch_i = 0

        # print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        # token_i = 0

        # print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))


        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()

        # Swap dimensions 0 and 1. “layers” and “tokens”
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()

        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            
            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)

        # print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # print("token vecs: ", token_vecs.size())
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        # print("sentence embedding: ", sentence_embedding.size())
        # pca = PCA(n_components=200)
        m = torch.nn.Linear(768, 200)
        sentence_reduced_dimensions_embedding = m(sentence_embedding)
        # sentence_reduced_dimensions = pca.fit(token_vecs)
        # sentence_reduced_dimensions = models.Dense(in_features=sentence_embedding.tolist(), out_features=200, activation_function=torch.nn.Tanh())

        # print ("Our final sentence embedding vector of shape:", sentence_reduced_dimensions)
        #print ("Our final sentence embedding vector of shape after reduction:", sentence_reduced_dimensions.size())

    return sentence_reduced_dimensions_embedding


# get_bert_embedding()