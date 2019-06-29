from pytorch_pretrained_bert.tokenization_xlnet import XLNetTokenizer
from pytorch_pretrained_bert.modeling_xlnet import XLNetForClassification
tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")

#create the pretrained model
model1 = XLNetForClassification.from_pretrained("xlnet-large-cased", clf_dropout=0.1, n_class=1)
