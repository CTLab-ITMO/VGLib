
from transformers import BertTokenizer, BertModel, BertConfig


def update_tokenizer(name_tokenizer='bert-base-uncased', num_token=5):
    name_read = 'READ'
    name_write = 'WRITE'
    special_tokens = ["<START>", "<PATH>", "<END>", "Line", "ARC", "<WRITE>", "QuadraticBezier", "CubicBezier"]
    for tok_ind in range(0, num_token):
        token_read = f"<{name_read}_{tok_ind}>"
        special_tokens.append(token_read)
    tokenizer = BertTokenizer.from_pretrained(name_tokenizer, do_basic_tokenize=False)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer, len(tokenizer)

