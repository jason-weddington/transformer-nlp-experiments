from transformers import AutoTokenizer, AutoModelWithHeads, RobertaTokenizer, RobertaModelWithHeads, RobertaConfig
from datasets import load_dataset


def loader(*, model_name: str, model_class: callable):
    """
    Loads a pre-trained model or tokenizer from HuggingFace
    :param model_name: string name of the model
    :param model_class: HuggingFace model or that implements from_pretrained() function
    e.g. AutoTokenizer, RobertaTokenizer, AutoModelWithHeads
    :return: pre-trained model and tokenizer
    """
    result = model_class.from_pretrained(model_name)
    return result


def get_model():
    return loader(model_name="roberta-base", model_class=RobertaModelWithHeads)


def get_tokenizer():
    return loader(model_name="roberta-base", model_class=RobertaTokenizer)


def adapt_model(*, model, adapter_name: str, adapter_arch: str):
    """
    Load an adapter and prediction head from Adapter Hub and
    apply it to the passed model.
    :param model: model to adapt
    :param adapter_name: name of the adapter to load
    :param adapter_arch: adapter architecture, see
    https://docs.adapterhub.ml/adapters.html#adapter-architectures
    for more information on architectures.
    """
    adapter = model.load_adapter(adapter_name_or_path=adapter_name, config=adapter_arch)
    model.set_active_adapters(adapter)


def load_data(*, data_set_name: str):
    """
    Loads a dateset from the HuggingFace library of
    public datasets.
    :param data_set_name: name of the dataset to load
    :return: the requested dataset
    """
    return load_dataset(data_set_name)


def encode_data_to_batches(*, data, model_name: str, tokenizer_class: callable):
    """
    Encodes data using the tokenizer
    :param data: data to encode into batches
    :param model_name: name of the HuggingFace model (tokenizer) to use
    :param tokenizer_class: tokenizer class from HuggingFace, e.g. RobertaTokenizer
    :return: tokenized data
    """
    tokenizer = loader(model_name=model_name, model_class=tokenizer_class)

    # inner function to do the encoding
    # from: https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/02_Adapter_Inference.ipynb
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

    data = data.map(encode_batch, batched=True)
    data.rename_column_("label", "labels")
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return data


def get_test_data():
    """
    Dummy function for returning some test data
    just to see if things are working
    :return: test rotten tomatoes data encoded into batches
    """
    dataset = load_dataset("rotten_tomatoes")
    dataset = encode_data_to_batches(data=dataset, model_name="roberta-base", tokenizer_class=RobertaTokenizer)

    return dataset
