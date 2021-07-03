from transformers import AutoTokenizer, AutoModelWithHeads


def load_model_and_tokenizer(*, model_name: str) -> tuple:
    """
    Loads a pre-trained model from HuggingFace
    :param model_name: string name of the model to load
    :return: pre-trained model and tokenizer
    """
    model = AutoModelWithHeads.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


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
