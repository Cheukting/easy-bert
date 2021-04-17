from easybert.bert import Bert

def test_load_from_hub():

    bert = Bert("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")
    x = bert.embed("A sequence")
    y = bert.embed(["Multiple", "Sequences"])
    assert len(x) == 768
    assert len(y) == 2
    assert len(y[0]) == 768
