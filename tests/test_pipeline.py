import pytest
import torch

from cobald_parser.pipeline import ConlluTokenClassificationPipeline


@pytest.fixture
def mock_model(mocker):
    """Create a mock model for testing"""
    model = mocker.MagicMock()
    
    # Mock model config
    model.config.vocabulary = {
        "lemma_rule": {
            0: "cut_prefix=0|cut_suffix=0|append_suffix=",
            1: "cut_prefix=0|cut_suffix=1|append_suffix=",
            2: "cut_prefix=0|cut_suffix=2|append_suffix=be"
        },
        "joint_feats": {
            0: "NOUN#N#Number=Sing",
            1: "VERB#V#Tense=Pres"
        },
        "ud_deprel": {0: "root", 1: "nsubj"},
        "eud_deprel": {0: "root", 1: "conj"},
        "misc": {0: "_", 1: "SpaceAfter=No"},
        "deepslot": {0: "ACT", 1: "PAT"},
        "semclass": {0: "person", 1: "action"}
    }

    # Mock model output
    output = {
        "words": [
            ['Congratulations', '!'],
            ['Very', 'strange', '#NULL', '.']
        ],
        "lemma_rules": torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 0, 0]]
        ),
        "joint_feats": torch.tensor(
            [[0, 1, 0, 0],
             [1, 1, 0, 1]]
        ),
        "deps_ud": torch.tensor(
            [[0, 0, 0, 0],
             [0, 0, 1, 1],
             [1, 1, 0, 1],
             [1, 1, 1, 0],
             [1, 1, 2, 1],
             [1, 0, 3, 1]]
        ),
        "deps_eud": torch.tensor(
            [[0, 0, 1, 1],
             [1, 1, 0, 1],
             [1, 2, 1, 1],
             [1, 2, 2, 0],
             [1, 0, 3, 1],
             [1, 1, 3, 1]]
        ),
        "miscs": torch.tensor(
            [[1, 0, 0, 0],
             [1, 1, 1, 0]]
        ),
        "deepslots": torch.tensor(
            [[1, 0, 0, 0],
             [1, 1, 1, 0]]
        ),
        "semclasses": torch.tensor(
            [[0, 1, 1, 1],
             [0, 0, 0, 1]]
        )
    }
    model.return_value = output
    # Required by Pipeline __init__ method.
    model.hf_device_map = {"model": 0}
    return model


@pytest.fixture
def pipeline(mock_model):
    """Create the pipeline for testing"""
    return ConlluTokenClassificationPipeline(
        model=mock_model,
        language='english',
        framework='pt'
    )


def test_forward_with_proper_input(pipeline, mock_model):
    """Test _forward method with proper input structure"""
    # Setup
    model_inputs = {
        "words": [["This", "is", "a", "test", "."]],
        "texts": ["This is a test."]
    }
    
    result = pipeline._forward(model_inputs)
    
    mock_model.assert_called_once_with(**model_inputs, inference_mode=True)
    assert result == mock_model.return_value


def test_preprocess(pipeline):
    """Test preprocess method"""
    text = "Congratulations! Now everybody's doing reality shows."
    
    result = pipeline.preprocess(text)
    
    assert result["words"] == [
        ['Congratulations', '!'],
        ['Now', 'everybody', "'s", 'doing', 'reality', 'shows', '.']
    ]


def test_preprocess_invalid_input(pipeline):
    """Test preprocess method with invalid input"""
    with pytest.raises(ValueError, match="pipeline input must be string"):
        pipeline.preprocess(123)
    
    with pytest.raises(ValueError, match="pipeline input must be string"):
        pipeline.preprocess(["This is one text", "This is another text"])


def test_enumerate_words_standard(pipeline):
    """Test enumeration of words with standard texts"""
    words = ["Hello", "world", "this", "is", "a", "test"]
    
    result = pipeline._enumerate_words(words)
    
    assert result == ["1", "2", "3", "4", "5", "6"]


def test_enumerate_words_with_null(pipeline):
    """Test enumeration of words with #NULL tokens"""
    words = ["#NULL", "Hello", "#NULL", "#NULL", "world", "#NULL"]
    
    result = pipeline._enumerate_words(words)
    
    assert result == ["0.1", "1", "1.1", "1.2", "2", "2.1"]


def test_decode_sentence(pipeline):
    """Test _decode_sentence method"""

    text = "This is test"
    words = ["This", "is", "test"]
    lemma_rule_ids = [0, 2, 0]
    joint_feats_ids = [0, 1, 1]
    deps_ud = [[0, 1, 1], [1, 0, 1]] # (from, to, rel)
    deps_eud = [[2, 2, 0]]
    misc_ids = [0, 0, 1]
    deepslot_ids = [1, 0, 1]
    semclass_ids = [1, 1, 0]

    result = pipeline._decode_sentence(
        text=text,
        words=words,
        lemma_rule_ids=lemma_rule_ids,
        joint_feats_ids=joint_feats_ids,
        deps_ud=deps_ud,
        deps_eud=deps_eud,
        misc_ids=misc_ids,
        deepslot_ids=deepslot_ids,
        semclass_ids=semclass_ids
    )

    assert result["text"] == "This is test"
    assert result["ids"] == ["1", "2", "3"]
    assert result["words"] == ["This", "is", "test"]
    assert result["lemmas"] == ["This", "be", "test"]
    assert result["upos"] == ["NOUN", "VERB", "VERB"]
    assert result["xpos"] == ["N", "V", "V"]
    assert result["feats"] == ["Number=Sing", "Tense=Pres", "Tense=Pres"]
    assert result["deps_ud"] == [('1', '2', "nsubj"), ('2', '1', "nsubj")]
    assert result["deps_eud"] == [('0', '3', "root")]
    assert result["miscs"] == ["_", "_", "SpaceAfter=No"]
    assert result["deepslots"] == ["PAT", "ACT", "PAT"]
    assert result["semclasses"] == ["action", "action", "person"]


def test_pipeline(pipeline):
    """Test whole pipeline"""

    result = pipeline("Congratulations! Very strange.")

    assert len(result) == 2
    sentence1, sentence2 = result

    assert sentence1["text"] == "Congratulations!"
    assert sentence1["ids"] == ["1", "2"]
    assert sentence1["words"] == ["Congratulations", "!"]
    assert sentence1["lemmas"] == ["Congratulation", "!"]
    assert sentence1["upos"] == ["NOUN", "VERB"]
    assert sentence1["xpos"] == ["N", "V"]
    assert sentence1["feats"] == ["Number=Sing", "Tense=Pres"]
    assert sentence1["deps_ud"] == [
        ('0', '1',  "root"),
        ('1', '2', "nsubj")
    ]
    assert sentence1["deps_eud"] == [('1', '2', "conj")]
    assert sentence1["miscs"] == ["SpaceAfter=No", "_"]
    assert sentence1["deepslots"] == ["PAT", "ACT"]
    assert sentence1["semclasses"] == ["person", "action"]

    assert sentence2["text"] == "Very strange."
    assert sentence2["ids"] == ["1", "2", "2.1", "3"]
    assert sentence2["words"] == ['Very', 'strange', '#NULL', '.']
    assert sentence2["lemmas"] == ['Very', 'strange', '#NULL', '.']
    assert sentence2["upos"] == ["VERB", "VERB", "NOUN", "VERB"]
    assert sentence2["xpos"] == ["V", "V", "N", "V"]
    assert sentence2["feats"] == [
        "Tense=Pres",
        "Tense=Pres",
        "Number=Sing",
        "Tense=Pres"
    ]
    assert sentence2["deps_ud"] == [
        ('2',   '1', "nsubj"),
        ('0',   '2',  "root"),
        ('2', '2.1', "nsubj"),
        ('1',   '3', "nsubj")
    ]
    assert sentence2["deps_eud"] == [
        (  '2',   '1', "conj"),
        ('2.1',   '2', "conj"),
        (  '0', '2.1', "root"),
        (  '1',   '3', "conj"),
        (  '2',   '3', "conj"),
    ]
    assert sentence2["miscs"] == [
        "SpaceAfter=No",
        "SpaceAfter=No",
        "SpaceAfter=No",
        "_"
    ]
    assert sentence2["deepslots"] == ["PAT", "PAT", "PAT", "ACT"]
    assert sentence2["semclasses"] == ["person", "person", "person", "action"]


def test_pipeline_conllu(pipeline):
    """Test pipeline output formatted as CoNLL-U"""

    text = "Congratulations! Very strange."
    result = pipeline(text, conllu=True)
    expected = (
        "# text = Congratulations!\n"
        "1\tCongratulations\tCongratulation\tNOUN\tN\tNumber=Sing\t0\troot\t_\tSpaceAfter=No\tPAT\tperson\n"
        "2\t!\t!\tVERB\tV\tTense=Pres\t1\tnsubj\t1:conj\t_\tACT\taction\n"
        "\n"
        "# text = Very strange.\n"
        "1\tVery\tVery\tVERB\tV\tTense=Pres\t2\tnsubj\t2:conj\tSpaceAfter=No\tPAT\tperson\n"
        "2\tstrange\tstrange\tVERB\tV\tTense=Pres\t0\troot\t2.1:conj\tSpaceAfter=No\tPAT\tperson\n"
        "2.1\t#NULL\t#NULL\tNOUN\tN\tNumber=Sing\t2\tnsubj\t0:root\tSpaceAfter=No\tPAT\tperson\n"
        "3\t.\t.\tVERB\tV\tTense=Pres\t1\tnsubj\t1:conj|2:conj\t_\tACT\taction"
    )
    assert result == expected