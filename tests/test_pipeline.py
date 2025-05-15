import pytest
import torch

from cobald_parser.pipeline import ConlluTokenClassificationPipeline


@pytest.fixture
def mock_model(mocker):
    """Create a mock model for testing"""
    model = mocker.MagicMock()
    
    # Mock model config
    model.config = mocker.MagicMock(
        id2lemma_rule = {
            0: "cut_prefix=0|cut_suffix=0|append_suffix=",
            1: "cut_prefix=0|cut_suffix=1|append_suffix=",
            2: "cut_prefix=0|cut_suffix=2|append_suffix=be"
        },
        id2morph_feats = {
            0: "NOUN#N#Number=Sing",
            1: "VERB#V#Tense=Pres"
        },
        id2rel_ud={0: "root", 1: "nsubj"},
        id2rel_eud={0: "root", 1: "conj"},
        id2misc={0: "_", 1: "SpaceAfter=No"},
        id2deepslot={0: "ACT", 1: "PAT"},
        id2semclass={0: "person", 1: "action"}
    )

    # Mock model output
    output = mocker.MagicMock(
        words=[
            ['Congratulations', '!'],
            ['Very', 'strange', '#NULL', '.']
        ],
        lemma_rules=torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 0, 0]]
        ),
        morph_feats=torch.tensor(
            [[0, 1, 0, 0],
             [1, 1, 0, 1]]
        ),
        deps_ud=torch.tensor(
            [[0, 0, 0, 0],
             [0, 1, 0, 1],
             [1, 0, 1, 1],
             [1, 1, 1, 0],
             [1, 2, 1, 1],
             [1, 3, 0, 1]]
        ),
        deps_eud=torch.tensor(
            [[0, 1, 0, 1],
             [1, 0, 1, 1],
             [1, 1, 2, 1],
             [1, 2, 2, 0],
             [1, 3, 0, 1],
             [1, 3, 1, 1]]
        ),
        miscs=torch.tensor(
            [[1, 0, 0, 0],
             [1, 1, 1, 0]]
        ),
        deepslots=torch.tensor(
            [[1, 0, 0, 0],
             [1, 1, 1, 0]]
        ),
        semclasses=torch.tensor(
            [[0, 1, 1, 1],
             [0, 0, 0, 1]]
        )
    )
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


def test_postprocess_sentence(pipeline):
    """Test _postprocess_sentence method"""

    text = "This is test"
    words = ["This", "is", "test"]
    lemma_rule_ids = [0, 2, 0]
    morph_feats_ids = [0, 1, 1]
    deps_ud = [[0, 1, 1], [1, 0, 1]] # (from, to, rel)
    deps_eud = [[2, 2, 0]]
    misc_ids = [0, 0, 1]
    deepslot_ids = [1, 0, 1]
    semclass_ids = [1, 1, 0]

    result = pipeline._postprocess_sentence(
        text=text,
        words=words,
        lemma_rule_ids=lemma_rule_ids,
        morph_feats_ids=morph_feats_ids,
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
    assert result["deps_eud"] == [('3', '0', "root")]
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
        ('1', '0',  "root"),
        ('2', '1', "nsubj")
    ]
    assert sentence1["deps_eud"] == [('2', '1', "conj")]
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
        (  '1', '2', "nsubj"),
        (  '2', '0',  "root"),
        ('2.1', '2', "nsubj"),
        (  '3', '1', "nsubj")
    ]
    assert sentence2["deps_eud"] == [
        (  '1',   '2', "conj"),
        (  '2', '2.1', "conj"),
        ('2.1',   '0', "root"),
        (  '3',   '1', "conj"),
        (  '3',   '2', "conj"),
    ]
    assert sentence2["miscs"] == [
        "SpaceAfter=No",
        "SpaceAfter=No",
        "SpaceAfter=No",
        "_"
    ]
    assert sentence2["deepslots"] == ["PAT", "PAT", "PAT", "ACT"]
    assert sentence2["semclasses"] == ["person", "person", "person", "action"]