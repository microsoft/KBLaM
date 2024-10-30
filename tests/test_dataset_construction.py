from kblam.utils.data_utils import DataPoint, Entity, save_entity, load_entities
import numpy as np

def test_dataset_QA():
    dataset_path = "tests/test_dataset.json"
    dataset = load_entities(dataset_path)

    assert len(dataset) == 2
    assert isinstance(dataset[0], dict)
    assert isinstance(dataset[1], dict)
