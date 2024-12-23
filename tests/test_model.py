import pytest
import torch
from deepforest import model
from deepforest import get_data
import pandas as pd
import os
import numpy as np
from torchvision import transforms
import cv2


# The model object is architecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.Model(config)


@pytest.fixture()
def crop_model():
    crop_model = model.CropModel(num_classes=2)

    return crop_model


def test_crop_model(crop_model):
    # Test forward pass
    x = torch.rand(4, 3, 224, 224)
    output = crop_model.forward(x)
    assert output.shape == (4, 2)

    # Test training step
    batch = (x, torch.tensor([True, False, True, False]).long())  # Convert to long
    loss = crop_model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)

    # Test validation step
    val_batch = (x, torch.tensor([True, False, True, False]).long())  # Convert to long
    val_loss = crop_model.validation_step(val_batch, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)

def test_crop_model_train(crop_model, tmpdir):
    df = pd.read_csv(get_data("testfile_multi.csv"))
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    root_dir = os.path.dirname(get_data("SOAP_061.png"))
    images = df.image_path.values
    crop_model.write_crops(boxes=boxes,
                           labels=df.label.values,
                           root_dir=root_dir,
                           images=images,
                           savedir=tmpdir)

    # Create a trainer
    crop_model.create_trainer(fast_dev_run=True)
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)

    # Test training dataloader
    train_loader = crop_model.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)

    # Test validation dataloader
    val_loader = crop_model.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    crop_model.trainer.fit(crop_model)
    crop_model.trainer.validate(crop_model)


def test_crop_model_custom_transform():
    # Create a dummy instance of CropModel
    crop_model = model.CropModel(num_classes=2)

    def custom_transform(self, augment):
        data_transforms = []
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(self.normalize)
        # Add transforms here
        data_transforms.append(transforms.Resize([300, 300]))
        if augment:
            data_transforms.append(transforms.RandomHorizontalFlip(0.5))
        return transforms.Compose(data_transforms)

    # Test custom transform
    x = torch.rand(4, 3, 300, 300)
    crop_model.get_transform = custom_transform
    output = crop_model.forward(x)
    assert output.shape == (4, 2)


def test_crop_model_output_labels(crop_model):
    # Test that the model outputs class indices 0 or 1
    x = torch.rand(4, 3, 224, 224)
    output = crop_model.forward(x)
    predicted_labels = torch.argmax(output, dim=1)
    assert all(label in [0, 1] for label in predicted_labels)
