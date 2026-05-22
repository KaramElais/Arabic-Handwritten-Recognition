from torchvision import transforms
import config


def get_train_transform(use_augmentation=True):

    transform_list = [

        transforms.Grayscale(num_output_channels=1),

        transforms.Resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE)
        ),
    ]

    if use_augmentation:

        transform_list.extend([

            transforms.RandomRotation(
                degrees=config.ROTATION_DEGREES
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=config.TRANSLATE,
                scale=config.SCALE,
                shear=config.SHEAR_DEGREES
            ),
        ])

    transform_list.extend([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])

    return transforms.Compose(transform_list)


def get_eval_transform():

    return transforms.Compose([

        transforms.Grayscale(num_output_channels=1),

        transforms.Resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])