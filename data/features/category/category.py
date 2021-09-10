from typing import Any
from layer import Dataset
import numpy as np


def build_feature(sdf: Dataset("catsdogs")) -> Any:
    df = sdf.to_pandas()
    df = df.sample(100, random_state=1)
    df = df[df['path'] != 'single_prediction']
    filenames = list(df['path'])
    categories = []

    for filename in filenames:

        category = filename.split('/')[1]
        if category == 'dogs':
            categories.append(1)
        else:
            categories.append(0)

    df['category'] = np.array(categories)

    feature_data = df[["category", "id"]]
    return feature_data
