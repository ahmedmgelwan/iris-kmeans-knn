from typing import Literal

from pydantic import BaseModel, Field


class IRISFeatures(BaseModel):
    sepal_width: float = Field(..., description="Sepal width of the iris flower", gt=0)
    petal_length: float = Field(..., description="Petal length of the iris flower", gt=0)
    petal_width: float = Field(..., description="Petal width of the iris flower", gt=0)

class IRISTarget(BaseModel):
    class_label: Literal["Iris-setosa", "Iris-versicolor", "Iris-virginica"]