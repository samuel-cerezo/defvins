import numpy as np
import pytest

from trajeval import synthetic


@pytest.fixture(params=["line", "arc", "s_curve"])
def shape(request):
    """The three canonical synthetic paths: straight line, curve and S."""
    return {
        "line": synthetic.straight_line(length=20.0, n=401),
        "arc": synthetic.arc(radius=10.0, angle=np.pi / 2, n=401),
        "s_curve": synthetic.s_curve(length=30.0, amplitude=3.0, n=601),
    }[request.param]
