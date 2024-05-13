import numpy as np

import krippendorff
import pandas as pd


def main():
    print("Example from https://en.wikipedia.org/wiki/Krippendorff's_Alpha")
    print()

    """
    Example of data layout:

    

    """

    # TP
    reliability_data_str = (
        "1    1    1    1   1",  # coder A
        "1    1    1    1   1",  # coder B
        "1    1    1    1   1",  # coder D
        "1    1    1    1   1",  # coder E
        "1    1    1    1   2",  # coder F
    )

    print("\n".join(reliability_data_str))
    print()

    reliability_data = [
        [np.nan if v == "*" else int(v) for v in coder.split()]
        for coder in reliability_data_str
    ]
    print(reliability_data)
    for row in reliability_data:
        print(row)

    print(
        "Krippendorff's alpha for nominal metric: ",
        krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="ordinal"
        ),
    )
    print(
        "Krippendorff's alpha for interval metric: ",
        krippendorff.alpha(reliability_data=reliability_data),
    )

    print()
    print()
    print("From value counts:")
    print()
    value_counts = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 2, 1],
            [0, 0, 0, 3],
            [1, 0, 1, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    print(value_counts)
    print(
        "Krippendorff's alpha for nominal metric: ",
        krippendorff.alpha(value_counts=value_counts, level_of_measurement="nominal"),
    )
    print(
        "Krippendorff's alpha for interval metric: ",
        krippendorff.alpha(value_counts=value_counts),
    )


if __name__ == "__main__":
    main()
