# first line: 70
def _transform_one(transformer, X=None, y=None):
    """Transform the data using one transformer."""
    args = []
    if "X" in signature(transformer.transform).parameters:
        args.append(X)
    if "y" in signature(transformer.transform).parameters:
        args.append(y)
    output = transformer.transform(*args)

    if isinstance(output, tuple):
        X, y = output[0], output[1]
    else:
        if len(output.shape) > 1:
            X, y = output, y  # Only X
        else:
            X, y = X, output  # Only y

    return X, y
