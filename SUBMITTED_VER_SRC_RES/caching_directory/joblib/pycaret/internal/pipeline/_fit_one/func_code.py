# first line: 57
def _fit_one(transformer, X=None, y=None, message=None, **fit_params):
    """Fit the data using one transformer."""
    with _print_elapsed_time("Pipeline", message):
        if hasattr(transformer, "fit"):
            args = []
            if "X" in signature(transformer.fit).parameters:
                args.append(X)
            if "y" in signature(transformer.fit).parameters:
                args.append(y)
            transformer.fit(*args, **fit_params)
    return transformer
