# first line: 106
def _full_transform(pipeline: "Pipeline", X, y, **kwargs):
    for _, _, transformer in pipeline._iter(**kwargs):
        X, y = pipeline._memory_transform(transformer, X, y)
    return X, y
