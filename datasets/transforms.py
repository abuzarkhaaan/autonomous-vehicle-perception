class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms or [])

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
