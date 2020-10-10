import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlpug",
    version="0.0.15",
    author="Freddy Snijder",
    author_email="mlpug@visionscapers.com",
    description="A machine learning library agnostic framework for model training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nuhame/mlpug",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorboard',
        # required for tensorboard, else this error occurs : ModuleNotFoundError: No module named 'past'
        # TODO : still required?
        'future'
    ],
    dependency_links=['git+https://github.com/visionscaper/pybase.git'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)