import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlpug",
    version="0.2.0",
    author="Freddy Snijder",
    author_email="mlpug@visionscapers.com",
    description="MLPug is a library for training and evaluating Machine Learning (ML) models, "
                "able to use different ML libraries as backends.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nuhame/mlpug",
    packages=setuptools.find_packages(),
    install_requires=[
        'visionscaper-pybase',
        'tensorboardX'
    ],
    dependency_links=['git+https://github.com/visionscaper/pybase.git'],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)