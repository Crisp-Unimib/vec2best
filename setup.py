# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="vec2best",
    version="1.1.0",
    description="A Unified Framework for Intrinsic Evaluation of Word-Embedding Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anna Giabelli",
    author_email="anna.giabelli@unimib.it",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    package_data={'vec2best': ['8-8-8/*', '8-8-8/8-8-8_Dataset/*', 'wiki-sem-500/*', 'wiki-sem-500/wiki-sem-500/en/*', 'wiki-sem-500/src/*', 'wiki-sem-500/src/lib/polyglot/*', 'wiki-sem-500/src/lib/polyglot/mapping/*'],},
    include_package_data=True,
    install_requires=["six", "scikit-learn==0.22.1", "word-embeddings-benchmarks==0.0.1"]
)
