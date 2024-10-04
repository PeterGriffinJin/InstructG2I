from setuptools import setup, find_packages

setup(
    name="instructg2i",  # Name of the package
    version="0.1",  # Initial version
    description="A package for InstructG2I project",  # Brief description
    long_description=open('README.md').read(),  # Description from your README
    long_description_content_type="text/markdown",  # Set this for markdown files
    url="https://github.com/PeterGriffinJin/InstructG2I",  # Your package URL
    author="Bowen Jin",
    author_email="bowenj4@illinois.edu",
    license="MIT",  # License type (MIT, GPL, etc.)
    packages=find_packages(),  # Automatically find and include packages
    install_requires=[
        'torch==2.0.1',
        'torchvision==0.15.2',
        'torchaudio==2.0.2',
        'diffusers==0.27.0',
        'transformers==4.37.2',
        'accelerate',
        'datasets',
        'wandb',
        'jupyter',
        'torchmetrics[image]'
    ],  # List of dependencies with versions
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Set Python version to 3.10
)
