import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GURA",
    version="0.0.1",
    author="toyz's dog",
    author_email="b08902126@csie.ntu.edu.tw",
    description="HEAR 2021",
    long_description=long_description,
    url="https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU.git",
    packages=setuptools.find_packages(),
    install_requires=[
		"transformers==4.11.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
