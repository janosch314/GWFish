import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GWFish",
    version="0.0.1",
    author="Jan Harms",
    author_email="jan.harms@gssi.it",
    description="Gravitational-wave Fisher matrix code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/janosch314/GWFish",
    project_urls={
        "GSSI website": "https://www.gssi.it",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
