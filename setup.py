
import setuptools


setuptools.setup(
    name="ginkgo",
    version="0.0.1",
    description="Ginkgo: Toy Jets Shower Generator for Particle Physics",
    url="https://github.com/SebastianMacaluso/ginkgo",
    author="Sebastian Macaluso, Kyle Cranmer, Duccio Pappadopulo",
    author_email="seb.macaluso@nyu.edu",
    license="MIT",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

