from setuptools import setup, find_packages

with open("LICENSE", "r") as file:
    license_content = file.read()

setup(
    name="ssimlossnd",
    version="0.0.1",
    description="PyTorch SSIMLoss with autograd support for 1D, 2D, or 3D inputs, as well as custom convolution operations.",
    author="5o1",
    author_email="assanekowww@gmail.com",
    url="https://github.com/5o1/SSIMLossnd",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True, 
    python_requires=">=3.9",
    license=license_content,
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)