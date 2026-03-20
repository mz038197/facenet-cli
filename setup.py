from setuptools import find_namespace_packages, setup


setup(
    name="facenet-cli",
    version="1.0.0",
    description="CLI harness for facenet-pytorch workflows.",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    install_requires=[
        "click>=8.1.0",
        "prompt-toolkit>=3.0.0",
        "facenet-pytorch>=2.5.0",
        "opencv-python>=4.8.1.78",
        "matplotlib>=3.10.0",
        "pillow>=10.4",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "fnet=cli_anything.facenet.cli:main",
        ]
    },
    python_requires=">=3.13",
)
